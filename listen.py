"""Always-on microphone with voice-activity detection.

Runs Silero VAD on the CPU (~0.35 ms per 32 ms frame, ~1% of one core, and no
GPU contention with CSM). Audio comes from arecord as raw PCM.

Deliberately does NOT import the `silero_vad` python package - that pulls in
torchaudio, and pip resolves a CUDA build of it which breaks this ROCm
environment. Only the bundled TorchScript model is loaded, on CPU.

The bundled .onnx exports were tried first and rejected: their `stateN` output
does not round-trip as the next call's `state` input, so carried state corrupts
the LSTM (speech scored 0.03 instead of 1.00). The .jit model manages its own
state and is correct.

Emits a complete utterance once speech has been followed by `silence_s` of
quiet. A pre-roll buffer keeps the audio just before speech was detected, so
the first consonant isn't clipped.

Headphones are assumed: the mic never hears her, so the listener can stay live
while she speaks, which is what makes voice barge-in possible.
"""
import collections
import math
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

import torch

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__))),
    "silero_vad", "data", "silero_vad.jit",
)

RATE = 16000
FRAME = 512                  # samples; Silero's required window at 16 kHz (32 ms)
SPEECH_ON = 0.6              # prob to call it speech (high: avoids fan/keyboard triggers)
SPEECH_OFF = 0.35            # hysteresis - lower bar to stay in speech
PREROLL_FRAMES = 10          # ~320 ms kept before onset so the first word survives
# The wait for the speaker to be done can be long (people pause mid-thought), but
# that whole pause shouldn't be shipped to the audio encoder - keep a short,
# natural-sounding tail and drop the rest.
KEEP_TAIL_FRAMES = 12        # ~380 ms of trailing silence retained

# A mic with too much gain rails at full scale, and a saturated signal scores as
# speech on any VAD - which produces an endless stream of phantom "utterances".
# Frames that are mostly clipped are rejected outright, and the noise floor is
# checked at startup so the cause is reported instead of silently spiralling.
CLIP_LEVEL = 0.985           # |sample| above this counts as clipped
CLIP_FRACTION = 0.02         # >2% clipped samples in a frame -> not usable speech
NOISE_FLOOR_WARN = 0.15      # idle RMS above this means the input gain is far too high
MAX_CAPTURE_RESTARTS = 5     # give up (and say so) rather than respawn forever
RESTART_BACKOFF_S = 0.5


class VoiceListener:
    """Continuous mic + VAD. Utterances land on `.utterances`; on_speech_start
    fires the instant speech begins (used for barge-in)."""

    def __init__(self, silence_s=4.0, min_speech_s=0.35, max_utterance_s=60.0,
                 on_speech_start=None, device=None):
        self.silence_frames = max(1, int(silence_s * RATE / FRAME))
        self.min_speech_frames = max(1, int(min_speech_s * RATE / FRAME))
        self.max_frames = int(max_utterance_s * RATE / FRAME)
        self.on_speech_start = on_speech_start
        self.device = device
        self.utterances = queue.Queue()
        self.enabled = threading.Event()
        self.enabled.set()
        self._stop = threading.Event()
        self._proc = None
        self._session = None
        self._thread = None
        self.error = None
        self.dead = False          # capture gave up for good
        self.restarts = 0          # times arecord had to be respawned

    # Opening a capture stream applies VREF bias to the mic pin, and the input
    # rails for a moment while that settles - measured here at ~3s to fall from
    # full scale to a noise floor of rms ~20. Calibrating across it reports
    # "clipping at idle" at ANY gain, including negative, because the transient
    # is not a gain phenomenon. Discard it rather than measure it.
    SETTLE_S = 3.5

    def calibrate(self, seconds=3.0):
        """Sample the idle mic and decide whether VAD can work on it.

        Judged by what actually matters - does the detector fire on silence? -
        rather than by raw level. A noisy-but-clean mic works fine; a clipped
        one reads as continuous speech and makes voice mode unusable.
        Returns (ok, rms, message).
        """
        total = self.SETTLE_S + max(1.0, seconds)
        cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "1",
               "-d", str(int(math.ceil(total)))]
        if self.device:
            cmd += ["-D", self.device]
        cmd += ["-"]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=total + 5).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return False, 0.0, f"could not read the microphone ({e})"
        if not out:
            return False, 0.0, "microphone produced no audio"

        pcm = np.frombuffer(out, dtype=np.int16)
        skip = int(self.SETTLE_S * RATE)
        if len(pcm) > skip + RATE:      # keep the post-settle portion only
            pcm = pcm[skip:]
        norm = pcm.astype(np.float32) / 32768.0
        rms = float(np.sqrt((norm ** 2).mean()))
        clipped = float((np.abs(norm) >= CLIP_LEVEL).mean())
        if clipped > 0.02:
            return False, rms, (
                f"microphone is clipping at idle ({clipped*100:.0f}% of samples, RMS {rms:.2f}). "
                "Input gain is much too high - try `amixer -c Generic_1 sset 'Front Mic Boost' 0` "
                "and lower `Capture`. Voice detection cannot work until this is fixed."
            )

        # The real test: run the detector over the idle sample.
        session = torch.jit.load(MODEL_PATH, map_location="cpu")
        session.eval()
        hits = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(pcm) - FRAME, FRAME):
                prob = float(session(torch.from_numpy(norm[i:i + FRAME]), RATE).item())
                hits += prob >= SPEECH_ON
                total += 1
        if total and hits / total > 0.02:
            return False, rms, (
                f"the detector fires on your idle microphone ({hits}/{total} frames, RMS {rms:.2f}) "
                "- it would generate phantom turns. Lower the input gain, e.g. "
                "`amixer -c Generic_1 sset Capture 25%`."
            )
        note = f"noise floor RMS {rms:.3f}"
        if rms > NOISE_FLOOR_WARN:
            note += " (high, but the detector stays quiet)"
        return True, rms, note

    def available(self):
        if not os.path.exists(MODEL_PATH):
            return False, f"Silero VAD model not found at {MODEL_PATH} (pip install silero-vad)"
        return True, None

    def _spawn_capture(self):
        cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "1"]
        if self.device:
            cmd += ["-D", self.device]
        cmd += ["-"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self.error = "arecord not found"
            return None
        # Drop the mic-bias settling transient (see SETTLE_S). The frame loop's
        # clip check catches the railed part of it, but the tail decays through
        # speech-like levels for a second or so and VAD will happily call that
        # an utterance - a phantom turn every time capture starts or respawns.
        try:
            to_drop = int(self.SETTLE_S * RATE) * 2
            while to_drop > 0:
                chunk = proc.stdout.read(min(8192, to_drop))
                if not chunk:
                    break
                to_drop -= len(chunk)
        except Exception:
            pass
        return proc

    def start(self):
        ok, why = self.available()
        if not ok:
            self.error = why
            return False
        self._session = torch.jit.load(MODEL_PATH, map_location="cpu")
        self._session.eval()
        self._proc = self._spawn_capture()
        if self._proc is None:
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def alive(self):
        """False once capture has stopped for good - voice input is dead."""
        return self._thread is not None and self._thread.is_alive() and not self.dead

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def mute(self):
        """Stop emitting utterances (audio keeps flowing so state stays warm)."""
        self.enabled.clear()

    def unmute(self):
        self.enabled.set()

    def _run(self):
        preroll = collections.deque(maxlen=PREROLL_FRAMES)
        speaking = False
        voiced, silence_run = [], 0
        nbytes = FRAME * 2

        while not self._stop.is_set():
            raw = self._proc.stdout.read(nbytes)
            if not raw or len(raw) < nbytes:
                # arecord ended (device hiccup, xrun, stolen by another client).
                # Silently breaking here used to kill voice input for the rest
                # of the session with no indication at all - respawn instead.
                if self._stop.is_set():
                    break
                if self.restarts >= MAX_CAPTURE_RESTARTS:
                    self.dead = True
                    print("\n(microphone capture failed repeatedly - voice input is off; "
                          "typing still works)", file=sys.stderr, flush=True)
                    break
                self.restarts += 1
                print(f"\n(microphone dropped out - restarting capture, attempt {self.restarts})",
                      file=sys.stderr, flush=True)
                try:
                    self._proc.kill()
                except Exception:
                    pass
                time.sleep(RESTART_BACKOFF_S)
                self._proc = self._spawn_capture()
                if self._proc is None:
                    self.dead = True
                    break
                # any partial utterance is garbage now
                preroll.clear()
                speaking = False
                voiced, silence_run = [], 0
                if hasattr(self._session, "reset_states"):
                    self._session.reset_states()
                continue
            pcm = np.frombuffer(raw, dtype=np.int16)
            norm = pcm.astype(np.float32) / 32768.0
            if float((np.abs(norm) >= CLIP_LEVEL).mean()) > CLIP_FRACTION:
                continue   # clipped garbage, not speech
            with torch.no_grad():
                prob = float(self._session(torch.from_numpy(norm), RATE).item())

            if not speaking:
                preroll.append(pcm)
                if prob >= SPEECH_ON and self.enabled.is_set():
                    speaking = True
                    voiced = list(preroll)
                    preroll.clear()
                    silence_run = 0
                    if self.on_speech_start:
                        try:
                            self.on_speech_start()
                        except Exception:
                            pass
                continue

            voiced.append(pcm)
            silence_run = silence_run + 1 if prob < SPEECH_OFF else 0
            done = silence_run >= self.silence_frames or len(voiced) >= self.max_frames
            if not done:
                continue

            speech_frames = len(voiced) - silence_run
            if speech_frames >= self.min_speech_frames and self.enabled.is_set():
                keep = min(len(voiced), speech_frames + KEEP_TAIL_FRAMES)
                self.utterances.put(np.concatenate(voiced[:keep]))
            speaking = False
            voiced, silence_run = [], 0
            if hasattr(self._session, "reset_states"):
                self._session.reset_states()

    @staticmethod
    def to_wav_bytes(pcm_int16):
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm_int16.tobytes())
        return buf.getvalue()
