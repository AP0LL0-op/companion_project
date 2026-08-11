"""Low-latency streaming speech: emit audio while CSM is still generating.

CSM produces audio frames autoregressively at 12.5 Hz, and transformers'
generate() exposes each frame through the `streamer` hook. The Mimi codec
decodes a partial frame sequence and - verified on this model - the decoded
prefix is stable as frames are appended (decode(0:k) matches decode(0:n) over
the overlap to max_err ~7e-4, i.e. inaudible). Re-decoding from frame 0 on each flush and emitting just the new tail is
seamless; decoding disjoint chunks and concatenating them is NOT (max_err
~0.39, audible clicks), so this deliberately re-decodes from the start. That
decode is quadratic in frame count though, so flushing at a fixed interval made
total cost cubic - 42% of wall time on a 10s utterance. flush_interval() backs
off once the opening frames are out.

Time to first sound drops from ~2.4 s (wait for the whole utterance) to
~0.4 s for CSM alone.

Generation runs slightly slower than playback, so audio is pre-buffered before
playback starts, and ReplyPlayer.lead() reports how far ahead the buffer is.
Callers use that to stretch the pause between sentences when they are falling
behind, instead of letting the stream run dry mid-word.
"""
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from tts import normalize_for_speech

FLUSH_EVERY_FRAMES = 4   # 4 -> ~0.39s TTFS; lower flushes more often but decode overhead dominates

# Each flush re-decodes the whole utterance so far (needed: decoding disjoint
# chunks leaves audible seams). Mimi's decoder is quadratic in frame count, so
# flushing at a fixed interval makes total decode cost grow cubically - 42% of
# wall time on a 10s utterance. Early flushes are what set time-to-first-sound,
# so keep those frequent and back off after.
def flush_interval(n_frames):
    if n_frames < 16:
        return 4          # first ~1.3s: responsiveness matters most
    if n_frames < 48:
        return 12
    return 24
PREBUFFER_S = 0.5        # default; prefer prebuffer_for() which scales with utterance length

REALTIME_FACTOR = 1.35   # measured end-to-end through the streaming path, not bare generation
SECONDS_PER_WORD = 0.25  # measured: ~0.23-0.28 s of speech per word


def prebuffer_for(text, floor=0.30, ceiling=1.20):
    """Seconds of audio to buffer before playback, so playback never outruns generation.

    For an utterance of duration D, generation takes REALTIME_FACTOR*D. Starting
    playback once B seconds are buffered means playback ends at
    REALTIME_FACTOR*B + D, which must be >= REALTIME_FACTOR*D, so
    B >= D*(REALTIME_FACTOR-1)/REALTIME_FACTOR. A margin is added on top.
    """
    est_duration = max(1, len(text.split())) * SECONDS_PER_WORD
    need = est_duration * (REALTIME_FACTOR - 1.0) / REALTIME_FACTOR
    # Small margin only: if the estimate is slightly short the buffer runs dry at
    # the very END of a chunk, which just lengthens the pause before the next
    # sentence rather than glitching mid-word.
    return min(ceiling, max(floor, need * 1.15))


class SpeechInterrupted(Exception):
    """Raised from inside generation when the user barges in."""


class FrameStreamer:
    """Receives codebook frames from generate(), emits decoded audio tails."""

    def __init__(self, model, on_audio, every=FLUSH_EVERY_FRAMES, abort_check=None):
        self.model = model
        self.on_audio = on_audio
        self.every = every
        self.abort_check = abort_check
        self.num_codebooks = model.config.num_codebooks
        self.eos = model.config.codebook_eos_token_id
        # Mimi's codebook size - the real upper bound on a valid audio code.
        # NOT the same as CSM's codebook vocab (2051, which includes pad=2050),
        # so codes valid to CSM can still be out of range for the codec.
        self.codebook_size = getattr(
            getattr(model.config, "codec_config", None), "codebook_size", 2048)
        self.frames = []       # pending CPU frames not yet moved to the GPU
        self.gpu_codes = None  # all codes so far, accumulated on-device
        self.emitted = 0
        self._last_flush = 0
        self.first_audio_wall = None   # absolute wall clock of first emitted audio

    def put(self, value):
        # Called once per generated frame (~every 80ms of audio), which makes it
        # the natural interrupt point: raising here aborts generate() promptly.
        if self.abort_check is not None and self.abort_check():
            raise SpeechInterrupted()
        # generate() also echoes the text prompt through the streamer; only
        # tensors that are one full frame wide are audio.
        if value.ndim != 2 or value.shape[-1] != self.num_codebooks:
            return
        frame = value[0]
        if bool((frame == self.eos).all()):
            return
        # Shape alone does not prove this is audio. generate() pushes every
        # tensor through the streamer hook, and a non-audio tensor that happens
        # to be num_codebooks wide passes the check above carrying text-range
        # token ids. Decoding one indexes Mimi's 2048-entry codebook with
        # values like 27136 or 128001 - an out-of-bounds GPU gather, which
        # surfaces as HSA_STATUS_ERROR_EXCEPTION and aborts the process.
        #
        # This was the intermittent "HSA fault": ~1 in 3 sessions, always
        # during synthesis, with a misleading stack (HIP reports async kernel
        # errors at the next API call, so it pointed at SetDevice). Confirmed
        # by bounds-checking our own accumulated codes at the flush boundary -
        # a single out-of-range flush killed the daemon immediately.
        #
        # Cheap: `value` already arrives on CPU, so this costs no GPU sync.
        if int(frame.max()) >= self.codebook_size or int(frame.min()) < 0:
            return
        self.frames.append(frame)
        n = self._n_total()
        if n - self._last_flush >= flush_interval(n):
            self.flush()

    def _n_total(self):
        seen = self.gpu_codes.shape[0] if self.gpu_codes is not None else 0
        return seen + len(self.frames)

    def end(self):
        self.flush()

    def flush(self):
        if not self.frames:
            return
        # Frames arrive on CPU from generate()'s streamer hook, so each new
        # one has to cross to the GPU once - but the old code re-stacked and
        # re-transferred every frame since the start of the utterance on every
        # flush, since self.frames was never cleared. That's redundant CPU->GPU
        # traffic growing quadratically over a long sentence, on a card sharing
        # VRAM with a second process (llama-server) under real memory pressure.
        # Move only what's new and grow a persistent on-device buffer instead;
        # decode() still runs on the full accumulated sequence, so the
        # re-decode-from-frame-0 behavior this depends on for clean audio
        # (see module docstring) is unchanged.
        new_codes = torch.stack(self.frames).to(self.model.device)
        self.frames = []
        self.gpu_codes = new_codes if self.gpu_codes is None else torch.cat(
            [self.gpu_codes, new_codes], dim=0
        )
        with torch.inference_mode():
            audio = self.model.codec_model.decode(
                self.gpu_codes.transpose(0, 1).unsqueeze(0)
            ).audio_values[0, 0]
        new = audio[self.emitted:]
        if new.numel() == 0:
            return
        self.emitted = audio.shape[-1]
        self._last_flush = self.gpu_codes.shape[0]
        if self.first_audio_wall is None:
            self.first_audio_wall = time.time()
        self.on_audio(new.float().cpu().numpy())


class ReplyPlayer:
    """One aplay process for a whole reply, fed from a queue by a writer thread.

    Generation of the next sentence therefore overlaps playback of the current
    one. Without this, synthesis of sentence N+1 only starts once N has finished
    playing, which leaves an audible ~0.4s hole before every sentence.
    """

    def __init__(self, sample_rate, prebuffer_s=PREBUFFER_S, capture=False):
        self.sample_rate = sample_rate
        self.prebuffer = int(sample_rate * prebuffer_s)
        self.queue = queue.Queue()
        self.aborted = False
        self.delivered = 0.0     # seconds of audio handed over
        self._t_play = None      # when playback actually began
        self._proc = None
        self._lock = threading.Lock()
        # What she actually heard herself say. Captured at the point audio is
        # written to aplay rather than when it is queued, because those differ
        # exactly where it matters: on barge-in, stop() discards whatever is
        # still queued, so queued audio was never spoken aloud.
        self.capture = capture
        self._heard = [] if capture else None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def heard(self):
        """Float32 PCM of what actually reached the speaker, or None.

        Note aplay buffers ~200ms, so on an interrupt the true tail is a
        fraction of a second shorter than this. Still far closer than the
        text-level guess it replaces.
        """
        if not self._heard:
            return None
        return np.concatenate(self._heard)

    def write(self, samples):
        self.delivered += len(samples) / self.sample_rate
        self.queue.put(samples)

    def lead(self):
        """Seconds of audio buffered ahead of the playhead. Negative means the
        stream has run dry - that is what choppiness sounds like."""
        if self._t_play is None:
            return self.delivered
        return self.delivered - (time.time() - self._t_play)

    def gap(self, seconds):
        """Queue a short silence - the natural beat between sentences, which
        also buys the buffer a little slack."""
        self.delivered += seconds
        self.queue.put(np.zeros(int(self.sample_rate * seconds), dtype=np.float32))

    def close(self):
        self.queue.put(None)
        self.thread.join()

    def stop(self):
        """Halt playback immediately: kill aplay mid-buffer and drop queued audio."""
        with self._lock:
            self.aborted = True
            if self._proc is not None:
                try:
                    self._proc.kill()   # unblocks a writer stuck in stdin.write
                except ProcessLookupError:
                    pass
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self.queue.put(None)
        self.thread.join()

    def _start(self):
        try:
            return subprocess.Popen(
                ["aplay", "-q", "-f", "S16_LE", "-r", str(self.sample_rate),
                 "-c", "1", "-t", "raw", "--buffer-time=200000", "-"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("(aplay not found - audio will not play)", file=sys.stderr)
            return None

    def _run(self):
        proc, pending, buffered, failed = None, [], 0, False
        pending_raw = []
        while True:
            item = self.queue.get()
            if item is None:
                break
            if failed or self.aborted:
                continue
            pcm = (np.clip(item, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            if proc is None:
                pending.append(pcm)
                if self.capture:
                    pending_raw.append(item)
                buffered += len(item)
                if buffered >= self.prebuffer:
                    with self._lock:
                        if self.aborted:
                            failed = True
                            continue
                        proc = self._proc = self._start()
                    if proc is None:
                        failed = True
                        continue
                    self._t_play = time.time()
                    for c in pending:
                        proc.stdin.write(c)
                    proc.stdin.flush()
                    if self.capture:
                        self._heard.extend(pending_raw)
                    pending, pending_raw = [], []
                continue
            try:
                proc.stdin.write(pcm)
                proc.stdin.flush()
                if self.capture:
                    self._heard.append(item)
            except (BrokenPipeError, ValueError):
                failed = True
        if proc is None and pending and not failed and not self.aborted:
            with self._lock:
                proc = self._proc = self._start()
            if proc is not None:
                for c in pending:
                    proc.stdin.write(c)
        if proc is not None:
            try:
                proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            proc.wait()


class NullPlayer:
    """Drop-in for ReplyPlayer that discards audio instead of playing it.

    Used to warm the model without making a sound, and - importantly - for
    stress and diagnostic runs. A forty-turn reproduction attempt will
    otherwise play forty synthesized replies aloud through whatever speakers
    are attached, which is antisocial if anyone else is in the house.

    Implements the whole ReplyPlayer surface, not just write/close: callers
    also use lead() to pace generation against playback, gap() for the beat
    between sentences, stop() for barge-in, and heard() for the self-check.
    Missing any of those turns a silent run into an AttributeError mid-reply.
    """

    def __init__(self, sample_rate=24000, prebuffer_s=0.0, capture=False):
        self.sample_rate = sample_rate
        self.capture = capture
        self._heard = [] if capture else None
        self.delivered = 0.0
        self.aborted = False

    def write(self, samples):
        self.delivered += len(samples) / self.sample_rate
        if self.capture:
            self._heard.append(samples)

    def lead(self):
        # Never behind: nothing is draining, so generation should never stall
        # waiting for playback that isn't happening.
        return 1e6

    def gap(self, seconds):
        self.delivered += seconds

    def stop(self):
        self.aborted = True

    def close(self):
        pass

    def heard(self):
        if not self._heard:
            return None
        return np.concatenate(self._heard)


class StreamPlayer:
    """Pipes raw PCM to a single persistent aplay process for gapless playback."""

    def __init__(self, sample_rate, prebuffer_s=PREBUFFER_S):
        self.sample_rate = sample_rate
        self.prebuffer = int(sample_rate * prebuffer_s)
        self.proc = None
        self.pending = []
        self.buffered = 0

    def write(self, samples):
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if self.proc is None:
            self.pending.append(pcm)
            self.buffered += len(samples)
            if self.buffered >= self.prebuffer:
                self._start()
            return
        try:
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def _start(self):
        try:
            self.proc = subprocess.Popen(
                ["aplay", "-q", "-f", "S16_LE", "-r", str(self.sample_rate),
                 "-c", "1", "-t", "raw", "--buffer-time=200000", "-"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("(aplay not found - audio will not play)", file=sys.stderr)
            self.pending = []
            return
        for chunk in self.pending:
            self.proc.stdin.write(chunk)
        self.proc.stdin.flush()
        self.pending = []

    def close(self):
        if self.proc is None and self.pending:
            self._start()
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            self.proc.wait()
            self.proc = None


def stream_speak(text, processor, model, player, speaker_id, max_new_tokens=600, abort_check=None):
    """Synthesize `text` and push audio into `player` as frames are produced.

    Returns the absolute wall-clock time of the first emitted audio, or None.
    Raises SpeechInterrupted if abort_check() turns true mid-generation.
    """
    text = normalize_for_speech(text)
    conversation = [{"role": str(speaker_id), "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True,
    ).to(model.device)

    streamer = FrameStreamer(model, player.write, abort_check=abort_check)
    with torch.inference_mode():
        model.generate(
            **inputs, max_new_tokens=max_new_tokens, output_audio=False, streamer=streamer,
        )
    return streamer.first_audio_wall
