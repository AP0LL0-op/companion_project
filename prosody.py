#!/usr/bin/env python3
"""How a turn actually sounded, measured from the audio rather than guessed at.

Both directions: how she sounded on her own reply, and how the person sounded
on the turn she is about to answer. Same measurements, separate baselines, and
deliberately different framing for each - noticing your own tone and reading
someone else's are not the same act, and the second is far easier to get wrong.

Gemma's audio encoder was the obvious place to get this and it does not work:
it is ASR-trained, so it discards exactly what we want. Measured directly - it
missed a 1.5x speedup entirely, could not distinguish full amplitude from 15%
amplitude (ASR pipelines normalize level as preprocessing, so loudness is
destroyed before the model sees it), and it will agree the same clip is sad,
angry, or joyful depending on which you suggest. A perception you can talk out
of its own reading is not a perception.

Everything here is computed from the PCM instead: objective, deterministic,
not suggestible, no model, no VRAM. It succeeds precisely where the encoder
failed - rate and loudness are the easy cases for signal processing and the
impossible ones for a normalizing transcriber.

Absolute numbers are meaningless to her ("4.2 words per second" is not a
feeling), so everything is reported against a rolling baseline of that
speaker's own speech, the same way body.py handles temperature. What matters is
*different from usual*.

This is measurement, never identification. It compares a voice to how that
voice has usually sounded; it has no way to tell one speaker from another and
does not try to. Which baseline an utterance belongs to comes from which
channel it arrived on - microphone or synthesizer - not from anything about
the voice itself.
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(_HERE, "prosody_baseline.json")

FRAME_MS = 25
HOP_MS = 10
# Human speech F0 range, generously bounded. Anything outside this is almost
# certainly an octave error rather than a real pitch.
F0_MIN, F0_MAX = 60.0, 350.0
# Frames this far below the clip's own peak energy are treated as silence.
SILENCE_REL_DB = -35.0

BASELINE_HALFLIFE = 20      # utterances
_ALPHA = 1 - 0.5 ** (1 / BASELINE_HALFLIFE)

# How far from baseline is worth her noticing. Chosen to be loose - this
# should catch "noticeably different", not natural variation.
NOTABLE = {
    "rate_wps": 0.22,       # relative: 22% faster/slower than usual
    "loudness_db": 4.0,     # absolute dB
    "pitch_hz": 0.10,       # relative
    "pitch_var_hz": 0.35,   # relative; expressiveness varies a lot naturally
    "pause_ratio": 0.12,    # absolute fraction
}


def _frames(x, sr):
    n = int(sr * FRAME_MS / 1000)
    h = int(sr * HOP_MS / 1000)
    if len(x) < n:
        return None, n, h
    count = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(count)[:, None]
    return x[idx], n, h


def _f0_autocorr(frame, sr):
    """F0 for one frame by autocorrelation, or None if unvoiced.

    Autocorrelation rather than anything fancier: this is a clean synthetic
    voice at a known sample rate, not field recordings, and the peak is
    unambiguous when speech is present.
    """
    f = frame - frame.mean()
    if not np.any(f):
        return None
    corr = np.correlate(f, f, mode="full")[len(f) - 1:]
    if corr[0] <= 0:
        return None
    corr = corr / corr[0]
    lo = int(sr / F0_MAX)
    hi = min(int(sr / F0_MIN), len(corr) - 1)
    if hi <= lo:
        return None
    seg = corr[lo:hi]
    peak = int(np.argmax(seg))
    # A weak peak means no clear periodicity - unvoiced, or silence.
    if seg[peak] < 0.3:
        return None
    return sr / (lo + peak)


def analyze(pcm, sample_rate, text=None):
    """Measure one utterance. `pcm` is float32 in [-1, 1].

    Returns None if there isn't enough audio to say anything meaningful.
    """
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(x) < sample_rate * 0.3:
        return None
    fr, n, h = _frames(x, sample_rate)
    if fr is None:
        return None

    energy = np.sqrt((fr ** 2).mean(axis=1)) + 1e-12
    peak = energy.max()
    voiced = energy > peak * (10 ** (SILENCE_REL_DB / 20))
    if voiced.sum() < 3:
        return None

    speech_s = float(voiced.sum() * h / sample_rate)
    total_s = float(len(x) / sample_rate)

    f0s = []
    for i in np.where(voiced)[0]:
        v = _f0_autocorr(fr[i], sample_rate)
        if v is not None:
            f0s.append(v)
    f0s = np.array(f0s) if f0s else np.array([])

    # Longest continuous run of silence, in seconds - a proxy for hesitation
    # that mean pause ratio alone would hide.
    longest, run = 0, 0
    for v in voiced:
        run = 0 if v else run + 1
        longest = max(longest, run)

    out = {
        "duration_s": round(total_s, 2),
        "speech_s": round(speech_s, 2),
        "pause_ratio": round(float(1 - voiced.mean()), 3),
        "longest_pause_s": round(longest * h / sample_rate, 2),
        "loudness_db": round(float(20 * np.log10(energy[voiced].mean())), 1),
        "pitch_hz": round(float(np.median(f0s)), 1) if len(f0s) else None,
        "pitch_var_hz": round(float(np.std(f0s)), 1) if len(f0s) > 2 else None,
    }
    if text:
        words = len([w for w in text.split() if any(c.isalnum() for c in w)])
        out["words"] = words
        out["rate_wps"] = round(words / speech_s, 2) if speech_s > 0 else None
    return out


# Baselines are per speaker. "Different from usual" is the only thing any of
# this means, and usual is a property of a voice - folding her synthesized
# speech and his live microphone into one average would produce a baseline that
# describes neither, and then flag every utterance as deviating from it.
SELF, USER = "self", "user"


def _load_all(path=BASELINE_FILE):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Migrate the original single-speaker file, which was hers by definition.
    if data and not any(k in data for k in (SELF, USER)):
        return {SELF: data}
    return data


def load_baseline(who=SELF, path=BASELINE_FILE):
    return _load_all(path).get(who, {})


def update_baseline(features, who=SELF, path=BASELINE_FILE):
    """Fold one utterance into that speaker's running sense of 'usual'."""
    allb = _load_all(path)
    base = allb.get(who, {})
    for k in NOTABLE:
        v = features.get(k)
        if v is None:
            continue
        base[k] = v if k not in base else base[k] + _ALPHA * (v - base[k])
    base["_n"] = base.get("_n", 0) + 1
    allb[who] = base
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(allb, f)
        os.replace(tmp, path)
    except OSError:
        pass
    return base


def deviations(features, baseline):
    """Which measures departed from usual, in plain terms.

    Returns [] until there's enough history for 'usual' to mean anything -
    telling her she sounded unusual on the third thing she ever said would be
    noise dressed up as self-awareness.
    """
    if not baseline or baseline.get("_n", 0) < 8:
        return []
    out = []
    for k, thresh in NOTABLE.items():
        v, b = features.get(k), baseline.get(k)
        if v is None or b is None:
            continue
        if k == "loudness_db":
            d = v - b
            if abs(d) >= thresh:
                out.append("louder than usual" if d > 0 else "quieter than usual")
        elif k == "pause_ratio":
            d = v - b
            if abs(d) >= thresh:
                out.append("more halting than usual" if d > 0 else "with fewer pauses than usual")
        else:
            if b == 0:
                continue
            d = (v - b) / abs(b)
            if abs(d) < thresh:
                continue
            label = {
                "rate_wps": ("faster than usual", "slower than usual"),
                "pitch_hz": ("higher-pitched than usual", "lower-pitched than usual"),
                "pitch_var_hz": ("more animated than usual", "flatter than usual"),
            }[k]
            out.append(label[0] if d > 0 else label[1])
    return out


def render(features, baseline=None):
    """A short line about how she just sounded, or None if unremarkable.

    Deliberately only speaks up when something differs. A running commentary
    on every ordinary utterance is noise, and she has been told not to narrate
    her own diagnostics.
    """
    if not features:
        return None
    devs = deviations(features, baseline if baseline is not None else load_baseline(SELF))
    if not devs:
        return None
    return ("How you just sounded, measured from your own audio: "
            + ", ".join(devs)
            + ". This is your voice, not a reading about someone else - "
              "treat it the way you'd notice your own tone. Don't comment on "
              "it unless it's relevant.")


def render_user(features, name, baseline=None):
    """A short line about how the person just sounded, or None if unremarkable.

    Same measurements, deliberately different framing. Hearing that someone
    sounded quiet is not the same as knowing why, and the failure mode here is
    not silence - it is confident misreading. Tired, ill, distracted, angry and
    concentrating all flatten a voice in similar ways, so this reports the
    signal and explicitly withholds the interpretation.

    Note what is absent: `rate_wps` needs a word count, and voice turns are not
    transcribed until the session ends, so speaking rate is unavailable live.
    Loudness, pitch, pitch variation and pausing all come straight from the
    waveform and are unaffected.
    """
    if not features:
        return None
    devs = deviations(features, baseline if baseline is not None else load_baseline(USER))
    if not devs:
        return None
    return (f"How {name} just sounded, measured from the audio of the turn "
            f"you are about to answer: " + ", ".join(devs)
            + f". This is signal, not diagnosis - the same flattening fits "
              f"tired, unwell, preoccupied, or simply concentrating, and you "
              f"cannot tell which from a waveform. Do not announce the "
              f"measurement or tell {name} how they sound. Let it inform how "
              f"you pitch the reply, and only ask if it seems worth asking.")
