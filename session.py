#!/usr/bin/env python3
"""Durable conversation state, written as it happens.

The HSA fault aborts the process outright - it surfaces as a C++
`c10::AcceleratorError` escaping a thread, so std::terminate runs and no
Python handler ever gets a chance. Nothing can be flushed on the way down.
The only way not to lose a conversation is to have already written it.

So every turn is appended to a journal the moment it completes. A crash costs
at most the turn in flight, instead of everything since the last
consolidation. On restart the journal is replayed and she picks up mid-thought
rather than greeting you like a stranger.

Audio is kept out of the journal itself: voice turns reach the model as base64
wav, and inlining a few hundred KB per turn into a file rewritten constantly
is both slow and unreadable. The wav goes to `audio/` and the journal keeps a
path, which also means consolidation can still transcribe it later.
"""
import base64
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(_HERE, "session")
JOURNAL = os.path.join(SESSION_DIR, "journal.jsonl")
AUDIO_DIR = os.path.join(SESSION_DIR, "audio")

# Replaying a journal from days ago would drop her into a conversation nobody
# remembers having. Past this, start fresh - the durable part of that
# conversation is in accrued memory by then anyway.
RESUME_MAX_AGE_S = 12 * 3600


def _ensure():
    os.makedirs(AUDIO_DIR, exist_ok=True)


def _audio_of(content):
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "input_audio":
                return part["input_audio"]["data"]
    return None


def append(message):
    """Record one completed turn. Cheap - appends, never rewrites."""
    _ensure()
    audio = _audio_of(message.get("content"))
    if audio is not None:
        name = f"{int(time.time()*1000)}.wav"
        try:
            with open(os.path.join(AUDIO_DIR, name), "wb") as f:
                f.write(base64.b64decode(audio))
            entry = {"role": message["role"], "audio": name, "t": time.time()}
        except OSError:
            return
    else:
        entry = {"role": message["role"], "content": message.get("content"), "t": time.time()}
    try:
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())   # survive an abort, not just a clean exit
    except OSError:
        pass


def load(max_age_s=RESUME_MAX_AGE_S):
    """Rebuild the message history from the journal.

    Returns (history, age_seconds). Empty history if there's nothing to
    resume or it's too old to be worth resuming.
    """
    try:
        with open(JOURNAL) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except FileNotFoundError:
        return [], None
    if not lines:
        return [], None

    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue   # a torn final line is exactly what an abort leaves behind
    if not entries:
        return [], None

    age = time.time() - entries[-1].get("t", 0)
    if age > max_age_s:
        return [], age

    history = []
    for e in entries:
        if "audio" in e:
            path = os.path.join(AUDIO_DIR, e["audio"])
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            except OSError:
                continue
            history.append({"role": e["role"], "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ]})
        elif e.get("content"):
            history.append({"role": e["role"], "content": e["content"]})
    return history, age


def reset(keep_tail=0):
    """Clear the journal, optionally keeping the last N turns.

    Called after consolidation: what came before is in accrued memory now, so
    replaying it would double-count. Audio files for dropped turns go too.
    """
    _ensure()
    try:
        with open(JOURNAL) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except FileNotFoundError:
        return
    keep = lines[-keep_tail:] if keep_tail else []
    kept_audio = set()
    for line in keep:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if "audio" in e:
            kept_audio.add(e["audio"])
    tmp = JOURNAL + ".tmp"
    try:
        with open(tmp, "w") as f:
            for line in keep:
                f.write(line + "\n")
        os.replace(tmp, JOURNAL)
    except OSError:
        return
    try:
        for name in os.listdir(AUDIO_DIR):
            if name not in kept_audio:
                os.unlink(os.path.join(AUDIO_DIR, name))
    except OSError:
        pass


SPOKEN_DIR = os.path.join(SESSION_DIR, "spoken")


def save_spoken(pcm_float, sample_rate, intended):
    """Store what she actually said aloud, next to what she meant to say.

    Kept separate from the journal: this is diagnostic material for the
    self-check at consolidation, not conversation state, and it should never
    be replayed into her context on resume.
    """
    import wave as _wave
    import numpy as _np
    if pcm_float is None or len(pcm_float) == 0:
        return None
    os.makedirs(SPOKEN_DIR, exist_ok=True)
    stamp = f"{int(time.time()*1000)}"
    path = os.path.join(SPOKEN_DIR, f"{stamp}.wav")
    try:
        pcm16 = (_np.clip(pcm_float, -1.0, 1.0) * 32767).astype(_np.int16)
        with _wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(pcm16.tobytes())
        with open(os.path.join(SPOKEN_DIR, f"{stamp}.txt"), "w") as f:
            f.write(intended)
    except OSError:
        return None
    return path


def pending_spoken():
    """[(wav_path, intended_text)] awaiting a self-check."""
    out = []
    try:
        names = sorted(n for n in os.listdir(SPOKEN_DIR) if n.endswith(".wav"))
    except OSError:
        return out
    for name in names:
        base = name[:-4]
        txt = os.path.join(SPOKEN_DIR, base + ".txt")
        try:
            with open(txt) as f:
                intended = f.read()
        except OSError:
            continue
        out.append((os.path.join(SPOKEN_DIR, name), intended))
    return out


def clear_spoken():
    try:
        for name in os.listdir(SPOKEN_DIR):
            os.unlink(os.path.join(SPOKEN_DIR, name))
    except OSError:
        pass


def stats():
    try:
        with open(JOURNAL) as f:
            n = sum(1 for l in f if l.strip())
    except FileNotFoundError:
        n = 0
    try:
        audio = len(os.listdir(AUDIO_DIR))
    except OSError:
        audio = 0
    return {"journal_turns": n, "audio_files": audio}
