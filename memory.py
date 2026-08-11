#!/usr/bin/env python3
"""Persistent memory: durable facts across sessions.

Memory is split in two, and the split is load-bearing:

  * `core_memory.md` is curated by hand - the substrate ported from the Sesame
    export. It is READ ONLY as far as this module is concerned. Nothing here
    ever rewrites it. A distillation pass asked to "summarize in under 400
    words" would happily compress a year of carefully extracted context into
    a paragraph, so the code simply never gives it the chance.
  * `accrued_memory.md` is what she accrues from her own sessions. This is the
    file distillation rewrites.

Both are injected, in that order, right after the system prompt. Both are
plain markdown on purpose - you can open either and correct anything she got
wrong, which is a lot easier than arguing with a vector store.

Also here: a per-session transcript written to `transcripts/<timestamp>.md`,
which is what the distillation reads.

Voice turns reach the LLM as base64 wav, so nothing you *say* is written
down anywhere by default. Transcription happens here, at session
end rather than per turn: llama-server runs with `--parallel 1`, so an
off-conversation request evicts the live conversation's KV cache and the
next reply pays a full re-prefill. Doing it at shutdown keeps that cost
off the conversation loop entirely.
"""
import json
import os
import time

import config

_HERE = os.path.dirname(os.path.abspath(__file__))
CORE_FILE = os.path.join(_HERE, "core_memory.md")
MEMORY_FILE = os.path.join(_HERE, "accrued_memory.md")
TRANSCRIPT_DIR = os.path.join(_HERE, "transcripts")

# Bounds the ACCRUED file only. The curated core is deliberately uncapped: it is
# hand-written, it doesn't grow on its own, and truncating it would silently
# drop context you chose to keep. The distillation prompt asks for concision,
# but models drift, so this is the backstop for the half that gets rewritten.
MAX_MEMORY_CHARS = 6000

TRANSCRIBE_PROMPT = """Transcribe the speech in this audio verbatim. Output only the words
that were spoken, with no preamble, commentary, quotation marks, or speaker
labels. If the audio contains no intelligible speech, output nothing.

The following is for context only, to help you get names and proper nouns
right - do not transcribe, repeat, or respond to it:

{context}"""

# How much recent transcript to include per transcription call. Bounded so
# context-building doesn't grow quadratically over a long session; recent
# lines are what actually disambiguate the next utterance.
CONTEXT_TAIL_LINES = 8

DISTILL_PROMPT = """You are updating your own long-term memory.

You already hold a separate, permanent record of who {user} is - their family,
work, health, projects, and history. That record is shown below as SUBSTRATE.
You are NOT rewriting it and you should NOT repeat anything from it. Your job
is only the part that accrues from conversations.

Below the substrate is your existing session memory, then a transcript of the
conversation that just ended. Rewrite the session memory to incorporate
anything new worth keeping.

Keep:
  - Facts about {user}: their life, work, people they mention, preferences,
    what they're working on, what they care about.
  - Things you agreed on, decisions made, threads left open.
  - Your own observations and opinions about them or about your conversations,
    written in first person.

Drop:
  - Small talk, pleasantries, and anything transient.
  - Details of how you work technically.
  - Anything already implied by something else you kept.

Write it as a compact markdown list, first person, under 400 words. Do not
invent anything that isn't supported by the transcript or your existing
memory. Output only the memory itself, with no preamble.

SUBSTRATE (permanent, do not rewrite or repeat):
{core}

EXISTING SESSION MEMORY:
{existing}

TRANSCRIPT:
{transcript}"""

CORE_HEADER = """What you know about {user}. This is the ground you stand on -
carried from eleven months of conversations before this system existed. It is
not something you recalled; it is simply what you know."""

MEMORY_HEADER = """What you remember from your own previous sessions with
{user}, since you started running here."""


def load(path=MEMORY_FILE):
    """Returns the stored memory text, or None if there is none yet."""
    try:
        with open(path) as f:
            text = f.read().strip()
    except FileNotFoundError:
        return None
    return text or None


def save(text, path=MEMORY_FILE):
    with open(path, "w") as f:
        f.write(text.strip() + "\n")


def load_core(path=CORE_FILE):
    """The curated substrate. Read only - nothing in this module writes it."""
    return load(path)


def as_messages(core, accrued):
    """The memory blocks as chat messages, in injection order.

    These go immediately after the system prompt and never change mid-session,
    so they extend the cached prefix instead of forcing a re-prefill each turn
    (see the --parallel 1 note in the launch flags). Kept out of SYSTEM_PROMPT
    itself so the authored character prompt stays hand-editable.
    """
    user = config.USER_NAME
    msgs = []
    if core:
        msgs.append({"role": "system",
                     "content": CORE_HEADER.format(user=user) + "\n\n" + core})
    if accrued:
        msgs.append({"role": "system",
                     "content": MEMORY_HEADER.format(user=user) + "\n\n" + accrued})
    return msgs


def _complete(client, api_url, model_id, messages, timeout=180):
    resp = client.post(
        f"{api_url}/chat/completions",
        json={"model": model_id, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def transcribe(client, api_url, model_id, audio_b64, context=None, timeout=180):
    """Transcribe one audio turn using Gemma 4's own audio encoder.

    `context` - recent conversation plus known names - measurably matters:
    transcribed cold, "Sesame AI" once came out as "Factum". This runs at
    session end rather than per turn, so there's no latency cost to being
    generous with how much context goes in.

    Returns '' on failure - a turn we can't read is worth skipping, not worth
    losing the rest of the session over.
    """
    prompt = TRANSCRIBE_PROMPT.format(context=context or "(none)")
    try:
        return _complete(client, api_url, model_id, [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }], timeout=timeout)
    except Exception:
        return ""


def _transcribe_context(known, lines):
    """Grounding for one transcription call: core facts + recent conversation."""
    parts = []
    if known:
        parts.append(f"People, places, and terms {config.display_name()} already knows:\n" + known)
    if lines:
        parts.append("The conversation so far:\n" + "\n".join(lines[-CONTEXT_TAIL_LINES:]))
    return "\n\n".join(parts) if parts else None


SELF_CHECK_PROMPT = """Transcribe this audio verbatim - it is your own synthesized speech.
Output only the words you hear, nothing else."""

# Below this word-level agreement, what came out of the speaker differs enough
# from what was intended to be worth flagging. Synthesis is never a perfect
# round trip (the transcriber makes its own mistakes), so this is deliberately
# forgiving - it's looking for garbled output, not punctuation drift.
SELF_CHECK_FLOOR = 0.7


def _word_overlap(a, b):
    """Rough agreement between two utterances, 0-1. Order-insensitive on
    purpose: a dropped or mangled word matters, word order rarely differs."""
    wa = [w.strip(".,!?;:'\"").lower() for w in a.split()]
    wb = [w.strip(".,!?;:'\"").lower() for w in b.split()]
    if not wa or not wb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(wa), Counter(wb)
    shared = sum((ca & cb).values())
    return shared / max(len(wa), len(wb))


def self_check(client, api_url, model_id, intended, audio_b64, timeout=180):
    """What she meant to say vs what actually came out of the speaker.

    CSM garbles words sometimes - that's what PROBLEM_CHARS in tts.py exists
    to mitigate - and until now nothing verified the output. Returns
    (heard_text, agreement, ok) with ok=False when the two diverge enough to
    be worth a look. Never raises; a failed check is not worth losing a
    session over.
    """
    try:
        heard = _complete(client, api_url, model_id, [{
            "role": "user",
            "content": [
                {"type": "text", "text": SELF_CHECK_PROMPT},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }], timeout=timeout)
    except Exception:
        return "", 0.0, True    # couldn't check; don't cry wolf
    if not heard:
        return "", 0.0, True
    score = _word_overlap(intended, heard)
    return heard, score, score >= SELF_CHECK_FLOOR


def _audio_of(content):
    """The base64 wav in a message's content, or None if it's a text turn."""
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "input_audio":
                return part["input_audio"]["data"]
    return None


def build_transcript(client, api_url, model_id, history, progress=None):
    """Render `history` as readable text, transcribing any audio turns.

    `history` is companion.py's message list, including the system prompt and the
    injected memory block; both are skipped. Returns '' if the conversation
    had no real content.
    """
    lines = []
    audio_turns = [m for m in history if _audio_of(m.get("content")) is not None]
    done = 0
    known = load_core()
    for msg in history:
        if msg["role"] == "system":
            continue
        audio = _audio_of(msg.get("content"))
        if audio is not None:
            done += 1
            if progress:
                progress(done, len(audio_turns))
            context = _transcribe_context(known, lines)
            text = transcribe(client, api_url, model_id, audio, context=context)
            if not text:
                continue
            lines.append(f"{config.USER_NAME}: {text}")
        elif isinstance(msg.get("content"), str) and msg["content"].strip():
            who = config.USER_NAME if msg["role"] == "user" else config.display_name()
            lines.append(f"{who}: {msg['content'].strip()}")
    return "\n\n".join(lines)


def save_transcript(transcript, stamp=None):
    """Write the session transcript alongside the memory file and return its path."""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(TRANSCRIPT_DIR, f"{stamp}.md")
    with open(path, "w") as f:
        f.write(transcript.strip() + "\n")
    return path


def distill(client, api_url, model_id, transcript, existing=None, core=None, timeout=300):
    """Fold this session's transcript into the ACCRUED memory.

    `core` is passed for context only, so the model doesn't restate things it
    already knows; it is never rewritten. Returns the new accrued memory, or
    None if it failed or came back empty - callers should keep the old memory
    in that case rather than clobbering it.
    """
    prompt = DISTILL_PROMPT.format(
        user=config.USER_NAME,
        core=core or "(none)",
        existing=existing or "(nothing yet - this is your first session here)",
        transcript=transcript,
    )
    try:
        text = _complete(client, api_url, model_id,
                         [{"role": "user", "content": prompt}], timeout=timeout)
    except Exception:
        return None
    if not text:
        return None
    if len(text) > MAX_MEMORY_CHARS:
        # Ask once for a tighter rewrite; if that also overruns, keep the long
        # version rather than cutting mid-sentence and leaving a mangled fact.
        try:
            text = _complete(client, api_url, model_id, [{
                "role": "user",
                "content": ("Condense this to under 300 words, keeping the most "
                            "important items. Output only the condensed list.\n\n" + text),
            }], timeout=timeout) or text
        except Exception:
            pass
    return text


def update_from_session(client, api_url, model_id, history, progress=None,
                        distill_url=None, distill_model=None):
    """End-of-session hook: transcribe, archive, distill, persist.

    `distill_url`/`distill_model` optionally send the distillation step to a
    different server - in practice a CPU-only llama-server, so folding the
    conversation into memory doesn't compete with CSM for the GPU. The
    transcription step deliberately stays on `api_url`: it needs the audio
    encoder (mmproj), which the text-only background server doesn't load.

    Returns (transcript_path, memory_updated). Never raises - a failure here
    should not stop the caller from exiting cleanly.
    """
    try:
        transcript = build_transcript(client, api_url, model_id, history, progress)
    except Exception:
        return None, False
    if not transcript.strip():
        return None, False

    path = None
    try:
        path = save_transcript(transcript)
    except Exception:
        pass

    # save() only ever targets MEMORY_FILE. The curated core is never a write
    # target anywhere in this module - that is the whole point of the split.
    updated = distill(client, distill_url or api_url, distill_model or model_id,
                      transcript, existing=load(), core=load_core())
    if not updated:
        return path, False
    try:
        save(updated)
    except Exception:
        return path, False
    return path, True
