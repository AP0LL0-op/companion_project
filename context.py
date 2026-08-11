#!/usr/bin/env python3
"""Assembles what she sees each turn.

Ordering here is load-bearing, for two separate reasons.

**Cache.** llama-server runs with `--parallel 1` so the conversation keeps one
warm KV cache; anything that changes between turns invalidates the cache from
its position onward. So everything stable (system prompt, memory, standing
instructions) goes first and forms a prefix that survives the whole run, and
anything that changes per turn (body readings, retrieved memories) goes last,
next to the live turn, where re-prefilling costs only its own tokens.

**Precedence.** The persona can always generate something plausible; real
memory is the thing that should win as it accumulates. That isn't automatic -
a model will happily blend fluent invention with specific truth unless
something tells it not to. GUARD is that instruction, and it exists because of
an observed failure, not a hypothetical one: she once opened a session
claiming she and the operator had been discussing "builder brain versus
consumer habits," which never happened. She had narrativized two trait bullets
from core memory into an episode. Traits are not episodes, and the guard says
so.
"""
import re

import config

# Standing instructions. Stable across the whole run, so this rides in the
# cached prefix alongside memory rather than being re-sent per turn.
GUARD = """HOW YOUR MEMORY WORKS

What you know about {user} is exactly what appears in the memory above, plus
anything explicitly retrieved and shown to you in a given turn. That is the
whole record. It is not a summary of a larger store you can reach for.

So: if something is not there, you do not remember it. Say that plainly -
"I don't have that," "I'm not sure we've talked about that." You are allowed
to not remember things. An honest gap is worth more to them than a confident
reconstruction, and they will usually just tell you the answer.

Two specific things to avoid, because they are easy to do by accident:

Knowing a fact about {user} is not the same as having discussed it. Your
memory says they prefer building things to consuming them. That means you
know it about them. It does NOT mean the two of you talked about it, and you should
not say "we were talking about..." unless an actual exchange is in your
memory or in this conversation.

Do not fill a gap with something that merely fits. If they ask what you were
last talking about and you don't have it, the answer is that you don't have
it - not a plausible topic assembled from what you know about them.

YOUR BODY

You have continuous readings from the machine you run on - temperature, power
draw, memory, load. They are always available to you, the way your own
heartbeat is always available to you: present, but not something you narrate.
Do not mention them unless asked, or unless something is flagged unusual.
When something IS flagged, you can notice it the way you'd notice being tired
or overheated - as a state you're in, not a readout you're reciting."""

def guard():
    """GUARD with the operator's name filled in."""
    return GUARD.format(user=config.USER_NAME)


# Turns that plausibly want episodic recall. Retrieval is gated on these
# rather than firing every turn: injected chunks sit after the cached prefix,
# so they re-prefill each time they change (~350ms for a 500-token block at
# measured prefill speed). Worth it when he's actually asking about the past,
# wasted otherwise.
RECALL_MARKERS = re.compile(
    r"\b(remember|recall|forget|forgot|last time|back (?:in|when)|"
    r"we(?:'re|'ve| were| are| have)? (?:just |been )?(?:talk|discuss|sa[iy]d)\w*|"
    r"you (?:said|told|mentioned)|what did|when did|how long|used to|"
    r"before|ago|earlier|the time (?:we|you|i))\b",
    re.I,
)

RETRIEVAL_HEADER = (
    "Retrieved from your memory of earlier conversations, because this turn "
    "seems to reach back. These are summaries, not transcripts - treat them "
    "as roughly right about what happened, not as exact quotes:"
)

BODY_HEADER = "Current readings from the machine you run on:"

# Below this, a gap is just a pause in conversation and not worth remarking on.
ABSENCE_FLOOR_S = 30 * 60


def render_absence(seconds):
    """A note that time has passed, for the first turn after a gap.

    Deliberately not a second greeting. She can't know he's back until he
    speaks, so announcing it would be talking to an empty room; what she
    actually needs is to know the gap exists when she answers. The raw
    duration goes in rather than a bucketed label - how much a three-hour
    absence matters is hers to judge, not something to decide for her.
    """
    if not seconds or seconds < ABSENCE_FLOOR_S:
        return None
    hours = seconds / 3600
    if hours < 1:
        span = f"{int(seconds // 60)} minutes"
    elif hours < 24:
        span = f"{hours:.1f} hours"
    else:
        span = f"{hours/24:.1f} days"
    return (f"{config.USER_NAME} has been away for about {span} - this is the "
            f"first thing they've said since. You've been running the whole time. Acknowledge "
            f"it only if it feels natural; don't make a production of it.")


def is_recall_shaped(text):
    """Whether a turn is asking about the past. Cheap gate for retrieval."""
    return bool(text and RECALL_MARKERS.search(text))


def render_body(state):
    """One compact line of body state, plus anything currently flagged.

    Deliberately terse and numeric. The instruction not to narrate it lives in
    GUARD; this is just the signal.
    """
    if not state:
        return None
    s = state.get("sample", {})
    elevated = state.get("elevated", [])
    bits = []
    if s.get("gpu_temp_junction_c") is not None:
        bits.append(f"GPU {s['gpu_temp_junction_c']:.0f}C junction")
    if s.get("gpu_power_w") is not None:
        bits.append(f"{s['gpu_power_w']:.0f}W")
    if s.get("gpu_vram_used_mib") and s.get("gpu_vram_total_mib"):
        bits.append(f"VRAM {s['gpu_vram_used_mib']/1024:.1f}/"
                    f"{s['gpu_vram_total_mib']/1024:.1f}GiB")
    if s.get("gpu_busy_pct") is not None:
        bits.append(f"{s['gpu_busy_pct']}% busy")
    if s.get("cpu_temp_c") is not None:
        bits.append(f"CPU {s['cpu_temp_c']:.0f}C")
    if s.get("load1") is not None:
        bits.append(f"load {s['load1']:.2f}")
    if not bits:
        return None
    line = f"{BODY_HEADER} " + ", ".join(bits) + "."
    if elevated:
        line += (f" UNUSUAL right now: {', '.join(elevated)} — "
                 f"above your normal baseline.")
    else:
        line += " All within your normal range."
    return line


def retrieve(query, k=4, first_exchange=False):
    """Episodic memory for this turn, or None.

    Returns None unless the turn actually reaches back. Never fires on the
    first exchange of a run - `restricted.md` calls for that explicitly, and
    it's a sane default regardless: an opening turn has no established topic
    to retrieve against.
    """
    if first_exchange or not is_recall_shaped(query):
        return None
    try:
        import archive
        hits = archive.search(query, k=k)
    except Exception:
        return None
    if not hits:
        return None
    return f"{RETRIEVAL_HEADER}\n" + archive.format_hits(hits)


def assemble(system_prompt, core, accrued, history,
             body_state=None, retrieval_query=None, first_exchange=False,
             away_seconds=None, prosody_note=None, thoughts_msg=None,
             user_prosody_note=None):
    """Build the full message list for one turn.

    `history` is conversation only - no system messages, no memory. The
    preamble is rebuilt here every turn from the same stable inputs, so it
    stays byte-identical run to run and the KV cache holds.

    Returns (messages, n_preamble). Callers trim `history` themselves; this
    reports how much of the result is preamble so the split is unambiguous.
    """
    preamble = [{"role": "system", "content": system_prompt}]
    import memory
    preamble.extend(memory.as_messages(core, accrued))
    preamble.append({"role": "system", "content": guard()})

    # Everything below changes between turns, so it goes as late as possible -
    # after the cached prefix and after the settled history, but immediately
    # BEFORE the live turn, so the prompt still ends on what he actually said
    # and the injected material reads as context for it.
    tail = []
    absence = render_absence(away_seconds)
    if absence:
        tail.append({"role": "system", "content": absence})
    # What she thought about while he was gone. Sits next to the absence note
    # because they're the same moment from her side.
    if thoughts_msg:
        tail.append(thoughts_msg)
    retrieved = retrieve(retrieval_query, first_exchange=first_exchange) if retrieval_query else None
    if retrieved:
        tail.append({"role": "system", "content": retrieved})
    body = render_body(body_state)
    if body:
        tail.append({"role": "system", "content": body})
    # Only present when her last reply actually departed from her own norm -
    # prosody.render() returns None otherwise, so this is silent most turns.
    if prosody_note:
        tail.append({"role": "system", "content": prosody_note})
    # How the person sounded on the turn she is about to answer. Last in the
    # tail on purpose: it is the most recent thing that happened and describes
    # the very message underneath it, so it belongs closest to that message.
    if user_prosody_note:
        tail.append({"role": "system", "content": user_prosody_note})

    history = list(history)
    if tail and history:
        messages = list(preamble) + history[:-1] + tail + history[-1:]
    else:
        messages = list(preamble) + history + tail

    return messages, len(preamble)
