#!/usr/bin/env python3
"""Per-operator settings. The only file a new install needs to touch.

Everything else in this repo is the same for everyone; this is what differs.
Kept as a module rather than a JSON file so it can carry explanation, and so
a missing value is an import error at startup instead of a KeyError halfway
through a conversation.

Override without editing the file:

    COMPANION_USER=Sam python companion.py --voice

or persistently, in an untracked `.env` beside this file:

    COMPANION_USER=Sam
    COMPANION_NAME=Ada
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_HERE, ".env")


def _load_env_file():
    """Read an untracked .env alongside this file, if present.

    So a real install doesn't depend on remembering to export anything, while
    the repo itself carries no personal values. Environment variables win over
    the file - a one-off `COMPANION_USER=... python companion.py` still overrides.

    Values are also pushed into os.environ (via setdefault, so a real
    environment variable still wins). Not everything that needs a per-machine
    setting reads this module: the GPU pinning and kernel toggles in tts.py are
    read straight out of os.environ, and have to be set before torch loads. If
    the file only populated this dict, install.sh could write
    HSA_OVERRIDE_GFX_VERSION or COMPANION_NO_FAST_GEMV into .env and they would
    silently do nothing.
    """
    values = {}
    try:
        with open(_ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                values[k] = v
                os.environ.setdefault(k, v)
    except OSError:
        pass
    return values


_FILE = _load_env_file()


def _get(key, default):
    return os.environ.get(key) or _FILE.get(key) or default


# What she calls you. Appears in her character prompt, in the transcripts she
# writes, and in what she's asked while reflecting - so it should be the name
# you'd actually want said out loud, not a username.
USER_NAME = _get("COMPANION_USER", "the user")

# Her name. There is deliberately no default: naming her is the operator's
# call, not a value shipped in a config file, and a repo that picks one is
# quietly asserting whose companion this is. ensure_configured() asks on first
# run. Changing it later does NOT make her someone else - identity lives in the
# character prompt and in memory, not in the label - but she should be called
# something from the start, because it appears in her own prompt.
ASSISTANT_NAME = _get("COMPANION_NAME", "")

# Where the LLM lives. The second one is optional: background thinking
# (consolidation, reflection, thread pressure) goes here so it can't compete
# with speech synthesis for the GPU. If it's absent everything falls back to
# the first, slower and noisier but working.
API_URL = _get("COMPANION_API_URL", "http://127.0.0.1:8080/v1")
BACKGROUND_API_URL = _get("COMPANION_BACKGROUND_API_URL", "http://127.0.0.1:8081/v1")


def display_name():
    """Her name for anything user-visible, with a neutral stand-in if unset.

    Only reachable when nobody has ever answered the first-run question - a
    piped or daemonised run, where ensure_configured() deliberately declines to
    block on a question that cannot be answered. The character prompt handles
    this differently: rather than assert this stand-in as her name, it omits
    the identity line entirely. Better to be unnamed than misnamed.
    """
    return ASSISTANT_NAME or "Companion"


def configured():
    """True once both names have actually been set.

    Not fatal - she works fine addressing you generically - but a fresh clone
    should be told once rather than silently sounding like it's talking to
    someone else.
    """
    return USER_NAME != "the user" and bool(ASSISTANT_NAME)


def _write_env(pairs):
    """Merge keys into .env, preserving whatever else is already in it.

    Rewritten rather than appended so running this twice doesn't leave two
    COMPANION_NAME lines with the later one silently winning.
    """
    existing = []
    try:
        with open(_ENV_FILE) as f:
            existing = f.read().splitlines()
    except OSError:
        pass
    kept = [ln for ln in existing
            if not any(ln.strip().startswith(k + "=") for k in pairs)]
    with open(_ENV_FILE, "w") as f:
        for line in kept:
            f.write(line + "\n")
        for k, v in pairs.items():
            f.write(f"{k}={v}\n")


def ensure_configured(stream=None):
    """Ask for the two names on first run, once, and remember the answers.

    Deliberately at first run rather than at install: what to call her is a
    decision about the character, not about the machine, and it has to be asked
    of people who clone and run directly or who installed non-interactively.
    install.sh can pre-seed .env, in which case this is a no-op.

    Silently does nothing when there is no terminal to ask on - a daemon or a
    piped run must not block forever waiting for an answer nobody can give.
    """
    global USER_NAME, ASSISTANT_NAME
    if configured():
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    out = stream or sys.stderr
    pairs = {}
    print("\nFirst run - two questions, asked once and saved to .env.\n", file=out)
    if USER_NAME == "the user":
        print("  What should she call you? This is said out loud, so use the name", file=out)
        print("  you'd actually want spoken, not a username.", file=out)
        answer = input("  > ").strip()
        if answer:
            USER_NAME = answer
            pairs["COMPANION_USER"] = answer
    if not ASSISTANT_NAME:
        print("\n  And what do you want to call her? There's no default on purpose.", file=out)
        print("  It goes into her own character prompt, so she'll use it about herself.", file=out)
        answer = input("  > ").strip()
        if answer:
            ASSISTANT_NAME = answer
            pairs["COMPANION_NAME"] = answer
    if pairs:
        os.environ.update(pairs)
        try:
            _write_env(pairs)
            print(f"\nSaved to {_ENV_FILE}. Edit that file to change either.\n", file=out)
        except OSError as e:
            print(f"\nCould not write {_ENV_FILE} ({e}) - you'll be asked again.\n", file=out)
