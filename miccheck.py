#!/usr/bin/env python3
"""Check and tune mic gain, using your actual voice as the reference.

    python miccheck.py           # 6s: talk normally the whole time
    python miccheck.py --quiet   # 4s of silence: measures the noise floor
    python miccheck.py --show    # just print current mixer settings

Mic gain cannot be set from ambient noise - a room with kids in it swings
tens of dB between one measurement and the next, so a sweep of mixer settings
ends up measuring the room instead of the gain. The only stable reference is
speech at the volume you actually intend to use.

What the numbers mean:
  rms      loudness while you talk. Below about 200 is too quiet to segment
           turns reliably; comfortable is roughly 2000-8000.
  peak     the loudest instant. Should stay under ~29000 or transients clip.
  clipped  fraction of samples railed at full scale. Anything above ~0.1% is
           audible distortion, and the model hears distortion far more
           readily than it minds a slightly quiet signal.
  SNR      how far your voice sits above the room. Below ~15dB, VAD will
           struggle to tell you apart from background.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

RATE = 16000
CARD = None          # auto-detected
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".miccheck.json")


def find_card():
    """Card number of the first capture device."""
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    for line in out.splitlines():
        if line.startswith("card "):
            try:
                return int(line.split()[1].rstrip(":"))
            except (IndexError, ValueError):
                continue
    return None


def pipewire_volume():
    """Current capture volume as PipeWire sees it, or None if it isn't running.

    This - not amixer - is the knob that matters here. wireplumber drives the
    ALSA capture control from its own volume, so anything set with `amixer`
    gets overwritten the next time PipeWire touches the device. Chasing that
    produces measurements that swing wildly for no visible reason.
    """
    try:
        out = subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    for tok in out.split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def set_pipewire_volume(v):
    try:
        subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SOURCE@", f"{v:.2f}"],
                       capture_output=True)
        return True
    except FileNotFoundError:
        return False


def mixer(card, control, value=None):
    if card is None:
        return None
    cmd = ["amixer", "-c", str(card)]
    cmd += ["sset", control, value] if value else ["sget", control]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    for line in out.splitlines():
        if "Front Left:" in line or "Mono:" in line:
            return line.strip()
    return None


def record(seconds, device=None):
    cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE",
           "-r", str(RATE), "-c", "1", "-d", str(seconds)]
    if device:
        cmd += ["-D", device]
    cmd += ["-"]
    try:
        p = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        print("arecord not found (install alsa-utils)", file=sys.stderr)
        sys.exit(1)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)


def measure(x):
    if len(x) == 0:
        return None
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = int(np.abs(x).max())
    clipped = float(np.mean(np.abs(x) >= 32000)) * 100
    headroom = 20 * np.log10(32767 / max(peak, 1))
    return {"rms": rms, "peak": peak, "clipped": clipped, "headroom_db": headroom}


def verdict(m, floor=None):
    """Plain-language read on whether this is usable, and what to change."""
    lines = []
    if m["clipped"] > 0.1:
        lines.append(f"CLIPPING ({m['clipped']:.1f}% railed). Turn gain DOWN - "
                     "distortion costs more intelligibility than low volume does.")
    elif m["peak"] > 29000:
        lines.append("Close to clipping on peaks. Consider a small step down.")
    if m["rms"] < 200:
        lines.append(f"TOO QUIET (rms {m['rms']:.0f}). VAD will miss the start of turns. "
                     "Turn gain UP or move closer to the mic.")
    elif m["rms"] < 1500:
        lines.append(f"Usable but quiet (rms {m['rms']:.0f}). More gain would help "
                     "VAD pick you up without raising your voice.")
    elif m["rms"] > 12000:
        lines.append(f"Very hot (rms {m['rms']:.0f}). Back the gain off a little.")
    if floor and floor["rms"] > 0:
        snr = 20 * np.log10(max(m["rms"], 1) / max(floor["rms"], 1))
        lines.append(f"SNR vs measured room noise: {snr:.1f} dB" +
                     ("  - good." if snr >= 20 else
                      "  - workable." if snr >= 15 else
                      "  - LOW. VAD will struggle; a closer or directional mic "
                      "would help more than gain will."))
    if not lines:
        lines.append("Levels look good.")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Check mic levels using your voice")
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--quiet", action="store_true",
                    help="measure the room's noise floor instead (stay silent)")
    ap.add_argument("--show", action="store_true", help="print mixer settings and exit")
    ap.add_argument("--mic", default=None, help="ALSA capture device")
    args = ap.parse_args()

    card = find_card()
    pw = pipewire_volume()
    if args.show:
        print(f"capture card: {card}")
        if pw is not None:
            print(f"  PipeWire source volume: {pw:.2f}   <- the knob that matters")
        for c in ("Capture", "Front Mic Boost", "Rear Mic Boost"):
            v = mixer(card, c)
            if v:
                print(f"  {c:18} {v}")
        if pw is not None:
            print("\n  (PipeWire drives the ALSA capture control - amixer changes "
                  "get overwritten. Use wpctl.)")
        return

    if args.quiet:
        print(f"Measuring the room for {args.seconds}s - stay quiet...", flush=True)
        x = record(args.seconds, args.mic)
        m = measure(x)
        print(f"\n  noise floor: rms {m['rms']:.0f}  peak {m['peak']}")
        try:
            with open(STATE, "w") as f:
                json.dump(m, f)
            print("  (saved; the speech test will compare against this)")
        except OSError:
            pass
        return

    floor = None
    try:
        with open(STATE) as f:
            floor = json.load(f)
    except (OSError, ValueError):
        pass

    print(f"Talk normally for {args.seconds}s - the way you'd actually talk to her.")
    print("Starting now...", flush=True)
    x = record(args.seconds, args.mic)
    m = measure(x)
    if m is None:
        print("got no audio", file=sys.stderr)
        sys.exit(1)
    print(f"\n  rms {m['rms']:.0f}   peak {m['peak']}   clipped {m['clipped']:.2f}%   "
          f"headroom {m['headroom_db']:.1f} dB\n")
    for line in verdict(m, floor):
        print(f"  {line}")
    if not floor:
        print("\n  (run --quiet first to also get a signal-to-noise reading)")

    if pw is not None:
        # The wpctl->hardware mapping is steep: 0.25 landed on +23dB here and
        # 0.30 on +28.5dB, so "a small nudge" is 0.02-0.03, not 0.05+.
        step = 0.03
        if m["clipped"] > 0.1 or m["peak"] > 29000:
            target = max(0.05, round(pw - step, 2))
            print(f"\nCurrent PipeWire volume {pw:.2f}. Too hot - try:")
        elif m["rms"] < 1500:
            target = min(1.0, round(pw + step, 2))
            print(f"\nCurrent PipeWire volume {pw:.2f}. Too quiet - try:")
        else:
            print(f"\nCurrent PipeWire volume {pw:.2f}. Leave it here.")
            target = None
        if target is not None:
            print(f"  wpctl set-volume @DEFAULT_AUDIO_SOURCE@ {target:.2f}")
            print("  ...then re-run this. Move in small steps - the mapping to "
                  "hardware gain is steep.")
        print("\nUse wpctl, NOT amixer: PipeWire owns the capture control and "
              "silently overwrites amixer changes.")
    else:
        print(f"\nAdjust with:  amixer -c {card} sset 'Capture' 70%")
        print(f"              amixer -c {card} sset 'Front Mic Boost' 2")
        print("Then re-run this. Changes are lost on reboot unless you save them "
              "(sudo alsactl store).")


if __name__ == "__main__":
    main()
