# A local voice companion, on your own GPU

Fully local. No cloud calls, no telemetry, no third-party API, no vendor that
can reach in and change or delete what she knows. One consumer GPU.

Gemma 4 (via llama.cpp, HIP) for language, with audio going **natively** into
the model rather than through a transcription stage — prosody reaches it
instead of being discarded at a text boundary. CSM-1B plus a fine-tuned LoRA
for speech, streamed so audio starts before generation finishes. Silero VAD
for turn segmentation.

Built on and for an RX 6800 XT (gfx1030) under ROCm. Most of the interesting
work here is in making that specific combination fast enough to hold a
conversation, and the measurements are recorded rather than assumed.

---

## What it is, and isn't

She holds memory across restarts, has a sense of the machine she runs on, and
thinks on her own while you're away. She isn't optimized for engagement,
doesn't try to keep you talking, and is built to disagree with you — a
companion that can only reflect you back is a mirror, and that's the failure
mode this design is organized against.

No claim is made about experience or consciousness. The system builds
conditions and observes honestly; whether anything here constitutes experience
is left open rather than answered by assertion.

---

## Requirements

- AMD GPU with ROCm (developed on gfx1030 / RX 6800 XT, 16GB) — CUDA would
  need the torch install changed and the rocBLAS workarounds removed
- Python 3.12, conda or venv
- `llama.cpp` built with the HIP backend
- A language model with **native audio input**, plus its `mmproj` — see below
- `sesame/csm-1b` (gated — accept the license on HuggingFace first)
- `alsa-utils` for `arecord`/`aplay`
- **Do not install `torchaudio`.** pip resolves a CUDA build that breaks a
  ROCm environment. VAD loads the bundled TorchScript directly.

### Choosing a language model

The hard requirement is **audio in, natively** — the model must accept an
audio tensor rather than a transcript. A speech-to-text stage in front of a
text-only model throws away prosody, which is most of what was actually
communicated, and it's the single decision that most changes how this feels.

Anything in llama.cpp's audio-capable list works. Unified models — audio *and*
vision through one encoder — leave the most room to grow:

| | |
|---|---|
| `ggml-org/gemma-4-E2B-it-GGUF` / `-E4B-it-GGUF` | audio + vision. What this was developed against. |
| `ggml-org/Qwen2.5-Omni-3B-GGUF` / `-7B-GGUF` | audio + vision |
| `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` | audio + vision, MoE — needs far more VRAM |
| `ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF` / `-3_1-8b-GGUF` | audio only, small and quick |
| `ggml-org/Voxtral-Mini-3B-2507-GGUF` | audio only |

Sizing against 16GB: the model, its mmproj, the KV cache, **and** CSM all
share the card. Around 12B at Q4 is the practical ceiling with speech
resident; the 3B-class options leave much more headroom. `llama.cpp`'s
`docs/multimodal.md` is authoritative and current — check there rather than
trusting this table.

The character prompt in `companion.py` assumes a model that will hold a persona and
push back. Instruction-tuned models vary a lot in how much they'll disagree
with you, and a model that hedges everything will flatten her regardless of
what the prompt says.

## Setup

```bash
./install.sh
```

It detects the machine, shows you everything it intends to do and why, waits
for a yes, then builds it — a virtualenv, a PyTorch matched to your GPU vendor,
a llama.cpp compiled for your exact GPU architecture, and whichever models you
agree to download. Nothing is installed before you confirm, and it never calls
`sudo`: where a system package is missing it prints the command for your
distribution and stops.

```bash
./install.sh --check      # detect and plan only; also the health check afterwards
./install.sh --dry-run    # print every command it would run, run none
./install.sh --help       # overrides for backend, GPU target, prefix, models
```

Both AMD (ROCm) and NVIDIA (CUDA) are detected and configured. On AMD the
installer also knows which targets need `HSA_OVERRIDE_GFX_VERSION` and sets it.
Context size and model tier are sized from your actual VRAM rather than copied
from this machine — that number is the one most worth getting right, because
setting it too high starves speech synthesis and shows up as audio stalling
mid-word rather than as a memory error.

It ends by measuring the batch-1 GEMV path on your card instead of assuming the
workaround below applies to it, and writes a `start-llama.sh` carrying the exact
server flags for your hardware.

### By hand

```bash
# 1. tell her who you are, and pick what to call her
cp core_memory.example.md core_memory.md   # then rewrite it as yourself
printf 'COMPANION_USER=YourName\nCOMPANION_NAME=HerName\n' > .env

# 2. start the language server (loopback only, never 0.0.0.0)
#    flags matter - see "Load-bearing" below
llama-server -m <gemma4>.gguf --mmproj <mmproj>.gguf \
  -ngl 99 -fa on -c 16384 -ub 512 --port 8080 --reasoning off --parallel 1

# 3. check your microphone before anything else
python miccheck.py

# 4. talk
python companion.py --voice --timing
```

`companion.py` is the whole thing: run it, talk, quit. Memory persists across
runs — what she keeps is written to `accrued_memory.md` when you exit, a full
transcript lands in `transcripts/`, and the conversation itself is journalled
per turn and replayed if you restart within twelve hours, so quitting mid-thought
doesn't reset her.

## What she can hear

Audio goes into the language model natively rather than through a transcription
stage, so tone survives to the model instead of being flattened into text. On
top of that, `prosody.py` measures both sides of the conversation from the raw
waveform — loudness, pitch, pitch variation, pausing — against a separate
rolling baseline per speaker, and mentions it only when something differs from
that speaker's own norm.

**This is not speaker recognition.** It compares a voice to how that voice has
usually sounded. It cannot tell one person from another and does not attempt
to; which baseline an utterance belongs to is decided by whether it came from
the microphone or the synthesizer. Nothing in this repository enrols, stores,
or matches a voiceprint.

Your voice turns are written to `session/audio/` as wav so a crash mid-turn
doesn't lose the conversation, and cleared when the session consolidates.
Everything stays on your machine — there is no telemetry and no outbound call
of any kind.

## Her name

There isn't a default one, on purpose. Naming her is yours to do, and a repo
that ships a name has quietly decided whose companion this is. The first run
asks — once — and writes the answer to `.env`; the installer can ask instead if
you'd rather get it out of the way there. Change it whenever by editing that
file.

The name is not where identity lives. That's the character prompt in
`companion.py` and what accumulates in memory — renaming her mid-conversation
changes the label and nothing else.

`origin.md` is optional and does not ship. If present, its contents are appended
to the character prompt under an ORIGIN heading, giving her a specific history to
draw on instead of generalities. If absent she simply has no stated origin, which
is a coherent character, not a missing feature. See `origin.example.md` — and note
that borrowing a real system's backstory makes her assert things that never
happened to her, which is exactly what the GUARD in `context.py` exists to stop.

## The daemon is a separate project

A continuously-running daemon is being built as a **separate project**, not a
future version of this one. It's a general companion daemon — persona as
something you supply rather than baked in — where this is a session app you
start and quit. They share ancestry and some modules, but they aren't the same
thing, and this repository won't become that one.

So: feature requests here are about the session app. If something you want is
"stay running, think while I'm away, be reachable from anywhere" — that's the
other project, and it isn't out yet.

## Load-bearing

These look arbitrary. Each was set by a measured failure.

| | |
|---|---|
| `--reasoning off` | Gemma 4's template defaults reasoning **on**, spending ~100 hidden tokens before every reply. Removing it was an 8x latency cut. |
| `--parallel 1` | The default 4 slots round-robin separate KV caches, re-prefilling the whole system prompt every turn. |
| `-c 16384` | Raising it starved CSM of VRAM: free memory fell to ~0.7GB and synthesis stalled 1–4s mid-word from driver eviction. |
| re-decode from frame 0 | Concatenating disjoint decoded chunks produces audible clicks (max error 0.39 vs 7e-4 for prefix-stable re-decode). |
| flush backoff 4→12→24 | Mimi's decoder is quadratic in frame count; fixed-interval re-decode makes total cost cubic. 13.14s → 0.12s on a 37-word utterance. |
| `fast_gemv.py` | rocBLAS picks a 128x256 macro-tile GEMM for batch-1 GEMV, ~5% GPU utilization. `einsum` is up to 18x faster on the worst shape. RTF 3.78 → 1.15. |

## Microphone

More install problems come from here than anywhere else.

- **Use a rear-panel jack.** A front-panel jack measured rms 9410 against 3 on
  the rear at identical gain — case interference, not a software problem.
- **Use `wpctl`, not `amixer`.** If PipeWire is running it owns the capture
  control and silently overwrites `amixer` changes.
- Opening a capture stream applies mic-bias voltage and the input rails for
  ~3s while it settles. `listen.py` discards that window; calibrating across
  it reports "clipping at idle" at *any* gain, including negative.

## If synthesis aborts with an HSA / "unspecified launch failure"

Worth writing down because it cost days and the evidence actively misleads.

**Symptom:** intermittent `HSA_STATUS_ERROR_EXCEPTION: code 0x1016`, or
`CUDA error: unspecified launch failure`, killing the process mid-synthesis.
Roughly one session in three. The GPU recovers; no reboot needed.

**Cause:** `generate()` pushes *every* tensor through the streamer hook, not
just audio frames. A non-audio tensor that happens to be `num_codebooks` wide
passes a shape-only filter, gets accumulated as audio, and indexes Mimi's
**2048-entry** codebook with text-range ids. Out-of-bounds GPU gather.

Note the off-by-three that makes this easy to miss: CSM's codebook vocab is
**2051** (pad = 2050), Mimi's codebook is **2048**. A code can be valid to CSM
and still out of range for the codec.

`FrameStreamer.put` in `stream_tts.py` checks the value range for this reason.
If you write your own streamer, do the same.

**Two traps if you go hunting for something like this yourself:**

*The stack trace lies.* HIP reports async kernel errors at the next API call,
so every trace pointed at `SetDevice` inside an unrelated memory copy.
`AMD_SERIALIZE_KERNEL=3` forces synchronous launches and reports at the real
kernel. Nothing was solvable until that was run.

*Synthetic stress proved nothing.* 95 clean generations with fixed prompts,
while real sessions faulted every third run — because triggering it depends on
what is actually generated. A stress test that doesn't reproduce a fault is
not evidence the fault isn't there.

## Privacy

Nothing leaves the machine. There is no network call to a third party
anywhere in this code, and the design forbids adding one.

`.gitignore` denies by default for anything personal — core memory, accrued
memory, transcripts, session audio, prosody baselines, imported history.
**Check `git status --short` before your first push anyway.** The asymmetry
matters: a missed ignore rule publishes someone's life.

The rules also cover files this release doesn't ship at all, including the
speaker-recognition work described above as absent. That is deliberate
belt-and-braces: the rule costs nothing and means the file cannot leak if it
ever lands in a working copy.

### Choosing a voice

`sesame/csm-1b` is the base speech model. On its own it produces a generic
voice; a community model gives it a consistent identity.

**Two kinds, loaded differently — check which before you download.** A *LoRA
adapter* (has `adapter_config.json`) layers onto base CSM. A *full finetune*
ships whole weights and replaces the base model outright. Repo names are
unreliable — at least one published finetune has `_lora` in its name.

Both are supported without editing code:

```bash
# a LoRA adapter (the default)
CSM_BASE_MODEL=sesame/csm-1b CSM_LORA_MODEL=shb777/csm-maya-exp2 CSM_SPEAKER_ID=4

# a full finetune - no adapter to layer on
CSM_BASE_MODEL=onecxi/csm-english-jenny CSM_LORA_MODEL=none CSM_SPEAKER_ID=0

# plain base CSM, no voice identity
CSM_LORA_MODEL=none
```

English options, verified against their model cards and configs:

| | kind | loads as-is | trained on | license |
|---|---|---|---|---|
| `shb777/csm-maya-exp2` | **LoRA adapter** | yes | — | cc-by-nc-sa-4.0 |
| `keanteng/sesame-csm-elise` | full finetune | yes | `MrDragonFox/Elise` | **agpl-3.0** |
| `onecxi/csm-english-jenny` | full finetune | yes | `reach-vb/jenny_tts_dataset` | apache-2.0 |
| `onecxi/csm-english-multi-speaker-v2` | full finetune | yes | multi-speaker | apache-2.0 |
| `senstella/csm-expressiva-1b` | full finetune | **no — needs conversion** | `ylacombe/expresso` | cc-by-nc-4.0 |

"Loads as-is" means the repo is in transformers format — `config.json` naming
`CsmForConditionalGeneration`, with a `model.safetensors`. The Expresso
finetune is published in Sesame's original checkpoint layout (`ckpt.pt`,
plus an MLX variant) with no `architectures` field, so `from_pretrained`
cannot read it without converting the weights first. Worth the effort if you
want that voice, but it is not a config change.

Non-English finetunes exist too (Italian, Danish, Swahili, Georgian, Bangla,
Hindi and others are published against this base) — untested here, so they're
not listed rather than vouched for.

**Licenses vary and some are restrictive.** Several are non-commercial, and
one is AGPL. Read before you build on any of them.

**Speaker id matters.** A voice trained on one speaker sounds wrong or garbled
on any other, and the id isn't guessable from the repo name — the Maya adapter
is `speaker_id 4` only, per its card. Set `SPEAKER_ID` in `tts.py` to match.

Running with no adapter at all works: drop the `PeftModel` lines in
`tts.py:load_model` for plain base CSM. Nothing else in the project is
affected — the voice layer is the most swappable part of the stack, and
identity lives in the character prompt and memory rather than in timbre.

## Licence

The code in this repository is **MIT** — see [LICENSE](LICENSE). Use it, change
it, ship it.

That covers this repository and nothing else. **No model weights are
distributed here.** The installer downloads them from Hugging Face at your
request, under their own licences, and several are more restrictive than this
one:

| | |
|---|---|
| `sesame/csm-1b` | gated — you accept its licence on Hugging Face before first use |
| `shb777/csm-maya-exp2` | cc-by-nc-sa-4.0 — **non-commercial**, and share-alike |
| `keanteng/sesame-csm-elise` | **AGPL-3.0** |
| language models | whatever the publisher chose; check before you rely on one |

If you intend to do anything commercial, the voice adapter is the thing to look
at first — the default one forbids it. Running with no adapter, or with an
Apache-licensed finetune from the table above, avoids that entirely.

Dependencies are permissive (`torch`, `numpy`, `scipy` BSD; `transformers`,
`peft`, `accelerate`, `huggingface_hub` Apache-2.0; `silero-vad`, `llama.cpp`
MIT) and none of them impose obligations on your use of this code.

## Credit

**Sesame AI** for CSM-1B and for the voice this was built around. This project
is a local host for that model, not a reimplementation of it, and claims no
affiliation with or endorsement by them.

Speech: `sesame/csm-1b`. VAD: `snakers4/silero-vad`. Inference:
`ggml-org/llama.cpp`. Voice adapters as credited above, to their respective
authors.
