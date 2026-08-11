# A local voice companion

A fully local, always-on voice assistant: you speak, she hears the raw waveform,
thinks, and answers in a cloned voice. No cloud, no NVIDIA, no speech-to-text stage.

Everything runs on one consumer GPU.

---

## 1. Hardware and platform

| | |
|---|---|
| GPU | AMD Radeon RX 6800 XT — gfx1030, RDNA2, 16GB |
| CPU | Ryzen 7 9800X3D |
| OS | Ubuntu 26 LTS |
| Compute | ROCm 7.2.2 (no `HSA_OVERRIDE_GFX_VERSION` needed — gfx1030 is natively supported) |
| Audio | ALSA/PipeWire, `arecord` in, `aplay` out |

Python env: 3.12.13, in a virtualenv or conda env. `install.sh` creates one
at `.venv/` beside the checkout unless you point it elsewhere.

| Package | Version |
|---|---|
| torch | 2.13.0**+rocm7.2** |
| transformers | 5.14.1 |
| peft | 0.20.0 |
| llama.cpp | `fc3f10b38`, built from source with HIP backend |

> **The single most important env rule:** every torch-adjacent package must be the
> ROCm build. pip will happily resolve CUDA wheels that fail on `libcudart.so.*`.
> This bit twice — see §6.

---

## 2. The stack

```
   microphone
       │  arecord, 16 kHz mono raw PCM
       ▼
┌──────────────────┐
│  Silero VAD      │  listen.py — CPU only, ~0.35 ms per 32 ms frame
│  (TorchScript)   │  detects speech, segments turns, triggers barge-in
└────────┬─────────┘
         │  wav bytes (base64)
         ▼
┌──────────────────┐
│  Audio-capable   │  llama-server, HIP backend, port 8080
│  LLM (~12B Q4)   │  native audio input via mmproj — NO Whisper/STT
│  + MTP draft     │  OpenAI-compatible streaming API
└────────┬─────────┘
         │  text deltas, streamed
         ▼
┌──────────────────┐
│  chunker         │  companion.py — splits reply into speakable units
└────────┬─────────┘
         │  sentence / clause chunks
         ▼
┌──────────────────┐
│  Sesame CSM-1B   │  tts.py + stream_tts.py — PyTorch ROCm
│  + voice adapter │  emits audio frames at 12.5 Hz while still generating
└────────┬─────────┘
         │  raw PCM, streamed
         ▼
   aplay ──▶ headphones
```

GPU sharing: llama-server stays resident (~9.2GB), CSM loads in the same process
as companion.py (~4.5GB). Both live on the one card simultaneously.

### Models

| Role | Model | Size |
|---|---|---|
| Brain | any audio-capable LLM + mmproj (measured on a 12B at Q4_K_M) | ~6.9GB |
| Speculative draft | matching MTP head, if the model ships one (optional) | ~254MB |
| Audio/vision encoder | the model's `mmproj-*.gguf` (**required** for audio) | ~175MB |
| Voice | `sesame/csm-1b` (**gated** — accept the license) | ~6.7GB |
| Voice character | a CSM LoRA — see README for options; **check its speaker_id** | ~54MB |
| Turn-taking | Silero VAD (bundled `.jit`) | 2.3MB |

Disk: ~7.3GB under `<prefix>/models`, ~6.8GB in the Hugging Face cache.

---

## 3. Files

All in the checkout directory.

| File | Lines | Role |
|---|---|---|
| `companion.py` | 666 | Main app: turn loop, chunking, barge-in, greeting, history |
| `stream_tts.py` | 319 | Streaming synthesis + playback buffer management |
| `listen.py` | 228 | Always-on mic, Silero VAD, turn segmentation, mic health check |
| `tts.py` | 260 | CSM model loading, text normalization, one-shot CLI |
| `fast_gemv.py` | 59 | rocBLAS GEMV workaround (§5.1) |
| `fast_depth.py` | 70 | Depth-decoder generate() bypass (§5.4) |
| `config.py` | 166 | Per-operator settings, `.env` loading, first-run naming |
| `context.py` | — | Assembles each turn's messages; cache order and the GUARD |
| `memory.py` | — | Two-tier durable memory; end-of-session distillation |
| `session.py` | — | Per-turn journal, replayed on restart (§8) |
| `archive.py` | — | SQLite FTS5 search over the derived corpus |
| `prosody.py` | — | How each side sounded, measured from the PCM (§3.1) |
| `miccheck.py` | — | Microphone calibration; run it before first use |

The llama.cpp build lives outside the checkout, under the installer's
`--prefix` (default `../runtime/llama.cpp`), so the repo holds no build
artifacts and no model weights.

### 3.1 Prosody, both directions

Measured from the waveform, never from the audio encoder — that was tried and
does not work. Gemma's encoder is ASR-trained: it missed a 1.5x speedup
entirely, could not tell full amplitude from 15% amplitude (ASR normalizes
level as preprocessing, so loudness is gone before the model sees it), and it
would agree the same clip was sad, angry, or joyful depending on which you
suggested. A reading you can talk out of itself is not a perception.

So `prosody.py` computes loudness, pitch, pitch variation, pause ratio and
speaking rate directly from the PCM. Two speakers, two baselines:

| | |
|---|---|
| **her own reply** | folded in after synthesis; surfaced on the *next* turn, since she can't hear herself before she speaks |
| **the incoming turn** | measured at capture; surfaced on the turn it describes, immediately before that message |

Both only speak up when something departs from that speaker's own norm, and
only after 8 utterances have established what "usual" means. Ordinary turns
produce nothing.

Two things this is not. It is **not speaker recognition** — it compares a voice
to how that voice has usually sounded, and cannot tell one speaker from
another. Which baseline an utterance lands in comes from the channel it arrived
on, microphone or synthesizer, not from the voice. And it is **not diagnosis**:
tired, unwell, preoccupied and simply concentrating all flatten a voice
similarly, so the note reports the signal and explicitly withholds the reading.

One measurement is missing on the incoming side. `rate_wps` needs a word count,
and voice turns aren't transcribed until the session ends (per-turn
transcription would evict the live KV cache — see §4). Loudness, pitch, pitch
variation and pausing all come straight from the waveform and are unaffected.

### Responsibilities

**`companion.py`** — owns the conversation. Streams from the LLM, cuts the text into
speakable chunks, hands them to synthesis, keeps history, and decides when a
turn has been interrupted.

**`stream_tts.py`** — `FrameStreamer` receives audio frames from CSM's
`generate()` via the `streamer` hook and decodes them incrementally.
`ReplyPlayer` owns one `aplay` process per reply, fed by a writer thread, and
tracks how far ahead the buffer is running.

**`listen.py`** — `VoiceListener` runs `arecord` continuously, scores each 32ms
frame with Silero, and emits a complete utterance after `silence_s` of quiet.
`calibrate()` refuses to enable voice mode if the mic is unusable.

**`tts.py`** — loads CSM + LoRA (merged), applies both perf patches, and holds
`normalize_for_speech()`, which every path uses.

---

## 4. Running it

Start the LLM server. `install.sh` writes a `start-llama.sh` under its
`--prefix` with these flags already filled in for your hardware; this is what
it contains. It is a process, not a service — a reboot loses it.

```bash
nohup env HIP_VISIBLE_DEVICES=0 <prefix>/llama.cpp/build/bin/llama-server \
  -m <model>.gguf \
  -md <model>-mtp.gguf --spec-type draft-mtp \
  --mmproj mmproj-<model>.gguf \
  -ngl 99 -fa on -c 16384 -ub 512 --port 8080 --reasoning off --parallel 1 \
  --temp 0.6 --top-k 64 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 &
```

Then:

```bash
# voice mode (always-on mic)
python companion.py --voice --timing

# typed mode, push-to-talk available with a bare Enter
python companion.py

# one-off text to wav
python tts.py "Hello there." -o out.wav
```

### Load-bearing server flags

| Flag | Why it matters |
|---|---|
| `--reasoning off` | Gemma 4's template defaults reasoning **on** — ~100 hidden tokens before every reply. Removing it was an **8x** latency cut, the single biggest win in the build. |
| `--parallel 1` | Default 4 slots round-robin with separate KV caches, so the ~1060-token system prompt kept getting re-prefilled (one request cost 777ms on a partial miss). |
| `-c 16384` | **Do not raise.** At 32K, concurrent free VRAM fell to ~0.7GB and CSM stalled for seconds mid-word. See §6.2. |
| `-fa on` | Flash attention. Already maxed; there is no gfx1030 FlashAttention kernel beyond this. |

### CLI flags (`companion.py`)

| Flag | Default | Notes |
|---|---|---|
| `--voice` | off | Always-on mic + VAD. Assumes **headphones**. |
| `--silence SEC` | 4.0 | Quiet needed to end your turn. Lower = snappier. |
| `--min-speech SEC` | 0.35 | Ignore shorter bursts. |
| `--mic DEV` | system default | ALSA capture device. |
| `--timing` | off | Prints time-to-first-sound, split LLM vs synth. |
| `--max-first-words N` | 0 (off) | Force-break a long opening sentence for faster first audio. |
| `--no-greeting` | off | Skip the spoken greeting (still warms models). |
| `--api-url URL` | `127.0.0.1:8080/v1` | Any OpenAI-compatible server. |

---

## 5. Optimizations (all measured)

### 5.1 rocBLAS picks catastrophic kernels for batch-1 GEMV — `fast_gemv.py`

During decode every `nn.Linear` sees a single token, so each matmul is
`[1,K] @ [K,N]` — a GEMV. rocBLAS selects a `MT128x256x16` macro-tile kernel.
At M=1, N=1024 that's a grid of **4 workgroups on a 72-CU GPU** (~5% utilization),
each grinding a long serial K reduction.

Measured on CSM's MLP `down_proj` (`[1,8192]@[8192,1024]`): **629µs against a
~39µs bandwidth roofline** — and 61% of all matmul time.

`torch.einsum('ij,j->i', W, x)` sidesteps the heuristic:

| Shape | `F.linear` | `einsum` | speedup |
|---|---|---|---|
| down_proj `[1,8192]@[8192,1024]` | 584.6µs | **32.5µs** | **18.0x** |
| up/gate `[1,1024]@[1024,8192]` | 142.8µs | 51.6µs | 2.8x |
| q/o_proj `[1,1024]@[1024,1024]` | 14.9µs | 9.1µs | 1.7x |
| kv_proj `[1,1024]@[1024,256]` | 13.7µs | 5.5µs | 2.5x |

Accuracy is unchanged — against an fp32 reference both show max error 0.11420
and mean relative 1.76e-04. Not bit-identical (different accumulation order),
but neither is closer to truth.

The GPU itself is healthy: 88% of peak on large GEMM, 83% of peak memory
bandwidth, 3.19µs kernel launch. **It is purely rocBLAS's shape heuristic.**

`fast_gemv.apply(model)` is model-agnostic — it swaps `nn.Linear.forward` only
when the input is a true single-token vector, so prefill and multi-token paths
are untouched. Reusable for any PyTorch model on this card.

Applied to 238 Linear layers. End-to-end: **RTF 3.78x → 1.15x** on a 25-word
utterance.

### 5.2 Streaming synthesis — `stream_tts.py`

CSM emits audio frames at 12.5 Hz, exposed through `generate(streamer=...)`.
Two facts make streaming viable:

- Full codec decode of a whole utterance costs only ~27ms.
- The decoded **prefix is stable**: `decode(0:k)` matches the final decode over
  the whole overlap to max error 7e-4 (inaudible).

So each flush re-decodes from frame 0 and emits only the new tail.
Decoding disjoint chunks and concatenating them is **not** seamless
(max error 0.39 — audible clicks), so this deliberately re-decodes from the start.

Time to first sound for CSM alone: **~2.4s → 0.39s**.

### 5.3 Adaptive flush backoff — the cubic bug

Mimi's decoder is **quadratic in frame count**. Re-decoding at a fixed 4-frame
interval makes total decode cost effectively **cubic**:

| utterance | decode time | share of wall |
|---|---|---|
| 9 words | 0.20s | 6% |
| 20 words | 0.30s | 3% |
| **37 words** | **13.14s** | **42%** |

Short replies hid it entirely — which is why it presented as "fine at first,
choppy toward the end of long answers."

`flush_interval()` keeps flushes frequent for the opening ~1.3s (that's what
sets TTFS) then backs off: 4 → 12 → 24 frames.

**Decode on the 37-word case: 13.14s → 0.12s (110x).** Wall time 31.2s → 13.2s.

### 5.4 Depth-decoder generate() bypass — `fast_depth.py`

CSM's `_sample` calls `depth_decoder.generate()` once per audio frame
(12.5x per second of audio). Each call rebuilds a `GenerationConfig`,
`LogitsProcessorList` and `StoppingCriteriaList` before doing any work —
~6ms/frame of pure setup, ~6.5% of synthesis time.

Replaced with a direct forward loop reproducing the same sampling. Verified:

- **Greedy: token sequences match exactly, 6/6 trials**
- **Sampling with identical seed: match exactly, 0/33 differing tokens, 6/6 trials**

> Note the depth decoder **samples** (do_sample=True, temp 0.9, top_k 50)
> regardless of the `do_sample` passed to the top-level `generate()` — that flag
> only controls the backbone.

### 5.5 Buffer-lead tracking and catch-up pauses

Generation runs slightly slower than playback, so a long multi-sentence reply
slowly drains the buffer. `ReplyPlayer.lead()` reports how far ahead the stream
is; when it drops below `TARGET_LEAD_S`, `companion.py` stretches the pause at the
**next sentence boundary** (capped at `MAX_CATCHUP_GAP_S`), where a pause reads
as thinking rather than a glitch.

Combined with §5.3 and the 14-word chunk cap, on a 6-chunk 26-second reply:
**0/31 samples starved**, buffer never below +0.37s, RTF 1.043x. Before these
fixes the same test starved on half its samples.

### 5.6 Rejected after measurement

| Attempt | Result |
|---|---|
| `torch.compile` on depth decoder | **0.43–0.79x — slower.** KV cache reshapes every step, blowing the recompile limit |
| `StaticCache` | **0.95x — slower.** Attention spans all 32 padded slots every step |
| KV cache quantization | Caches are 128KB (depth) / 1.56MB (backbone). Nothing to save |
| CPU-offloading the codec | Codec is 1.9% of GPU work; CPU is **19x slower** on the TTFS-critical first flush |
| Fusing gate+up / q+k+v projections | Only 2.14 ms/frame |
| `PYTORCH_ALLOC_CONF=expandable_segments` | Halved stalls under pressure but **crashes intermittently** inside Mimi decode on ROCm 7.2 |

---

## 6. Bug fixes

### 6.1 Curly punctuation garbled the voice

The LLM emits typographic punctuation; CSM's tokenizer only has clean tokens for
ASCII forms:

| Text | Tokens |
|---|---|
| `It's` | `['It', "'s"]` — what the LoRA trained on |
| `It’s` | `['It', 'âĢĻs']` — byte-level token it never learned to pronounce |

Audible as garbled sound on possessives. The LoRA's own model card also lists
`" ; ( ) [ ] /` as unreliable — confirmed audibly on quotes and semicolons.

`normalize_for_speech()` in `tts.py` folds all of it to speakable ASCII
(curly→straight quotes, dashes→commas, `;`→`,`, quotes dropped, `…`→`...`,
`/`→"or"), then collapses doubled/leading commas. Applied in **both** the CLI
and streaming paths.

### 6.2 VRAM starvation caused multi-second mid-word stalls

Raising llama-server to `-c 32768` left ~0.7GB free. CSM's streaming decode
allocates a growing transient buffer each flush; near-zero headroom made the
driver evict and migrate memory (visible as `svm_range_restore_work` churn in
`dmesg`), stalling generation in 1–4s bursts that worsened as the utterance grew.

Reproduced by squeezing free VRAM to 0.7GB: **70.8s to generate 10s of audio,
gaps to 3.7s**. Fixed by reverting to `-c 16384 -ub 512` (~450MB back, headroom
0.7GB → 2.9GB) and resizing the history cap to 120 messages.

### 6.3 Fragment chunks sounded clipped

`SENTENCE_BOUNDARY` counted the `.` in an ellipsis as a sentence end, so
*"It's... actually quite peaceful"* split after **"It's"** — a one-word
utterance with its own prefill and abrupt start/stop.

Fixed with a negative lookbehind `(?<=[.!?])(?<!\.\.)\s+`, plus
`MIN_CHUNK_WORDS = 3` so short sentences merge forward.

### 6.4 No overlap between sentences

An earlier per-chunk player blocked on `close()` until the sentence finished
playing, so synthesis of sentence N+1 didn't start until N had fully played —
a ~0.4s hole before every sentence. `ReplyPlayer` now owns one `aplay` per reply
with a writer thread, so the next sentence generates while the current one plays.

### 6.5 Railed microphone produced phantom turns

Front-panel mic ran with **+30dB capture gain + 30dB boost = +60dB**, saturating
the input from electrical noise alone (idle RMS 1.00, 100% clipped). A constant
full-scale signal reads as speech to any VAD, generating a phantom "utterance"
every ~1.4s, each interrupting her — an unusable feedback spiral.

Fixed at the source with `amixer` (boost 0, capture ~40%): idle RMS **1.00 →
0.06**, clipped **100% → 0%**, VAD triggers on silence **continuous → 0/187**.

`listen.py` now hard-fails at startup rather than spiralling. The check tests
what actually matters — *does the detector fire on silence?* — rather than raw
level, because a noisy-but-clean mic works fine while a clipped one never can.

### 6.6 `silero-vad` pulls a CUDA torchaudio

`pip install silero-vad` installs **torchaudio**, and pip resolves the **CUDA**
build → `OSError: libcudart.so.13`. torch itself survived, but this is the exact
trap that starts every AMD build.

torchaudio was uninstalled. `listen.py` imports **only** the bundled
`silero_vad/data/silero_vad.jit` via `torch.jit.load(..., map_location="cpu")`.

> The bundled `.onnx` exports are also unusable for streaming: `stateN` does not
> round-trip as the next call's `state` input, so carried state corrupts the
> LSTM and real speech scores **0.03** instead of 1.00. Caught only by testing
> against known speech — "it runs" would have shipped a mic that never triggers.

### 6.7 Barge-in truncation

When interrupted, history keeps only what she **audibly said**: nothing if cut
before any sound, the partial chunk with a cut marker if mid-sentence, completed
sentences otherwise. Interrupt latency is one audio frame (~50–80ms).

---

## 7. Measured performance

Time to first sound, warm, typical conversational turns:

| | |
|---|---|
| Best observed | **0.90s** |
| Typical | **1.0 – 1.4s** |
| Split | LLM 0.42–0.69s + synth 0.41–0.79s |

Speech generation, end to end through the streaming path:

| | |
|---|---|
| Current | **1.04 – 1.19x realtime** |
| Before optimization | 3.78x realtime |

Component detail:

| Metric | Value |
|---|---|
| CSM model load | 2.75s (one-time) |
| LLM first token (prompt cached) | ~0.30s |
| LLM first speakable chunk | ~0.5s |
| CSM first audio | ~0.39s |
| Codec decode, whole utterance | ~27ms |
| Silero VAD | 0.35ms per 32ms frame (CPU) |
| VRAM: llama-server resident | ~9.2GB |
| VRAM: CSM transient | ~4.5GB |
| VRAM: free with both live | ~2.9GB |

Voice input: Gemma 4 transcribes a 7.4s clip in ~1.8s through the already-loaded
mmproj — no extra VRAM, no separate STT model.

---

## 8. Known issues and limits

- **`sesame/csm-1b` is gated.** Accept the license and `hf auth login` before
  first use.
- **~~Transient `HSA_STATUS_ERROR_EXCEPTION`~~ — found and fixed.** It looked
  like a driver fault (intermittent, ~1 in 3 sessions, killed CSM mid-synthesis
  with a HIP "unspecified launch failure"). It was a bug in `FrameStreamer.put`:
  `generate()` pushes every tensor through the streamer hook, and one that
  happened to be `num_codebooks` wide passed the shape+EOS filter, got
  accumulated as audio, and indexed Mimi's 2048-entry codebook with text-range
  ids. Out-of-bounds GPU gather. Fixed by also checking the value range.

  Two things made it hard, both worth remembering. **The stack trace lied** —
  HIP reports async kernel errors at the next API call, so every trace blamed
  `SetDevice` in an unrelated memory copy; `AMD_SERIALIZE_KERNEL=3` was what
  finally pointed at the real kernel. And **synthetic stress never reproduced
  it** — 95 clean generations with fixed prompts, because triggering it needs a
  non-audio tensor to coincidentally be 32 wide, which depends on what is
  actually being generated.
- **Voice mode assumes headphones.** On speakers the mic hears her and barge-in
  becomes a feedback loop. Would need half-duplex gating or echo cancellation.
- **Streaming output is ~5dB quieter** than the wav path, which normalizes to
  0.98 peak. Streaming doesn't normalize (a per-chunk peak would cause volume
  jumps). Raw peaks measured 0.33–0.57, so there's headroom if wanted.
- **Sub-1.0x realtime not reached.** The depth-decoder loop alone is 0.860x, but
  ~49% of a forward is many small non-matmul ops (attention `bmm`, RMSNorm,
  rope, KV-cache growth). Closing that needs hand-written fused kernels; the
  roofline says even perfect linear kernels only save ~4.4 ms/frame against the
  ~7 needed.

---

## 9. Next steps

1. **Triton fused kernels** to cross 1.0x realtime. Would make TTFS a flat
   ~0.9s regardless of sentence length (prebuffer → 0), and is a prerequisite
   for true full-duplex.
2. **Echo cancellation** for speaker use (PipeWire `module-echo-cancel`).
