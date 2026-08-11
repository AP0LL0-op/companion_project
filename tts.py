#!/usr/bin/env python3
"""CSM-1B + voice LoRA text-to-speech CLI.

Usage:
    python tts.py "Hello there, this is a test."
    python tts.py -f script.txt -o script.wav
    python tts.py "Hi!" -o hi.wav --sample --temperature 0.7

Base model: sesame/csm-1b (gated - run `hf auth login` and accept the
license at huggingface.co/sesame/csm-1b before first use).
LoRA: shb777/csm-maya-exp2 (speaker_id 4 only, per its model card).
"""
import os
import sys

# Everything down to the torch import has to happen before torch touches the
# GPU, because none of it can be changed afterwards.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402  reads .env into os.environ; imports nothing heavy

# Pin to one device. Which variable does that is vendor-specific, and a machine
# can present more than one agent: on the reference box device 1 is the CPU's
# tiny gfx1036 iGPU, and letting torch land there produces "no kernel image is
# available" rather than anything that names the real problem.
#
# setdefault throughout, so an explicit environment variable or a value written
# into .env by install.sh always wins over this fallback.
if os.path.isdir("/opt/rocm") or "HIP_VISIBLE_DEVICES" in os.environ:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# NOTE: PYTORCH_ALLOC_CONF=expandable_segments:True was tried here to soften
# low-VRAM allocator stalls (it halved them under pressure) but proved flaky on
# ROCm 7.2 - intermittent "SymIntArrayRef expected to contain only concrete
# integers" crashes inside Mimi's decode. Do not re-enable without retesting.
# The actual fix for the stalls was restoring VRAM headroom (llama-server at
# -c 16384 -ub 512 rather than -c 32768).

# Some AMD targets are not officially supported by ROCm and need
# HSA_OVERRIDE_GFX_VERSION to load the nearest supported kernel set - gfx1031/2
# and other RDNA2 parts want 10.3.0, gfx1103 wants 11.0.0. gfx1030 (RX 6800 XT)
# is officially supported and needs nothing; install.sh detects which case a
# machine is in and writes the override into .env when one is required, which
# config has already loaded by this point.
#
# If you are setting this up by hand and hit "no kernel image is available" or
# "invalid device function", that is the symptom this fixes.

import argparse
import re

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, CsmForConditionalGeneration

import fast_depth  # noqa: E402  skips the per-frame generate() wrapper in the depth decoder
import fast_gemv  # noqa: E402  routes batch-1 decode matmuls around rocBLAS's bad GEMV kernels

# The voice, in two parts. Community models come in two shapes and load
# differently, so both paths are supported here rather than left as an edit:
#
#   LoRA adapter  - has adapter_config.json. Leave BASE_MODEL at sesame/csm-1b
#                   and point LORA_MODEL at the adapter.
#   full finetune - ships whole weights. Point BASE_MODEL at it and set
#                   LORA_MODEL = None.
#
# Repo names are not reliable for telling them apart; at least one published
# finetune has "_lora" in its name. Check for adapter_config.json.
#
# A finetune must also be in transformers format - config.json listing
# `architectures: ["CsmForConditionalGeneration"]` and a model.safetensors.
# Some are published in Sesame's original checkpoint layout (ckpt.pt) instead,
# which from_pretrained cannot read without conversion first.
BASE_MODEL = os.environ.get("CSM_BASE_MODEL", "sesame/csm-1b")
_lora = os.environ.get("CSM_LORA_MODEL", "shb777/csm-maya-exp2")
LORA_MODEL = None if _lora.lower() in ("", "none", "0") else _lora

# Whose voice the adapter/finetune was trained on. NOT guessable from the repo
# name, and wrong values sound garbled rather than merely different - Sesame's Maya
# adapter is speaker_id 4 only, per its model card. Check yours.
SPEAKER_ID = int(os.environ.get("CSM_SPEAKER_ID", "4"))

FALLBACK_SAMPLE_RATE = 24000  # Mimi codec default, used only if we can't introspect it
SILENCE_GAP_S = 0.15
TOKENS_PER_WORD = 90  # calibrated from the model card's example: 5 words -> 375 tokens
MIN_TOKENS_PER_CHUNK = 375
MAX_TOKENS_PER_CHUNK = 2000
MAX_WORDS_PER_CHUNK = 40
PROBLEM_CHARS = set('()";[]/')  # noted in the LoRA's model card as generation-unfriendly

# LLMs emit typographic punctuation, but CSM's tokenizer only has clean tokens
# for the ASCII forms. "It's" -> ['It', "'s"] whereas "It’s" -> ['It', 'âĢĻs'],
# a byte-level token the LoRA never learned to pronounce - it comes out garbled.
# Map every typographic form back to ASCII before synthesis.
UNICODE_PUNCT = {
    "’": "'", "‘": "'", "ʼ": "'",   # right/left single quote, modifier apostrophe
    "“": "", "”": "",                    # curly double quotes
    "—": ", ", "–": ", ",                # em/en dash -> comma keeps the pause
    "…": "...",                               # ellipsis -> ASCII (single clean token)
    " ": " ", "​": "",                   # nbsp, zero-width space
    # ASCII forms the LoRA's model card lists as unreliable ("struggles with
    # ( ) \" ; [ ] /"). Confirmed audible garbling on " and ; in practice.
    '"': "",                                      # quotes: drop; speech carries the emphasis
    ";": ",",                                      # semicolon -> comma, same prosodic beat
    "(": ", ", ")": ", ",
    "[": ", ", "]": ", ",
    "/": " or ",
}


def normalize_for_speech(text):
    """Fold punctuation the model can't pronounce into forms it tokenizes cleanly."""
    for src, dst in UNICODE_PUNCT.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+([,.!?])", r"\1", text)      # "word , x" -> "word, x"
    text = re.sub(r",\s*([,.!?])", r"\1", text)     # ",." or ",," -> single mark
    text = re.sub(r"^[,\s]+", "", text)             # no leading comma from a stripped paren
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text):
    """Split into sentence-grouped chunks so long input doesn't blow past one generate() call."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if current and count + len(words) > MAX_WORDS_PER_CHUNK:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def load_model(device):
    print(f"Loading {BASE_MODEL} on {device} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    try:
        model = CsmForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=torch.float16, device_map=device,
        )
    except TypeError:
        # older transformers releases use `torch_dtype` instead of `dtype`
        model = CsmForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map=device,
        )

    if LORA_MODEL:
        print(f"Applying LoRA {LORA_MODEL} ...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, LORA_MODEL)
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(f"Note: could not merge LoRA weights ({e}); running unmerged.",
                  file=sys.stderr)
    else:
        # Either base CSM, or a full finetune already loaded above as
        # BASE_MODEL. Nothing to layer on in both cases.
        print("No LoRA adapter (using the model as loaded).", file=sys.stderr)
    model.eval()

    # Both of these replace stock paths with hand-written ones on the hot
    # synthesis path, which makes them the first things to rule out when
    # chasing an intermittent HSA fault. Env toggles rather than edits so the
    # comparison can be run without touching code:
    #   COMPANION_NO_FAST_GEMV=1   keep rocBLAS's own (slow) batch-1 kernels
    #   COMPANION_NO_FAST_DEPTH=1  keep transformers' per-frame generate() wrapper
    # Expect synthesis to get materially slower with either disabled - that's
    # the point of them; this is for diagnosis, not normal running.
    parts = []
    if os.environ.get("COMPANION_NO_FAST_GEMV") == "1":
        parts.append("fast GEMV DISABLED")
    else:
        parts.append(f"Fast GEMV on {fast_gemv.apply(model)} Linear layers")
    if os.environ.get("COMPANION_NO_FAST_DEPTH") == "1":
        parts.append("fast depth-decoder DISABLED")
    else:
        fast_depth.apply(model)
        parts.append("fast depth-decoder loop enabled")
    print("; ".join(parts) + ".", file=sys.stderr)
    return processor, model


def get_sample_rate(processor, model):
    candidates = [
        (getattr(processor, "feature_extractor", None), "sampling_rate"),
        (getattr(model.config, "audio_encoder_config", None), "sampling_rate"),
        (model.config, "sampling_rate"),
    ]
    for obj, attr in candidates:
        if obj is not None and getattr(obj, attr, None):
            return int(getattr(obj, attr))
    return FALLBACK_SAMPLE_RATE


def audio_output_to_numpy(audio):
    # generate(..., output_audio=True) returns a list of tensors, one per batch item.
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        return audio.detach().to(torch.float32).cpu().reshape(-1).numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def synthesize(text, processor, model, device, do_sample, temperature, verbose=True):
    text = normalize_for_speech(text)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text to synthesize.")

    bad_chars = PROBLEM_CHARS & set(text)
    if bad_chars and verbose:
        print(
            f"Warning: text contains characters this LoRA struggles with: {''.join(sorted(bad_chars))}",
            file=sys.stderr,
        )

    sample_rate = get_sample_rate(processor, model)
    if verbose:
        print(f"Sample rate: {sample_rate} Hz, {len(chunks)} chunk(s)", file=sys.stderr)
    silence = np.zeros(int(sample_rate * SILENCE_GAP_S), dtype=np.float32)

    segments = []
    for i, chunk in enumerate(chunks, 1):
        if verbose:
            preview = chunk[:60] + ("..." if len(chunk) > 60 else "")
            print(f"  [{i}/{len(chunks)}] {preview}", file=sys.stderr)

        conversation = [{"role": str(SPEAKER_ID), "content": [{"type": "text", "text": chunk}]}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True,
        ).to(device)

        n_words = max(1, len(chunk.split()))
        max_new_tokens = int(
            min(MAX_TOKENS_PER_CHUNK, max(MIN_TOKENS_PER_CHUNK, TOKENS_PER_WORD * n_words))
        )

        gen_kwargs = {"max_new_tokens": max_new_tokens, "output_audio": True}
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            audio = model.generate(**inputs, **gen_kwargs)

        segments.append(audio_output_to_numpy(audio))
        if i < len(chunks):
            segments.append(silence)

    return np.concatenate(segments), sample_rate


def write_wav(path, samples, sample_rate):
    from scipy.io import wavfile

    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak * 0.98
    pcm16 = (samples * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, pcm16)


def main():
    parser = argparse.ArgumentParser(description="CSM-1B + voice LoRA text-to-speech")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("text", nargs="?", help="Text to speak")
    src.add_argument("-f", "--file", help="Path to a text file to read instead of TEXT")
    parser.add_argument("-o", "--output", default="output.wav", help="Output wav path (default: output.wav)")
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (with --sample)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    if not text or not text.strip():
        parser.error("No text provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: ROCm/CUDA not available, falling back to CPU (will be slow).", file=sys.stderr)
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)

    processor, model = load_model(device)

    try:
        samples, sample_rate = synthesize(text, processor, model, device, args.sample, args.temperature)
    except torch.cuda.OutOfMemoryError:
        print(
            "Out of VRAM. Close other GPU applications (e.g. LM Studio) and retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    except RuntimeError as e:
        msg = str(e)
        if "HSA" in msg or "kernel image" in msg or "invalid device function" in msg:
            print(
                "GPU kernel error - your gfx1030 may need an arch override. "
                "Uncomment the HSA_OVERRIDE_GFX_VERSION line near the top of this script and rerun.",
                file=sys.stderr,
            )
        raise

    write_wav(args.output, samples, sample_rate)
    duration = len(samples) / sample_rate
    print(f"Wrote {args.output} ({duration:.1f}s, {sample_rate} Hz)")


if __name__ == "__main__":
    main()
