"""Skip transformers' generate() wrapper on CSM's per-frame depth decoder.

CSM's _sample loop calls depth_decoder.generate() once per audio frame
(12.5x per second of audio). Each call rebuilds a GenerationConfig, a
LogitsProcessorList and a StoppingCriteriaList before doing any work, even
though every frame is identical in shape: exactly num_codebooks-1 tokens with
the same sampling parameters. That fixed setup costs ~6 ms per frame, about
6.5% of total synthesis time.

This replaces the per-frame generate() with a direct forward loop that
reproduces the same sampling. Verified against the stock path:
  - greedy: token sequences match exactly (6/6 trials)
  - sampling with an identical seed: match exactly, 0/33 differing tokens
    (6/6 trials), confirming temperature-then-top_k warper order and params

Note the depth decoder samples (do_sample=True, temperature 0.9, top_k 50)
regardless of the do_sample passed to the top-level generate(), which only
controls the backbone.
"""
import torch
from transformers.cache_utils import DynamicCache


def _fast_decode(dd, input_ids, backbone_last_hidden_state, num_new, do_sample, temperature, top_k):
    cache = DynamicCache(config=dd.config)
    cur = input_ids
    hidden = backbone_last_hidden_state
    generated = []
    for _ in range(num_new):
        out = dd(
            input_ids=cur,
            backbone_last_hidden_state=hidden,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = out.logits[:, -1, :].float()
        if do_sample:
            if temperature and temperature != 1.0:
                logits = logits / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(nxt)
        cur = nxt
        hidden = None  # only the first step seeds position 0 with the backbone state
    return torch.cat([input_ids] + generated, dim=1)


def apply(model):
    """Swap depth_decoder.generate for the direct loop. Returns True if installed."""
    dd = model.depth_decoder
    if getattr(dd, "_fast_depth_installed", False):
        return False
    num_new = model.config.num_codebooks - 1

    def fast_generate(input_ids=None, backbone_last_hidden_state=None, **kwargs):
        gen_cfg = dd.generation_config
        return _fast_decode(
            dd, input_ids, backbone_last_hidden_state, num_new,
            gen_cfg.do_sample, gen_cfg.temperature, gen_cfg.top_k,
        )

    dd.generate = fast_generate
    dd._fast_depth_installed = True
    return True
