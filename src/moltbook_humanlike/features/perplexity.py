"""
Perplexity-based feature extraction using a language model.

What this does:
    Runs each post through a language model (Llama 3.2 1B by default,
    falling back to GPT-2 medium if unavailable) and measures how
    "surprised" the model is by the text.  High surprise (perplexity)
    means the text is unusual for the model — which can indicate
    more human-like, idiosyncratic writing.

What it produces:
    Per-post features: mean perplexity, perplexity variance, and
    tail perplexity (95th percentile of token-level surprisals).

Why it matters:
    Language models learn the statistical patterns of their training
    data.  Text that deviates from those patterns — as measured by
    perplexity — is a strong signal of atypicality.  Human writing
    often has higher and more variable perplexity than LLM output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger("moltbook")


def _get_device() -> torch.device:
    """Return the best available torch device: mps > cuda > cpu."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_lm(
    model_name: str,
    fallback_model: str = "gpt2-medium",
) -> tuple:
    """Load a causal language model and tokenizer.

    Tries ``model_name`` first.  If the model is gated or otherwise
    inaccessible, falls back to ``fallback_model``.

    Automatically moves the model to MPS/CUDA if available.

    Returns:
        (model, tokenizer, actual_model_name, device)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _get_device()

    for name in (model_name, fallback_model):
        try:
            logger.info("Loading language model: %s", name)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float32,
            )
            model.eval()
            model.to(device)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Successfully loaded: %s (device: %s)", name, device)
            return model, tokenizer, name, device
        except Exception as exc:
            logger.warning(
                "Could not load '%s': %s. %s",
                name,
                exc,
                "Trying fallback..." if name == model_name else "No more fallbacks.",
            )

    raise RuntimeError(
        f"Failed to load both '{model_name}' and fallback '{fallback_model}'. "
        "Check your internet connection and HuggingFace authentication."
    )


def compute_perplexity(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_length: int = 1024,
    stride: int = 512,
) -> dict[str, float]:
    """Compute token-level perplexity statistics for a single text.

    Uses a sliding-window approach to handle texts longer than the
    model's context window.

    Returns dict with ppl_mean, ppl_var, ppl_tail_95.
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length * 4,  # allow long texts, we'll window
    )
    input_ids = encodings.input_ids[0]
    seq_len = input_ids.size(0)

    if seq_len < 2:
        return {"ppl_mean": 0.0, "ppl_var": 0.0, "ppl_tail_95": 0.0}

    all_nlls: list[float] = []

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_chunk = input_ids[begin:end].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)

        # Per-token negative log-likelihoods
        logits = outputs.logits[:, :-1, :]
        targets = input_chunk[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_nlls = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        all_nlls.extend(token_nlls[0].cpu().tolist())

        if end == seq_len:
            break

    if not all_nlls:
        return {"ppl_mean": 0.0, "ppl_var": 0.0, "ppl_tail_95": 0.0}

    arr = np.array(all_nlls)
    return {
        "ppl_mean": float(np.exp(arr.mean())),
        "ppl_var": float(arr.var()),
        "ppl_tail_95": float(np.exp(np.percentile(arr, 95))),
    }


def compute_perplexity_features(
    texts: list[str],
    model_name: str = "meta-llama/Llama-3.2-1B",
    fallback_model: str = "gpt2-medium",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute perplexity features for all texts, with caching.

    If a cache file exists at ``cache_path``, loads from there instead
    of recomputing.  Otherwise computes and saves to cache.
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached perplexity features from %s", cache_path)
            return pd.read_parquet(cache_path)

    model, tokenizer, actual_name, device = load_lm(model_name, fallback_model)
    logger.info(
        "Computing perplexity for %d texts with %s on %s",
        len(texts), actual_name, device,
    )

    results: list[dict[str, float]] = []
    for text in tqdm(texts, desc="Perplexity"):
        if not text or not str(text).strip():
            results.append({"ppl_mean": 0.0, "ppl_var": 0.0, "ppl_tail_95": 0.0})
        else:
            results.append(compute_perplexity(str(text), model, tokenizer, device))

    df = pd.DataFrame(results)
    df["ppl_model_used"] = actual_name

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Saved perplexity cache to %s", cache_path)

    return df
