"""
Embedding-based outlier features using sentence-transformers.

What this does:
    Encodes each post into a dense vector using a lightweight
    sentence-transformer model, then measures how far each post
    is from its neighbors and from the overall centroid.

What it produces:
    Per-post features: mean nearest-neighbor distance, local density,
    and distance to the global centroid.

Why it matters:
    Posts that live in sparse regions of the embedding space are
    semantically unusual.  Combining embedding distance with other
    features gives a more robust picture of atypicality.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

logger = logging.getLogger("moltbook")


def compute_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: str | Path | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with a sentence-transformer; cache to .npy."""
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached embeddings from %s", cache_path)
            return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    logger.info("Encoding %d texts with %s", len(texts), model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Saved embedding cache to %s", cache_path)

    return embeddings


def compute_embedding_features(
    embeddings: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """Derive outlier-relevant features from an embedding matrix.

    Features:
        emb_mean_nn_dist  — mean cosine distance to k nearest neighbors
        emb_local_density — 1 / mean distance (higher = denser neighborhood)
        emb_centroid_dist — cosine distance to the global centroid
    """
    logger.info("Computing embedding features (k=%d neighbors)", k)

    # Use cosine metric via NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings)), metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # Exclude self (first column is distance to self ≈ 0)
    neighbor_dists = distances[:, 1:]

    mean_nn_dist = neighbor_dists.mean(axis=1)
    local_density = 1.0 / (mean_nn_dist + 1e-10)

    # Centroid distance
    centroid = embeddings.mean(axis=0, keepdims=True)
    # Cosine distance = 1 - cosine_similarity
    norms_emb = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    norms_cen = np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-10
    cosine_sim = (embeddings @ centroid.T) / (norms_emb * norms_cen)
    centroid_dist = 1.0 - cosine_sim.flatten()

    return pd.DataFrame({
        "emb_mean_nn_dist": mean_nn_dist,
        "emb_local_density": local_density,
        "emb_centroid_dist": centroid_dist,
    })
