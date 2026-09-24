"""Stage 2: derive the intent taxonomy from the data. NOT on the reproduction path.

Contract: docs/12 §2. Recipe: docs/03 §1. Requires an API key; `make labels` only.

The honest lesson from prior work (eve-bot, and an independent replication):
unsupervised clustering alone DOES NOT produce a usable taxonomy. Silhouette 0.036
at k=16 on real tweets. The LLM naming + human merge/split rounds are the step that
makes it usable -- the clustering is a lens, not the answer.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np

# UMAP: n_neighbors=30 rather than the default 15 for noisy short text.
# ALWAYS set random_state or runs are not reproducible.
UMAP_PARAMS = {"n_neighbors": 30, "n_components": 5, "metric": "cosine", "random_state": 42}
K_CANDIDATES = (15, 25, 40)


def project(embeddings):
    """UMAP with the fixed, seeded params above. Never call UMAP without random_state."""
    import umap

    return umap.UMAP(**UMAP_PARAMS).fit_transform(np.asarray(embeddings))


def select_k(embeddings, candidates: Sequence[int] = K_CANDIDATES, seed: int = 42) -> int:
    """Choose cluster count by STABILITY ACROSS SEEDS, not silhouette or elbow.

    Both mislead on UMAP-projected embeddings: silhouette rewards compact spherical
    clusters, which UMAP manufactures. Method: cluster twice on 80% bootstraps,
    measure ARI between partitions (docs/03 §1).

    A silhouette of 0.036 at k=16 on real tweets (the eve-bot replication) is not
    evidence that k=16 is wrong -- it is evidence that silhouette is the wrong
    instrument on this geometry.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    X = np.asarray(embeddings)
    rng = np.random.default_rng(seed)
    n = len(X)
    scores = {}

    for k in candidates:
        aris = []
        for rep in range(5):
            idx_a = rng.choice(n, size=int(0.8 * n), replace=False)
            idx_b = rng.choice(n, size=int(0.8 * n), replace=False)
            shared = np.intersect1d(idx_a, idx_b)
            if len(shared) < k:
                continue
            labels_a = KMeans(n_clusters=k, random_state=seed + rep, n_init=10).fit_predict(X[idx_a])
            labels_b = KMeans(n_clusters=k, random_state=seed + 100 + rep, n_init=10).fit_predict(X[idx_b])
            pos_a = {v: i for i, v in enumerate(idx_a)}
            pos_b = {v: i for i, v in enumerate(idx_b)}
            aris.append(adjusted_rand_score(
                [labels_a[pos_a[s]] for s in shared],
                [labels_b[pos_b[s]] for s in shared],
            ))
        scores[k] = float(np.mean(aris)) if aris else 0.0

    best = max(scores, key=scores.get)
    print(f"stability (mean ARI across 5 bootstrap pairs): {scores} -> k={best}")
    return best


def name_clusters(clusters, texts: Sequence[str], config=None) -> list[dict]:
    """Per cluster: 15 centroid-nearest + 5 random members -> LLM names it.

    Returns name, one-sentence definition, and 3 boundary cases per cluster.
    TnT-LLM (arXiv:2403.12173) phase 1 + GoalEx propose, in one call.
    Goal conditioning matters: cluster by WHAT THE CUSTOMER WANTS THE COMPANY TO DO,
    not by product.

    The 5 random members alongside the 15 nearest are deliberate: naming a cluster
    from its centroid alone produces a label that describes the cluster's core and
    silently excludes its edge, and the edge is where the merge/split decisions are.
    """
    from src import load_config
    from src.cache import cache_key, get as cache_get, put as cache_put
    from src.llm_client import complete

    config = config or load_config()
    model_cfg = config.models["distiller"]
    labels = np.asarray(clusters)
    rng = np.random.default_rng(42)
    out = []

    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            continue  # handled by inspect_outlier_bin
        members = np.flatnonzero(labels == cluster_id)
        sample = [texts[i] for i in members[:15]]
        if len(members) > 15:
            sample += [texts[i] for i in rng.choice(members[15:], size=min(5, len(members) - 15), replace=False)]

        prompt = (
            "These customer-support messages were grouped together.\n\n"
            "Name the group by WHAT THE CUSTOMER WANTS THE COMPANY TO DO -- not by "
            "product, feature or sentiment. A group named after a product is not an "
            "intent and cannot drive routing.\n\n"
            + "\n".join(f"- {t}" for t in sample)
            + '\n\nReturn JSON: {"name": snake_case, "definition": one sentence, '
              '"boundary_cases": [3 messages that ALMOST belong but do not, and why]}'
        )
        key = cache_key(model_cfg["id"], "induce-name-v1", prompt)
        cached = cache_get(key, config)
        raw = cached["response"] if cached else None
        if raw is None:
            raw, usage = complete(model_cfg, prompt, json_mode=True)
            cache_put(key, raw, usage, config)
        try:
            out.append({"cluster_id": int(cluster_id), "size": int(len(members)),
                        **json.loads(raw.strip().strip("`").lstrip("json"))})
        except json.JSONDecodeError:
            out.append({"cluster_id": int(cluster_id), "size": int(len(members)),
                        "name": f"cluster_{cluster_id}", "definition": raw[:200]})
    return out


def refine(candidate_labels: Sequence[dict], target_range=(8, 12), rounds: int = 2, config=None):
    """Feed all candidate labels back with an explicit merge/split/drop instruction.

    Twice. This is TnT-LLM's iterative-refinement step, explicitly analogised to
    SGD over minibatches. Granularity target 8-12: Banking77's 77 intents are the
    cautionary tale (>=14% of its training utterances may be mislabelled).

    Fine granularity with semantic overlap MANUFACTURES label noise -- the classes
    stop being separable by any annotator, including the one who wrote them.
    """
    from src import load_config
    from src.cache import cache_key, get as cache_get, put as cache_put
    from src.llm_client import complete

    config = config or load_config()
    model_cfg = config.models["distiller"]
    current = list(candidate_labels)

    for round_no in range(rounds):
        prompt = (
            f"Here are {len(current)} candidate support intents with definitions and "
            f"sizes.\n\n"
            + json.dumps(current, indent=2)
            + f"\n\nMERGE overlapping classes, SPLIT classes covering two distinct "
              f"customer asks, DROP classes that are not support requests.\n"
              f"Target {target_range[0]}-{target_range[1]} in-scope intents.\n"
              f"For every merge or split, state which and why.\n"
              f"Return JSON: {{\"intents\": [{{\"name\":..., \"definition\":..., "
              f"\"not_this\": [...]}}], \"changes\": [...]}}"
        )
        key = cache_key(model_cfg["id"], f"induce-refine-v1-r{round_no}", prompt)
        cached = cache_get(key, config)
        raw = cached["response"] if cached else None
        if raw is None:
            raw, usage = complete(model_cfg, prompt, json_mode=True)
            cache_put(key, raw, usage, config)
        try:
            payload = json.loads(raw.strip().strip("`").lstrip("json"))
            current = payload["intents"]
            print(f"round {round_no + 1}: {len(current)} intents; changes: {payload.get('changes')}")
        except (json.JSONDecodeError, KeyError):
            print(f"round {round_no + 1}: refinement output unparseable, keeping previous set")
            break
    return current


def inspect_outlier_bin(hdbscan_labels, texts: Sequence[str], sample: int = 30) -> dict:
    """Read the HDBSCAN noise cluster before discarding it.

    On multi-domain short text the outlier bin has held >74% of the dataset. For
    INDUCTION that is a feature -- it is a preview of your OTHER class.

    Discarding it unread is the mistake: a 74% noise bin does not mean the
    clustering failed, it means most inbound messages do not fall into tight
    semantic groups, which is itself the finding that justifies having an
    other_unclear class at all.
    """
    labels = np.asarray(hdbscan_labels)
    noise = np.flatnonzero(labels == -1)
    rng = np.random.default_rng(42)
    drawn = rng.choice(noise, size=min(sample, len(noise)), replace=False) if len(noise) else []
    return {
        "n_noise": int(len(noise)),
        "noise_fraction": float(len(noise) / len(labels)) if len(labels) else 0.0,
        "sample": [texts[i] for i in drawn],
        "interpretation": (
            "A large noise bin is a PREVIEW OF other_unclear, not a clustering "
            "failure. Read it before discarding it."
        ),
    }


def main() -> int:
    print(
        "Taxonomy induction requires an API key and is NOT on the reproduction path.\n"
        "Order: embed -> project() -> select_k() -> cluster -> inspect_outlier_bin()\n"
        "       -> name_clusters() -> refine(rounds=2) -> hand-edit taxonomy/intents.yaml\n"
        "       -> fill `positive` examples -> set status: frozen -> ONLY THEN label.\n"
        "Editing the taxonomy after golden_200.csv is labelled invalidates the golden set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
