from datasets import load_from_disk, Dataset, load_dataset
import numpy as np

def balanced_subsample(dataset: Dataset, cap: int, seed: int = 42) -> Dataset:
    """Subsample so each source contributes at most cap/n_sources, with leftover redistributed."""
    rng = np.random.default_rng(seed)
    sources = dataset.unique("source")
    n_sources = len(sources)
    per_source_limit = cap // n_sources

    counts = {src: dataset.filter(lambda x, s=src: x["source"] == s).num_rows for src in sources}
    taken = {src: min(counts[src], per_source_limit) for src in sources}
    used = sum(taken.values())
    leftover = cap - used
    remaining = {src: counts[src] - taken[src] for src in sources if counts[src] > taken[src]}

    while leftover > 0 and remaining:
        share = max(1, leftover // len(remaining))
        filled = []
        for src in list(remaining):
            give = min(share, remaining[src], leftover)
            taken[src] += give
            remaining[src] -= give
            leftover -= give
            if remaining[src] == 0:
                filled.append(src)
        for src in filled:
            del remaining[src]

    sampled_splits = []
    for src in sources:
        src_subset = dataset.filter(lambda x, s=src: x["source"] == s)
        n_take = taken[src]
        if src_subset.num_rows <= n_take:
            sampled_splits.append(src_subset)
        else:
            idxs = rng.choice(src_subset.num_rows, size=n_take, replace=False)
            sampled_splits.append(src_subset.select(idxs))

    out = sampled_splits[0].to_list()
    for s in sampled_splits[1:]:
        out.extend(s.to_list())
    return Dataset.from_list(out)


if __name__ == "__main__":
    train_ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    subset_ds_0_05 = train_ds.filter(
        lambda x: x["llama8b_solve_rate"] is not None and x["llama8b_solve_rate"] <= 0.5
    )
    subset_ds_0_05_40k = balanced_subsample(subset_ds_0_05, 40000, seed=42)
    subset_ds_0_05_40k.save_to_disk("./bigmathrl_subset.ds")
