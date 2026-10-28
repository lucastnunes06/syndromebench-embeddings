"""Writes a small synthetic pickle with the same nested structure as
mini_gm_public_v0.1.p:

    {syndrome_id: {subject_id: {image_id: [320 floats]}}}

Purpose: let the whole pipeline be exercised end-to-end without the real
dataset (which is not redistributable). Each syndrome gets its own random
centre in 320-d space plus noise, so the task is learnable but not trivial;
subjects contribute 1-3 correlated images each, reproducing the subject-level
correlation that makes grouped cross-validation necessary in the first place.

Usage:
    python make_synthetic_dataset.py --output /tmp/synthetic.p
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

DIM = 320


def build(n_syndromes=6, subjects_per_syndrome=40, seed=0,
          centre_scale=0.18, subject_scale=0.55, image_scale=0.40):
    rng = np.random.default_rng(seed)
    data = {}
    for s in range(n_syndromes):
        syndrome_id = f"9000000{s:02d}"
        centre = rng.normal(0, centre_scale, DIM)
        subjects = {}
        # Vary the subject count so the classes are imbalanced, as in the real data.
        n_subj = subjects_per_syndrome + int(rng.integers(-12, 13))
        for j in range(max(n_subj, 15)):
            subject_id = f"{syndrome_id}_subj{j:04d}"
            subject_vec = centre + rng.normal(0, subject_scale, DIM)
            n_images = int(rng.integers(1, 4))
            images = {}
            for i in range(n_images):
                vec = subject_vec + rng.normal(0, image_scale, DIM)
                images[f"{subject_id}_img{i}"] = vec.astype(np.float64).tolist()
            subjects[subject_id] = images
        data[syndrome_id] = subjects
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Generate a synthetic embedding pickle.")
    p.add_argument("--output", type=str, default="synthetic_dataset.p")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_syndromes", type=int, default=6)
    p.add_argument("--subjects_per_syndrome", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    data = build(n_syndromes=args.n_syndromes,
                 subjects_per_syndrome=args.subjects_per_syndrome,
                 seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f)

    n_subj = sum(len(v) for v in data.values())
    n_img = sum(len(im) for v in data.values() for im in v.values())
    print(f"Wrote {out}: {len(data)} syndromes, {n_subj} subjects, {n_img} images")


if __name__ == "__main__":
    main()
