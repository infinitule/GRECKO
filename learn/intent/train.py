"""Train the intent model on scripted doctrines and save the checkpoint.

Usage: python -m learn.intent.train [--out learn/checkpoints/intent_mlp.npz]
"""
from __future__ import annotations

import argparse

import numpy as np

from learn.intent.doctrines import generate_dataset
from learn.intent.model import IntentMLP, build_training_set

TRAIN_DROPOUT = 0.4    # observation dropout — the robustness mechanism


def train(seed: int = 0, n_per_doctrine: int = 30,
          samples_per_scenario: int = 6, epochs: int = 400) -> tuple:
    scenarios = generate_dataset(n_per_doctrine, seed=seed)
    X, y = build_training_set(scenarios, dropout=TRAIN_DROPOUT,
                              samples_per_scenario=samples_per_scenario,
                              seed=seed + 1)
    model = IntentMLP(seed=seed)
    losses = model.fit(X, y, epochs=epochs, verbose=True)
    probs = model.predict_proba(X)
    acc = float((probs.argmax(axis=1) == y).mean())
    return model, acc, len(X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="learn/checkpoints/intent_mlp.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, acc, n = train(seed=args.seed)
    model.save(args.out)
    print(f"trained on {n} cluster samples (40% observation dropout); "
          f"train accuracy {acc:.3f}; saved {args.out}")


if __name__ == "__main__":
    main()
