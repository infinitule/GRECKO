"""Intent classification model — numpy MLP with observation-dropout training.

A deliberately small two-hidden-layer MLP over per-cluster collective
features. The plan's full graph-attention + Trajectron++-style decoder is the
documented v2 upgrade (ADR-005b); this implementation keeps the identical
interface (cluster features in, intent distribution out) so the swap is a
checkpoint change, not an architecture change for any consumer.

Training applies observation dropout: every training sample is extracted from
a randomly occluded view of the swarm (a fraction of agents hidden), so the
model learns to infer collective intent from subsets. This is the robustness
contribution the acceptance test measures.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from learn.intent.doctrines import INTENT_CLASSES, N_STEPS, SAMPLE_DT, Scenario
from learn.intent.features import N_FEATURES, cluster_agents, cluster_features, observe


class IntentMLP:
    def __init__(self, n_in: int = N_FEATURES, n_h1: int = 32, n_h2: int = 16,
                 n_out: int = len(INTENT_CLASSES), seed: int = 0):
        rng = np.random.default_rng(seed)
        s1 = math_sqrt = np.sqrt(2.0 / n_in)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_h1))
        self.b1 = np.zeros(n_h1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_h1), (n_h1, n_h2))
        self.b2 = np.zeros(n_h2)
        self.W3 = rng.normal(0, np.sqrt(2.0 / n_h2), (n_h2, n_out))
        self.b3 = np.zeros(n_out)

    # -- forward -----------------------------------------------------------

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, tuple]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3
        z3 = z3 - z3.max(axis=1, keepdims=True)
        e = np.exp(z3)
        probs = e / e.sum(axis=1, keepdims=True)
        return probs, (X, z1, a1, z2, a2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(np.atleast_2d(X))[0]

    # -- training ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 400,
            lr: float = 0.01, weight_decay: float = 1e-4,
            verbose: bool = False) -> List[float]:
        n = X.shape[0]
        Y = np.zeros((n, len(INTENT_CLASSES)))
        Y[np.arange(n), y] = 1.0
        losses = []

        # Adam state
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            probs, (Xc, z1, a1, z2, a2) = self.forward(X)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
            losses.append(float(loss))

            d3 = (probs - Y) / n
            gW3 = a2.T @ d3 + weight_decay * self.W3
            gb3 = d3.sum(axis=0)
            d2 = (d3 @ self.W3.T) * (z2 > 0)
            gW2 = a1.T @ d2 + weight_decay * self.W2
            gb2 = d2.sum(axis=0)
            d1 = (d2 @ self.W2.T) * (z1 > 0)
            gW1 = Xc.T @ d1 + weight_decay * self.W1
            gb1 = d1.sum(axis=0)

            grads = [gW1, gb1, gW2, gb2, gW3, gb3]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                mh = m[i] / (1 - beta1 ** epoch)
                vh = v[i] / (1 - beta2 ** epoch)
                p -= lr * mh / (np.sqrt(vh) + eps)

            if verbose and epoch % 100 == 0:
                acc = float((probs.argmax(axis=1) == y).mean())
                print(f"epoch {epoch}: loss={loss:.4f} acc={acc:.3f}")
        return losses

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path: str) -> "IntentMLP":
        data = np.load(path)
        model = cls()
        model.W1, model.b1 = data["W1"], data["b1"]
        model.W2, model.b2 = data["W2"], data["b2"]
        model.W3, model.b3 = data["W3"], data["b3"]
        return model


# ---------------------------------------------------------------------------
# Dataset construction with observation dropout
# ---------------------------------------------------------------------------

def build_training_set(
    scenarios: List[Scenario],
    dropout: float,
    samples_per_scenario: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (cluster_features, majority_label) samples from randomly
    occluded observations at random times across each scenario."""
    rng = np.random.default_rng(seed)
    X_rows, y_rows = [], []
    class_idx = {c: i for i, c in enumerate(INTENT_CLASSES)}

    for sc in scenarios:
        for _ in range(samples_per_scenario):
            step = int(rng.integers(4, N_STEPS))
            pos, vel, visible = observe(sc.trajectories, step, dropout, rng)
            clusters = cluster_agents(pos, vel)
            for members in clusters:
                if len(members) < 2:
                    continue
                feats = cluster_features(pos, vel, members, sc.asset_pos, len(visible))
                true_labels = [sc.labels[visible[m]] for m in members]
                majority = max(set(true_labels), key=true_labels.count)
                X_rows.append(feats)
                y_rows.append(class_idx[majority])

    return np.array(X_rows), np.array(y_rows)
