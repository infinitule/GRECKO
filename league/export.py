"""Export discovered policies as Scenario objects for PB retraining.

The exported Scenarios use the identical container format as
learn/intent/doctrines.py, so the PB intent model can retrain on them
without any code changes.  Each file is an .npz containing:
  doctrine      — string tag (e.g. "league_gen5")
  trajectories  — (n_agents, N_STEPS, 4) float array
  labels        — (n_agents,) string array of intent labels
  asset_pos     — (2,) asset position
  policy_theta  — (N_POLICY_PARAMS,) raw policy vector
  fitness       — scalar fitness achieved by this policy
"""
from __future__ import annotations

import json
import pathlib
from typing import List

import numpy as np

from league.policy import SwarmPolicy
from learn.intent.doctrines import Scenario


def policy_to_scenario(policy: SwarmPolicy, seed: int = 0) -> Scenario:
    """Convert a SwarmPolicy to a Scenario object via trajectory sampling."""
    rng = np.random.default_rng(seed)
    return policy.to_scenario(rng)


def export_tactic_library(
    policies: List[SwarmPolicy],
    out_dir: str = "learn/discovered_doctrines",
    seed: int = 0,
) -> List[str]:
    """Export top policies as Scenario .npz files and a JSON manifest.

    Returns the list of saved file paths.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for i, policy in enumerate(policies):
        sc = policy_to_scenario(policy, seed=seed + i)
        name = f"league_gen{policy.generation}_p{policy.policy_id}"
        path = out / f"{name}.npz"
        np.savez(
            str(path),
            doctrine=np.array([sc.doctrine]),
            trajectories=sc.trajectories,
            labels=np.array(sc.labels),
            asset_pos=sc.asset_pos,
            policy_theta=policy.theta,
            fitness=np.array([policy.fitness]),
        )
        saved.append(str(path))

    manifest = {
        "n_doctrines": len(policies),
        "files": saved,
        "policies": [p.to_dict() for p in policies],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return saved


def load_scenario_from_npz(path: str) -> Scenario:
    """Load a league-exported Scenario .npz.  Compatible with PB pipeline."""
    d = np.load(path, allow_pickle=True)
    return Scenario(
        doctrine=str(d["doctrine"][0]),
        trajectories=d["trajectories"],
        labels=list(d["labels"]),
        asset_pos=d["asset_pos"],
    )


def validate_scenario(sc: Scenario) -> bool:
    """Check structural validity for PB compatibility."""
    from learn.intent.doctrines import INTENT_CLASSES, N_STEPS
    if sc.trajectories.ndim != 3:
        return False
    if sc.trajectories.shape[1] != N_STEPS:
        return False
    if sc.trajectories.shape[2] != 4:
        return False
    if len(sc.labels) != sc.trajectories.shape[0]:
        return False
    valid_labels = set(INTENT_CLASSES) | {"screen"}
    if not all(lab in valid_labels for lab in sc.labels):
        return False
    return True
