"""Minimal checkpoint save/load: params (as numpy) + metadata, via pickle.

Kept dependency-light and portable. The AEI inference bridge loads these.
"""

import os
import pickle

import jax
import numpy as np


def save(path, params, meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # dir may not exist yet
    obj = {
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), params),
        "meta": meta,
    }
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["params"], obj["meta"]
