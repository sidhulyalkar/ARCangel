import numpy as np

from arc3lab.perception.scene import components
from arc3lab.perception.diffs import infer_hud_mask


def test_components_and_shape_hash_translation_invariant():
    a = np.zeros((8, 8), dtype=np.int8); a[1:3, 1:3] = 9
    b = np.zeros((8, 8), dtype=np.int8); b[4:6, 5:7] = 9
    ca = components(a, 0)[0]; cb = components(b, 0)[0]
    assert ca.shape_hash == cb.shape_hash
    assert ca.centroid != cb.centroid


def test_hud_mask_only_edges():
    grids = []
    for i in range(6):
        g = np.zeros((12, 12), dtype=np.int8)
        g[0, i : i + 2] = 8
        g[6, 6] = i % 2
        grids.append(g)
    mask = infer_hud_mask(grids, edge_width=2)
    assert mask is not None
    assert not mask[6, 6]
