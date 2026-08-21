from arc3lab.types import Component


def test_center_cell_is_occupied_for_hollow_shape():
    ring = Component(
        color=9,
        pixels=8,
        bbox=(0, 0, 2, 2),
        centroid=(1.0, 1.0),
        edge_touch=False,
        shape_hash="ring",
        cells=((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
    )
    assert ring.center_cell in ring.cells
    assert ring.center_cell != (1, 1)
