"""Tests for pipeline.swc_normalize.

Run from the repository root:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import SWCNode  # noqa: E402
from pipeline.swc_normalize import normalize_custom_types, normalize_swc  # noqa: E402


def _chain(start_id: int, parent: int, n: int, node_type: int, x0: float, dx: float = 2.0,
           y: float = 0.0, radius: float = 0.5) -> list[SWCNode]:
    out = []
    for k in range(n):
        nid = start_id + k
        out.append(SWCNode(nid, node_type, x0 + dx * k, y, 0.0, radius, parent if k == 0 else nid - 1))
    return out


def _cell_with_untyped_axon_start() -> list[SWCNode]:
    """Soma; a basal dendrite; an axon whose first 8 nodes are type 0.

    The type-0 run ends at a branch point: one child continues as axon
    (type 2), the other as more type-0 nodes.
    """
    nodes = [SWCNode(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
    nodes += _chain(2, 1, 10, 3, -2.0, dx=-2.0)            # basal, ids 2..11
    nodes += _chain(12, 1, 8, 0, 2.0)                       # untyped run, ids 12..19
    nodes += _chain(20, 19, 10, 2, 18.0, y=1.0)             # axon, ids 20..29
    nodes += _chain(30, 19, 6, 0, 18.0, y=-1.0)             # untyped side branch, ids 30..35
    return nodes


class SomaAncestorTests(unittest.TestCase):
    """A custom/undefined stretch whose nearest standard ancestor is the soma."""

    def test_stretch_at_the_soma_stays_undefined(self) -> None:
        out, _ = normalize_custom_types(_cell_with_untyped_axon_start())
        types = {nd.id: nd.type for nd in out}
        self.assertTrue(all(types[i] == 0 for i in range(12, 20)), types)
        self.assertTrue(all(types[i] == 0 for i in range(30, 36)), types)
        self.assertEqual(sum(1 for t in types.values() if t == 1), 1)

    def test_soma_is_not_merged_moved_or_inflated(self) -> None:
        out, info = normalize_swc(_cell_with_untyped_axon_start())
        soma = [nd for nd in out if nd.type == 1]
        self.assertEqual(len(soma), 1)
        self.assertEqual((soma[0].x, soma[0].y, soma[0].z, soma[0].radius), (0.0, 0.0, 0.0, 3.0))
        self.assertFalse(info["soma"]["changed"])
        self.assertEqual(len(out), len(_cell_with_untyped_axon_start()))

    def test_typed_children_keep_their_parents(self) -> None:
        out, _ = normalize_swc(_cell_with_untyped_axon_start())
        parents = {nd.id: nd.parent for nd in out}
        self.assertEqual(parents[20], 19)
        self.assertEqual(parents[12], 1)


class UnchangedRuleTests(unittest.TestCase):
    """The other rules behave as in 1.0.0."""

    def test_untyped_nodes_inside_a_typed_branch_take_its_type(self) -> None:
        nodes = [SWCNode(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
        nodes += _chain(2, 1, 4, 2, 2.0) + _chain(6, 5, 3, 0, 10.0) + _chain(9, 8, 4, 2, 16.0)
        out, _ = normalize_custom_types(nodes)
        self.assertTrue(all(nd.type == 2 for nd in out if nd.id >= 2))

    def test_untyped_branch_below_a_neurite_takes_the_neurite_type(self) -> None:
        nodes = [SWCNode(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
        nodes += _chain(2, 1, 4, 3, 2.0)                    # dendrite to a branch point at 5
        nodes += _chain(6, 5, 3, 3, 10.0, y=1.0)            # typed child
        nodes += _chain(9, 5, 3, 0, 10.0, y=-1.0)           # untyped child segment
        out, _ = normalize_custom_types(nodes)
        self.assertTrue(all(nd.type == 3 for nd in out if nd.id in (9, 10, 11)))

    def test_custom_codes_still_follow_their_branch(self) -> None:
        nodes = [SWCNode(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
        nodes += _chain(2, 1, 3, 4, 0.0, dx=0.0) + _chain(5, 4, 2, 10, 0.0, dx=0.0)
        out, _ = normalize_custom_types(nodes)
        self.assertTrue(all(nd.type == 4 for nd in out if nd.id >= 2))


@unittest.skipUnless((ROOT / "pipeline" / "models" / "seed123" / "branch_classifier.pkl").is_file(),
                     "released models not present")
class LabelingTests(unittest.TestCase):
    """With the released models, the stretch is labeled with its branch."""

    def test_untyped_stretch_at_the_soma_is_labeled_as_a_neurite(self) -> None:
        from pipeline.infer_pipeline import run_pipeline

        nodes = _cell_with_untyped_axon_start()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.swc"
            path.write_text("".join(
                f"{nd.id} {nd.type} {nd.x} {nd.y} {nd.z} {nd.radius} {nd.parent}\n" for nd in nodes),
                encoding="utf-8")
            result = run_pipeline(path)
        labels = list(result.node_labels)
        self.assertEqual(len(labels), len(nodes))
        self.assertEqual(labels.count(1), 1)
        self.assertTrue(all(label in (2, 3, 4) for label in labels[1:]), labels)


if __name__ == "__main__":
    unittest.main()
