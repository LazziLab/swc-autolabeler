"""Retrained Emissah et al. GCN baseline on this corpus's splits.

Faithful port of the dendrite-classification model released with the
NeuroMorpho standardization pipeline (Emissah, Tecuatl and Ascoli 2026;
github.com/HerveEmissah/nmo_swc_standardization, ML/mpi_swc_processing.py at
commit c3491e15486c1fbc325e9fdd4506bfd0402c0f85), retrained on our seed
hash-split so it can be scored on the same held-out cells as every other
baseline. The released weights cannot be used: they were
fitted on 20,500 NeuroMorpho pyramidal cells under a random split that
overlaps any test set drawn from the repository. The released code is
MIT-licensed, Copyright (c) 2024 HerveEmissah; the model classes, the file
adjacency and the prediction rules below follow it, and its notice is
reproduced in THIRD_PARTY_NOTICES.md.

The method, as released: each primary tree (child of the soma) is one graph
node; a file's trees form a fully connected graph; each tree carries a single
scalar feature, the "cumulative Sholl value"; a two-layer GCN (hidden 1024,
dropout 0.5) emits apical/basal/other per tree. At prediction time two
constraints apply, both taken from the released code: trees whose ground
truth is "other" (axon) keep that label, and exactly one non-other tree is
apical (the largest apical-basal margin), all others basal. The comparison is
therefore generous to the baseline: it receives the ground-truth axon/soma
assignment and the one-apical prior, and is only asked to choose which
dendritic tree is the apical one and score at tree granularity.

Feature note: the released cumulative Sholl value sums, over every target
node (their branch-point test on a directed graph selects every internal
node, and leaves are the rest), the number of nodes of the soma-to-target
path that fall in shells of width 0.1 x the sum of those nodes' distances
from the soma. In exact arithmetic each node falls in exactly one shell, so
the per-path value is the path's node count and the cumulative value is the
sum of node depths (in nodes) over the tree. We compute that closed form.
The released shells come from np.arange, and floating-point rounding at a
shell boundary can count a node twice or not at all. A tree's first node
lies on a boundary by construction, its distance being ten shell widths, so
the released value can differ from the closed form by one.

Training matches the release: Adam (lr 1e-6, weight decay 1e-2), OneCycleLR
(max_lr 1e-3, pct_start 0.1) stepped per epoch, 50 epochs, batch of one file,
unweighted cross-entropy, best epoch by validation file-exact accuracy (10%
of the training files held out for validation).

The scheduler is built for one step per file and stepped once per epoch, so it
advances 50 of the 300,000-odd steps it is laid out over and the learning rate
stays at max_lr/25 for the whole run instead of rising to max_lr. That is the
released behavior, not an error in this port: ML/mpi_swc_processing.py of the
commit above builds it the same way at line 616 and steps it at line 668, under
a comment keeping it deliberately. Stepping per file here would train a model
their published weights do not correspond to, so it is left alone.

    python -m evaluation.baselines.emissah_gcn --seeds 123

Output: evaluation/results/baselines/emissah_gcn_seed{S}.{json,txt} with the
paper's node-level metrics (tree labels expanded to their nodes; soma, axon
trees and sub-2-node trees keep ground truth, as the method requires).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                              # noqa: E402
from evaluation.baselines.baseline_report import build_full_report   # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1
from pipeline.split import in_test  # noqa: E402

DATA = ROOT / "data" / "pyramidal"
RESULTS = ROOT / "evaluation" / "results" / "baselines"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_EPOCHS = 50


# ---- released architecture -------------------------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1024, output_dim=3):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = self.dropout(torch.relu(self.bn1(self.gc1(x, adj))))
        x = torch.relu(self.bn2(self.gc2(x, adj)))
        return self.fc3(x)


def file_adjacency(n):
    adj = torch.ones((n, n)) + torch.eye(n)
    d = adj.sum(1).clamp(min=1.0).pow(-0.5)
    return d[:, None] * adj * d[None, :]


# ---- released tree extraction + feature ------------------------------------
def extract_trees(nodes):
    """Primary trees exactly as the released classify_trees: one per child of a
    soma node, label apical if any node-4 else basal if any node-3 else other,
    trees of fewer than two nodes skipped; feature = sum of node depths."""
    by_id = {n.id: n for n in nodes}
    children: dict[int, list[int]] = {}
    for n in nodes:
        children.setdefault(n.parent, []).append(n.id)
    soma_ids = [n.id for n in nodes if n.type == 1]
    trees = []
    for sid in soma_ids:
        for root in children.get(sid, []):
            if by_id[root].type == 1:
                continue
            tree_ids, depth, stack = [], {root: 1}, [root]
            while stack:
                cur = stack.pop()
                if by_id[cur].type == 1:
                    continue
                tree_ids.append(cur)
                for ch in children.get(cur, []):
                    if ch not in depth:
                        depth[ch] = depth[cur] + 1
                        stack.append(ch)
            if len(tree_ids) <= 1:
                continue
            types = {by_id[i].type for i in tree_ids}
            label = 0 if 4 in types else 1 if 3 in types else 2   # apical/basal/other
            feature = float(sum(depth[i] for i in tree_ids))
            trees.append({"ids": tree_ids, "label": label, "feature": feature})
    return trees


def constrain(outputs, labels):
    """The released prediction rules: GT-other frozen; exactly one apical."""
    predicted = outputs.argmax(1)
    predicted[labels == 2] = 2
    non_other = (labels != 2).nonzero(as_tuple=True)[0]
    if non_other.numel() > 0:
        margins = outputs[non_other, 0] - outputs[non_other, 1]
        predicted[non_other] = 1
        predicted[non_other[margins.argmax()]] = 0
    return predicted


def run_seed(args, seed: int) -> int:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # The epoch order comes from its own generator rather than numpy's global
    # state, so anything else that draws from numpy cannot shift it.
    order_rng = np.random.default_rng(seed)

    files = sorted(DATA.glob("*.swc"))
    train_files = [p for p in files if not in_test(p.name, seed, args.test_size)]
    test_files = [p for p in files if in_test(p.name, seed, args.test_size)]
    rng = np.random.RandomState(seed)
    rng.shuffle(train_files)
    n_val = max(1, int(len(train_files) * 0.10))
    val_files, train_files = train_files[:n_val], train_files[n_val:]
    if args.limit:
        train_files, val_files, test_files = (train_files[:args.limit],
                                              val_files[:max(2, args.limit // 8)],
                                              test_files[:args.limit])
    print(f"seed {seed}: train {len(train_files)}  val {len(val_files)}  "
          f"test {len(test_files)}", flush=True)

    def build(paths, keep_nodes=False):
        out, t0 = [], time.perf_counter()
        for i, p in enumerate(paths, 1):
            nodes = parse_swc(p)
            if not nodes:
                continue
            trees = extract_trees(nodes)
            if not trees:
                continue
            rec = {"file": p.name, "trees": trees}
            if keep_nodes:
                rec["nodes"] = nodes
            out.append(rec)
            if i % 1000 == 0:
                print(f"  features {i}/{len(paths)} ({(time.perf_counter()-t0)/60:.1f} min)",
                      flush=True)
        return out

    train_recs = build(train_files)
    val_recs = build(val_files)
    test_recs = build(test_files, keep_nodes=True)

    def tensors(rec):
        feats = torch.tensor([[t["feature"]] for t in rec["trees"]])
        labels = torch.tensor([t["label"] for t in rec["trees"]])
        return feats, labels, file_adjacency(len(rec["trees"]))

    model = GCN()
    opt = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
    sched = OneCycleLR(opt, max_lr=1e-3, epochs=args.epochs,
                       steps_per_epoch=max(1, len(train_recs)), pct_start=0.1)
    crit = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        order = order_rng.permutation(len(train_recs))
        for idx in order:
            feats, labels, adj = tensors(train_recs[idx])
            if len(labels) < 2:      # BatchNorm needs >1 tree in train mode
                continue
            opt.zero_grad()
            loss = crit(model(feats, adj), labels)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            for r in val_recs:
                feats, labels, adj = tensors(r)
                if torch.equal(constrain(model(feats, adj), labels), labels):
                    correct += 1
        acc = correct / max(1, len(val_recs))
        # Strictly greater, so a tie keeps the earlier epoch, and a run stopped at
        # any later epoch keeps the same model. File-exact accuracy ties often.
        if acc > best_val:
            best_val, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: val file-exact {acc:.3f} (best {best_val:.3f})",
                  flush=True)
    model.load_state_dict(best_state)

    # ---- node-level scoring on the held-out cells --------------------------
    model.eval()
    gtp, prp, records = [], [], []
    with torch.no_grad():
        for r in test_recs:
            feats, labels, adj = tensors(r)
            predicted = constrain(model(feats, adj), labels)
            nodes = r["nodes"]
            gt = [n.type for n in nodes]
            pred = list(gt)                      # soma, axon trees, skipped trees: GT
            idx = {n.id: i for i, n in enumerate(nodes)}
            TREE2SWC = {0: 4, 1: 3}
            for tree, plab in zip(r["trees"], predicted.tolist()):
                if plab in TREE2SWC:
                    for nid in tree["ids"]:
                        pred[idx[nid]] = TREE2SWC[plab]
            gtp.extend(gt)
            prp.extend(pred)
            records.append({"file": r["file"], "cell_type": "pyramidal",
                            "n_nodes": len(gt), "gt": gt, "pred": pred})

    rep = build_full_report("emissah_gcn", gtp, prp, records)
    c = rep["corpus"]
    pc = [per_cell_neurite_f1(r["gt"], r["pred"], "pyramidal") for r in records]
    pc.sort()
    p10 = float(np.percentile(pc, 10))
    out = {"seed": seed, "n_test_cells": len(records),
           "test_size": args.test_size,
           "val_file_exact": best_val, "best_epoch": best_epoch,
           "epochs": args.epochs,
           "protocol": "GT cell type, GT axon/soma supplied, one-apical prior "
                       "enforced (both from the released prediction rules)",
           "corpus": {"accuracy": c["accuracy"],
                      "neurite_macro_f1": c["neurite_macro_f1"],
                      "per_class": {k: v["f1"] for k, v in c["per_class"].items()},
                      "pcf1_mean": float(np.mean(pc)), "pcf1_p10": p10}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A reduced run is not the published one, so it never takes its file name.
    tag = f"_limit{args.limit}" if args.limit else ""
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    if args.epochs != DEFAULT_EPOCHS:
        tag += f"_ep{args.epochs}"
    (args.out_dir / f"emissah_gcn_seed{seed}{tag}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"Emissah GCN retrained on seed {seed} split "
             f"({len(records)} held-out pyramidal cells)",
             f"  {out['protocol']}",
             f"  val file-exact acc {best_val:.3f}",
             f"  node acc {c['accuracy']:.4f}  macroF1 {c['neurite_macro_f1']:.4f}  "
             f"pcF1 mean {np.mean(pc):.4f}  P10 {p10:.4f}",
             "  per-class F1: " + "  ".join(f"{k} {v['f1']:.4f}"
                                            for k, v in c["per_class"].items())]
    (args.out_dir / f"emissah_gcn_seed{seed}{tag}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="123",
                    help="comma-separated; each is trained and scored in turn")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction; must match the run this is compared against")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="cells per split (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        rc = run_seed(args, seed)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
