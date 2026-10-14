"""Compile the generalization results into the paper's tables.

Reads the JSON written by ``evaluation.generalization`` and formats five
tables:

  A      External validation: frozen model on ten unseen pyramidal
         laboratories, per archive and per species, deployment and
         given-cell-type configurations.
  A-INT  The same for the five unseen interneuron laboratories.
  A2     Frozen models on the never-trained cell classes, deployment and
         given-schema configurations, on each class corpus's held-out third.
  B      Flag transfer: triage on the external pyramidal set, scope escape,
         and the Stage 1 confidence gate.
  C      Per-corpus retraining: fresh models trained on one corpus's own cells
         (three worst-case archives and three never-trained cell classes)
         against the frozen model on the same held-out cells, and two of them
         again on every usable cell of their archive.

    python -m report.generalization_tables
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results" / "generalization"
REPORT = Path(__file__).resolve().parent
OUT_TXT = REPORT / "generalization_tables.txt"
OUT_JSON = REPORT / "generalization_tables.json"

SEED = 123


def _load(name: str):
    path = RESULTS / name
    if not path.is_file():
        raise SystemExit(f"MISSING {path}; run the evaluation.generalization drivers first")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, width=9) -> str:
    if v is None:
        return f"{'--':>{width}s}"
    return f"{v:{width}.3f}" if isinstance(v, float) else f"{v:>{width}}"


def main() -> int:
    deploy = {r["group"]: r for r in _load(f"external_eval_seed{SEED}.json")}
    gtct = {r["group"]: r for r in _load(f"external_eval_seed{SEED}_gtct.json")}
    idep_p = RESULTS / f"external_eval_seed{SEED}_interneuron.json"
    igt_p = RESULTS / f"external_eval_seed{SEED}_interneuron_gtct.json"
    ideploy = ({r["group"]: r for r in _load(idep_p.name)} if idep_p.is_file() else {})
    igtct = ({r["group"]: r for r in _load(igt_p.name)} if igt_p.is_file() else {})
    flag = _load("flag_transfer.json")
    retrain = _load("lab_retrain.json")
    newclass = _load("new_class.json")
    more = _load("more_cells.json")

    L: list[str] = []

    L += ["TABLE A -- FROZEN MODEL ON TEN UNSEEN LABORATORIES "
          f"(seed {SEED}, {deploy['ALL']['n_cells']} cells)",
          "", f"{'group':30s} {'n':>5s} {'s1acc':>7s} "
          f"{'acc':>7s} {'apical':>7s} {'pcF1':>7s} {'P10':>7s}   {'acc':>7s} "
          f"{'apical':>7s} {'pcF1':>7s} {'P10':>7s}",
          f"{'':30s} {'':>5s} {'':>7s} {'---- deployment ----':>31s}   "
          f"{'---- GT cell type ----':>31s}"]
    order = [g for g in deploy if g == "ALL" or not g.startswith(("sp:", "axon-"))]
    order += [g for g in deploy if g.startswith("sp:")] + ["axon-GT-only"]
    for g in order:
        d, t = deploy[g], gtct.get(g, {})
        L.append(f"{g:30s} {d['n_cells']:5d} {d['stage1_acc']:7.3f} "
                 f"{d['node_acc']:7.3f} {_fmt(d['apical_f1'], 7)} {d['pcf1_mean']:7.3f} "
                 f"{d['pcf1_p10']:7.3f}   {_fmt(t.get('node_acc'), 7)} "
                 f"{_fmt(t.get('apical_f1'), 7)} {_fmt(t.get('pcf1_mean'), 7)} "
                 f"{_fmt(t.get('pcf1_p10'), 7)}")
    L.append("")

    if ideploy:
        L += ["TABLE A-INT -- FROZEN MODEL ON UNSEEN INTERNEURON LABORATORIES", "",
              f"{'group':30s} {'n':>5s} {'s1acc':>7s} {'acc':>7s} {'pcF1':>7s} "
              f"{'P10':>7s}   {'acc':>7s} {'pcF1':>7s} {'P10':>7s}",
              f"{'':30s} {'':>5s} {'':>7s} {'-- deployment --':>23s}   "
              f"{'-- GT cell type --':>23s}"]
        iorder = [g for g in ideploy if g == "ALL" or not g.startswith(("sp:", "axon-"))]
        iorder += [g for g in ideploy if g.startswith("sp:")]
        for g in iorder:
            d, x = ideploy[g], igtct.get(g, {})
            L.append(f"{g:30s} {d['n_cells']:5d} {d['stage1_acc']:7.3f} "
                     f"{d['node_acc']:7.3f} {d['pcf1_mean']:7.3f} {d['pcf1_p10']:7.3f}   "
                     f"{_fmt(x.get('node_acc'), 7)} {_fmt(x.get('pcf1_mean'), 7)} "
                     f"{_fmt(x.get('pcf1_p10'), 7)}")
        L.append("")

    L += ["TABLE A2 -- FROZEN MODELS ON NEVER-TRAINED CLASSES (every cell)", "",
          f"  {'corpus':22s} {'cells':>5s} {'pyr-routed':>10s} "
          f"{'acc':>7s} {'pcF1':>7s} {'P10':>7s}   {'acc':>7s} {'pcF1':>7s} {'P10':>7s}",
          f"  {'':22s} {'':>5s} {'':>10s} {'---- deployment ----':>23s}   "
          f"{'--- given schema ---':>23s}"]
    for name, r in newclass.items():
        if name.startswith("_"):   # run metadata, not a corpus
            continue
        d, f = r.get("frozen_deployment_all"), r.get("frozen_all")
        if not d or not f:
            continue
        L.append(f"  {name:22s} {r['n_all']:5d} {d['stage1_pyramidal_frac']:10.1%} "
                 f"{d['node_acc']:7.3f} {d['pcf1_mean']:7.3f} {d['pcf1_p10']:7.3f}   "
                 f"{f['node_acc']:7.3f} {f['pcf1_mean']:7.3f} {f['pcf1_p10']:7.3f}")
    L.append("")

    L += ["TABLE B -- FLAG TRANSFER ON THE EXTERNAL SET", "",
          f"  pyramidal set: {flag['n_scoped']} cells routed to pyramidal; "
          f"{flag['n_routed_interneuron']} routed to interneuron, scored by the "
          f"interneuron flag (their pcF1 mean {flag['routed_interneuron_pcf1_mean']:.3f})", "",
          f"  {'reject':>8s} {'kept mean':>10s} {'kept P10':>9s}"]
    for rej, v in flag["triage"].items():
        L.append(f"  {float(rej):8.0%} {v['mean']:10.3f} {v['p10']:9.3f}")
    L += ["", "  Stage 1 confidence gate (misrouted caught vs correct sacrificed):"]
    for thr, v in flag["gate"].items():
        L.append(f"    threshold {thr}: catches {v['caught']:5.1%}, "
                 f"costs {v['cost']:5.1%}")
    L.append("")

    L += ["TABLE C -- FRESH MODELS FROM ONE CORPUS'S OWN CELLS", "",
          f"  {'corpus':22s} {'train':>5s} {'test':>5s} "
          f"{'frozen P10':>11s} {'own P10':>8s} {'frozen mean':>12s} {'own mean':>9s}"]
    for name, r in {**retrain, **newclass}.items():
        if name.startswith("_"):
            continue
        L.append(f"  {name:22s} {r['n_train']:5d} {r['n_test']:5d} "
                 f"{r['frozen']['pcf1_p10']:11.3f} {r['own']['pcf1_p10']:8.3f} "
                 f"{r['frozen']['pcf1_mean']:12.3f} {r['own']['pcf1_mean']:9.3f}")
    for name, r in {**retrain, **newclass}.items():
        if name.startswith("_"):
            continue
        if r.get("note"):
            L.append(f"  note ({name}): {r['note']}")
    L += ["", "  trained on every usable cell outside the same held-out third:"]
    for name, r in more.items():
        if name.startswith("_"):
            continue
        L.append(f"  {r['archive']:22s} {r['n_train']:5d} {r['n_test']:5d} "
                 f"{r['frozen']['pcf1_p10']:11.3f} {r['own']['pcf1_p10']:8.3f} "
                 f"{r['frozen']['pcf1_mean']:12.3f} {r['own']['pcf1_mean']:9.3f}")

    text = "\n".join(L)
    print(text)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {"external": deploy, "external_gtct": gtct,
         "external_interneuron": ideploy, "external_interneuron_gtct": igtct,
         "flag_transfer": flag, "retrain": retrain, "new_class": newclass,
         "more_cells": more},
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_TXT.relative_to(ROOT)} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
