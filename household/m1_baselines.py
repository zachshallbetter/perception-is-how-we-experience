#!/usr/bin/env python3
"""M1 baselines (titan-node#699): simple features on the exported windows, scored on a time-ordered split (train on
the earlier days, test on the last full day: nothing from the future leaks in).

Targets (Zach, 10-02):
  home      (a) who's home: adultN_home per window; and arrival-time error (minutes) on the test day
  tv        (b) evening routines: tv_on in evening windows (17:00-24:00 local)
  heating   (c) thermostat schedule: heating per window; setpoint error
Baselines: persistence (the previous window's label) and hour-of-week majority (from the training days; falls back to
hour-of-day, then the training majority). Reported with the test-window count, so a score on thin data reads as such.

  python3 household/m1_baselines.py windows-YYYYMMDD.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def split(ws: list[dict]) -> tuple[list[dict], list[dict]]:
    days = sorted({w["t"][:10] for w in ws})
    full = [d for d in days if sum(1 for w in ws if w["t"][:10] == d) >= 90]   # ≥ 22.5 h of windows
    if len(full) < 2:
        return ws, []
    test_day = full[-1]
    return [w for w in ws if w["t"][:10] < test_day], [w for w in ws if w["t"][:10] == test_day]


def score_binary(train: list[dict], test: list[dict], key: str, keep=lambda w: True) -> dict:
    tr = [w for w in train if key in w and keep(w)]
    te = [w for w in test if key in w and keep(w)]
    if not tr or not te:
        return {"n_test": len(te), "note": "no data"}
    how = defaultdict(Counter)
    hod = defaultdict(Counter)
    for w in tr:
        how[(w["weekday"], w["hour"])][w[key]] += 1
        hod[w["hour"]][w[key]] += 1
    overall = Counter(w[key] for w in tr).most_common(1)[0][0]

    def hw(w):
        c = how.get((w["weekday"], w["hour"])) or hod.get(w["hour"])
        return c.most_common(1)[0][0] if c else overall
    allw = sorted([w for w in train + test if key in w], key=lambda w: w["t"])
    prev = {}
    last = None
    for w in allw:
        prev[w["t"]] = last
        last = w[key]
    acc = lambda pred: round(sum(1 for w in te if pred(w) == w[key]) / len(te), 3)   # noqa: E731
    return {"n_train": len(tr), "n_test": len(te), "positives_test": sum(w[key] for w in te),
            "persistence": acc(lambda w: prev.get(w["t"]) if prev.get(w["t"]) is not None else overall),
            "hour_of_week": acc(hw), "majority": acc(lambda w: overall)}


def arrival_error(train: list[dict], test: list[dict], key: str) -> dict:
    """Minutes between the predicted (training days' mean) and the actual first evening arrival on the test day."""
    def arrivals(ws):
        out, prev = {}, None
        for w in sorted([w for w in ws if key in w], key=lambda w: w["t"]):
            if prev == 0 and w[key] == 1 and w["hour"] >= 12 and w["t"][:10] not in out:
                out[w["t"][:10]] = w["hour"] * 60 + w["minute"]
            prev = w[key]
        return out
    tr, te = arrivals(train), arrivals(test)
    if not tr or not te:
        return {"note": "no afternoon/evening arrival on the training and test days", "train_days": len(tr),
                "test_days": len(te)}
    pred = sum(tr.values()) / len(tr)
    actual = next(iter(te.values()))
    return {"train_days": len(tr), "error_min": round(abs(pred - actual))}


def main() -> int:
    ws = load(sys.argv[1])
    train, test = split(ws)
    out = {"windows": len(ws), "days": sorted({w["t"][:10] for w in ws}), "test_day": test[0]["t"][:10] if test else None}
    if not test:
        out["note"] = "fewer than 2 full days: nothing to score"
        print(json.dumps(out, indent=1))
        return 0
    homes = sorted({k for w in ws for k in w if k.endswith("_home")})
    out["home"] = {k: {**score_binary(train, test, k), "arrival": arrival_error(train, test, k)} for k in homes}
    out["tv_evening"] = score_binary(train, test, "tv_on", keep=lambda w: w["hour"] >= 17)
    out["heating"] = score_binary(train, test, "heating")
    sp = [w for w in test if "setpoint" in w]
    tr_sp = [w for w in train if "setpoint" in w]
    if sp and tr_sp:
        mean = sum(w["setpoint"] for w in tr_sp) / len(tr_sp)
        out["setpoint_mae"] = {"n_test": len(sp), "mean_baseline": round(sum(abs(w["setpoint"] - mean) for w in sp) / len(sp), 2)}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
