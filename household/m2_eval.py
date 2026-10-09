#!/usr/bin/env python3
"""M2 evaluation (titan-node#700, decided 10-09): rolling-origin scoring and a computed sufficiency gate on the M1
windows, so no model is trained (M3) against targets that can't tell a model from a baseline.

Rolling origin: each of the last up-to-ROLLING_DAYS full days is a test day in turn, trained on every window before
it; each baseline's accuracy (or a setpoint's MAE) is reported as mean ± sd over those days, with the positives per
day, instead of M1's single test day.

The gate (computed, never judged): a target is "scorable" only when
  (a) the history holds at least MIN_DAYS full days,
  (b) every weekday has at least MIN_PER_WEEKDAY full days (an hour-of-week bin needs two samples to mean anything),
  (c) the test days' positives are between MIN_POS and MAX_POS of their windows (a target that's 95% one class can't
      tell a model from the majority), for the binary targets,
  (d) for "who's home", at least MIN_ADULTS adults have place events (one person home 95% of the time can't test it),
  (e) for a setpoint, it changed at least once across the test days (no change: no schedule to learn).
Every target stays in the report either way; a failing one says why ("insufficient: one adult").

Stop rule (epic #700): if no target passes by STOP_DATE, perception pauses; the report says so.

  python3 household/m2_eval.py windows-YYYYMMDD.jsonl [--log /ai/logs/perception-m1.jsonl] [--today YYYY-MM-DD]

--log appends one counts-and-scores-only line (no window, no time of day, no person) for the weekly run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_baselines import load, score_binary   # noqa: E402

ROLLING_DAYS = 7
MIN_DAYS = 21
MIN_PER_WEEKDAY = 2
MIN_POS, MAX_POS = 0.10, 0.90
MIN_ADULTS = 2
FULL_DAY_WINDOWS = 90          # ≥ 22.5 h of 15-minute windows
STOP_DATE = "2026-11-01"
BASELINES = ("persistence", "hour_of_week", "majority")


def full_days(ws: list[dict]) -> list[str]:
    per = Counter(w["t"][:10] for w in ws)
    return sorted(d for d, n in per.items() if n >= FULL_DAY_WINDOWS)


def _stat(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "sd": None}
    return {"mean": round(statistics.fmean(xs), 3), "sd": round(statistics.pstdev(xs), 3) if len(xs) > 1 else 0.0}


def rolling_binary(ws: list[dict], test_days: list[str], key: str, keep=lambda w: True) -> dict:
    per_day, pos, n = defaultdict(list), [], 0
    for d in test_days:
        train = [w for w in ws if w["t"][:10] < d]
        test = [w for w in ws if w["t"][:10] == d]
        r = score_binary(train, test, key, keep)
        if "note" in r:
            continue
        n += 1
        for b in BASELINES:
            per_day[b].append(r[b])
        pos.append(r["positives_test"] / r["n_test"] if r["n_test"] else 0.0)
    return {"test_days": n, **{b: _stat(per_day[b]) for b in BASELINES},
            "positives_share": [round(p, 3) for p in pos]}


def rolling_setpoint(ws: list[dict], test_days: list[str], key: str) -> dict:
    mean_err, hod_err, changes, n = [], [], 0, 0
    for d in test_days:
        tr = [w for w in ws if w["t"][:10] < d and key in w]
        te = sorted((w for w in ws if w["t"][:10] == d and key in w), key=lambda w: w["t"])
        if not tr or not te:
            continue
        n += 1
        mean = sum(w[key] for w in tr) / len(tr)
        by_hour: dict[int, list[float]] = defaultdict(list)
        for w in tr:
            by_hour[w["hour"]].append(w[key])
        hod = lambda w: (sum(by_hour[w["hour"]]) / len(by_hour[w["hour"]])) if by_hour.get(w["hour"]) else mean  # noqa: E731
        mean_err.append(sum(abs(w[key] - mean) for w in te) / len(te))
        hod_err.append(sum(abs(w[key] - hod(w)) for w in te) / len(te))
        changes += sum(1 for a, b in zip(te, te[1:]) if a[key] != b[key])
    return {"test_days": n, "mae_mean": _stat(mean_err), "mae_hour_of_day": _stat(hod_err), "changes": changes}


def gate(kind: str, result: dict, days: list[str], weekday_days: dict[int, int], adults: int) -> dict:
    """{"scorable": bool, "why": [reasons it isn't]}: the same rule for every run, computed from the numbers."""
    why = []
    if len(days) < MIN_DAYS:
        why.append(f"history: {len(days)} of {MIN_DAYS} days")
    thin = sorted(wd for wd in range(7) if weekday_days.get(wd, 0) < MIN_PER_WEEKDAY)
    if thin:
        why.append(f"hour-of-week: {len(thin)} weekday(s) with fewer than {MIN_PER_WEEKDAY} days")
    if result.get("test_days", 0) == 0:
        why.append("no test day")
    if kind in ("binary", "home") and result.get("positives_share"):
        share = statistics.fmean(result["positives_share"])
        if not MIN_POS <= share <= MAX_POS:
            why.append(f"balance: {share:.0%} positive (needs {MIN_POS:.0%}-{MAX_POS:.0%})")
    if kind == "home" and adults < MIN_ADULTS:
        why.append("insufficient: one adult" if adults == 1 else f"insufficient: {adults} adults with place events")
    if kind == "setpoint" and result.get("changes", 0) == 0:
        why.append("no setpoint change on the test days (no schedule yet)")
    return {"scorable": not why, "why": why}


def evaluate(ws: list[dict], today: str | None = None) -> dict:
    days = full_days(ws)
    test_days = days[1:][-ROLLING_DAYS:]          # each needs at least one day before it
    weekday_days = Counter(date.fromisoformat(d).weekday() for d in days)
    homes = sorted({k for w in ws for k in w if k.endswith("_home")})
    out: dict = {"windows": len(ws), "full_days": len(days), "test_days": test_days, "targets": {}}
    for k in homes:
        r = rolling_binary(ws, test_days, k)
        out["targets"][k] = {**r, "gate": gate("home", r, days, weekday_days, len(homes))}
    if not homes:
        out["targets"]["home"] = {"test_days": 0, "gate": {"scorable": False, "why": ["insufficient: no adult with place events"]}}
    r = rolling_binary(ws, test_days, "tv_on", keep=lambda w: w["hour"] >= 17)
    out["targets"]["tv_evening"] = {**r, "gate": gate("binary", r, days, weekday_days, len(homes))}
    r = rolling_binary(ws, test_days, "heating")
    out["targets"]["heating"] = {**r, "gate": gate("binary", r, days, weekday_days, len(homes))}
    for key in ("setpoint", "setpoint_low", "setpoint_high"):
        if any(key in w for w in ws):
            r = rolling_setpoint(ws, test_days, key)
            out["targets"][key] = {**r, "gate": gate("setpoint", r, days, weekday_days, len(homes))}
    passing = sorted(t for t, v in out["targets"].items() if v["gate"]["scorable"])
    today = today or date.today().isoformat()
    out["passing"] = passing
    out["status"] = ("m3-ready" if passing else
                     "pause: no target passed the gate by the stop date" if today >= STOP_DATE else "gathering")
    return out


def log_line(out: dict, run_day: str) -> dict:
    """The weekly log's line: counts, scores and the gate only."""
    return {"day": run_day, "windows": out["windows"], "full_days": out["full_days"], "status": out["status"],
            "passing": out["passing"],
            "targets": {t: {"scorable": v["gate"]["scorable"], "why": v["gate"]["why"],
                            **({b: v[b]["mean"] for b in BASELINES if b in v})}
                        for t, v in out["targets"].items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("--log")
    ap.add_argument("--today")
    a = ap.parse_args(argv)
    out = evaluate(load(a.windows), a.today)
    print(json.dumps(out, indent=1))
    if a.log:
        p = Path(a.log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(log_line(out, a.today or date.today().isoformat())) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
