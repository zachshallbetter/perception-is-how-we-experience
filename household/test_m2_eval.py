#!/usr/bin/env python3
"""M2 evaluation (titan-node#700): rolling-origin scoring and the sufficiency gate, on synthetic windows only (no
household data). Run: python3 household/test_m2_eval.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_eval as M  # noqa: E402

passed = failed = 0


def check(label, got, expected):
    global passed, failed
    ok = got == expected
    passed, failed = passed + ok, failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f": got={got!r:.400} expected={expected!r:.400}"))


def make(days: int, adults: int = 1, tv=lambda d, h: h >= 19, heat=lambda d, h: h in (6, 7), home=lambda a, d, h: True,
         setpoint=lambda d, h: 20.0):
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)   # a Monday
    ws = []
    for i in range(days * 96):
        t = start + timedelta(minutes=15 * i)
        d, h = i // 96, t.hour
        w = {"t": t.isoformat(), "weekday": t.weekday(), "hour": h, "minute": t.minute,
             "tv_on": int(tv(d, h)), "heating": int(heat(d, h)), "setpoint_low": setpoint(d, h)}
        for a in range(adults):
            w[f"adult{a + 1}_home"] = int(home(a, d, h))
        ws.append(w)
    return ws


print("── rolling origin ──")
out = M.evaluate(make(10), today="2026-10-09")
check("the last 7 full days are test days, each with a day before it", (len(out["test_days"]), out["full_days"]), (7, 10))
tv = out["targets"]["tv_evening"]
check("each baseline reports mean ± sd over the test days, with positives per day",
      (tv["test_days"], sorted(tv["hour_of_week"]), len(tv["positives_share"])), (7, ["mean", "sd"], 7))
check("a 2-day history has 1 test day; a 1-day history has none", (len(M.evaluate(make(2))["test_days"]),
      len(M.evaluate(make(1))["test_days"])), (1, 0))

print("── the gate: computed, and every target stays in the report ──")
g = {t: v["gate"] for t, v in out["targets"].items()}
check("10 days: nothing passes (history and hour-of-week), and every target is listed",
      (sorted(g), all(not x["scorable"] for x in g.values()),
       all(any(w.startswith("history: 10 of 21") for w in x["why"]) for x in g.values())),
      (["adult1_home", "heating", "setpoint_low", "tv_evening"], True, True))
check("who's home with one adult: 'insufficient: one adult' (kept, not dropped)",
      "insufficient: one adult" in g["adult1_home"]["why"], True)
check("a target that's 100% one class fails on balance", any(w.startswith("balance: 100%") for w in g["adult1_home"]["why"]), True)
check("a setpoint that never changed: no schedule yet", any("no setpoint change" in w for w in g["setpoint_low"]["why"]), True)
check("status before the stop date with nothing passing: gathering", out["status"], "gathering")

print("── enough data: targets pass on their own merits ──")
big = M.evaluate(make(28, adults=2, home=lambda a, d, h: not (9 <= h < 17) if a == 0 else h < 8 or h >= 18,
                      setpoint=lambda d, h: 18.0 if h < 6 else 21.0), today="2026-10-20")
gb = {t: v["gate"] for t, v in big["targets"].items()}
check("28 days, two adults, balanced: TV, home and the setpoint pass", sorted(t for t, x in gb.items() if x["scorable"]),
      ["adult1_home", "adult2_home", "setpoint_low", "tv_evening"])
check("…heating (2 of 24 hours, 8%) still fails on balance", gb["heating"]["why"], ["balance: 8% positive (needs 10%-90%)"])
check("…status: m3-ready, listing what passed", (big["status"], big["passing"]),
      ("m3-ready", ["adult1_home", "adult2_home", "setpoint_low", "tv_evening"]))

print("── the stop rule ──")
check("nothing passing on or after 11-01: pause", M.evaluate(make(10), today="2026-11-01")["status"],
      "pause: no target passed the gate by the stop date")

print("── the weekly log line: counts, scores and the gate only ──")
with tempfile.TemporaryDirectory() as d:
    src, log = Path(d) / "w.jsonl", Path(d) / "log" / "perception-m1.jsonl"
    src.write_text("\n".join(json.dumps(w) for w in make(10)))
    M.main([str(src), "--log", str(log), "--today", "2026-10-11"])
    M.main([str(src), "--log", str(log), "--today", "2026-10-18"])
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    check("one line per run, appended", [l["day"] for l in lines], ["2026-10-11", "2026-10-18"])
    flat = json.dumps(lines[0])
    check("…no window, no time, no raw label in it", ('"t"' in flat, "T00:" in flat, '"hour"' in flat), (False, False, False))
    check("…each target's gate and baseline means", sorted(lines[0]["targets"]["tv_evening"]),
          ["hour_of_week", "majority", "persistence", "scorable", "why"])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
