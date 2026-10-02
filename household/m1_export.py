#!/usr/bin/env python3
"""M1 export (titan-node#699): Home Assistant history and adults' own place events → 15-minute windows of derived
features and labels only. Runs on titan, read-only against HA and titan's places store; nothing raw is kept.

Kept: one JSON line per window at /ai/datasets/perception/m1/windows-<day>.jsonl (0600), purged after 30 days.
Excluded by decision (10-02): kids, audio, cameras. People appear as "adult1", "adult2"… never names or ids.

  python3 household/m1_export.py            (on titan, the agent env sourced; titan-node's scripts on PYTHONPATH)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(os.environ.get("PERCEPTION_M1_DIR", "/ai/datasets/perception/m1"))
KEEP_DAYS = 30
STEP = timedelta(minutes=15)
TV_ON = {"playing", "paused", "on", "buffering"}
HISTORY_DAYS = 12


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def ha_history(http, entity: str, days: int = HISTORY_DAYS, minimal: bool = True) -> list[dict]:
    """A day at a time (a single long query starting before the first recorded state comes back truncated)."""
    out, seen = [], set()
    now = datetime.now(timezone.utc)
    for d in range(days, 0, -1):
        a, b = now - timedelta(days=d), now - timedelta(days=d - 1)
        from urllib.parse import quote
        iso = lambda d: quote(d.strftime("%Y-%m-%dT%H:%M:%S+00:00"), safe="")   # noqa: E731 ("+" would read as a space)
        q = (f"/api/history/period/{iso(a)}?filter_entity_id={entity}&end_time={iso(b)}"
             + ("&minimal_response" if minimal else ""))
        rows = (http("GET", q, None) or [[]])
        for r in (rows[0] if rows else []):
            k = (r.get("last_changed"), r.get("state"), json.dumps(r.get("attributes") or {}, sort_keys=True))
            if k not in seen:
                seen.add(k)
                out.append(r)
    return sorted(out, key=lambda r: r["last_changed"])


def state_at(rows: list[dict], t: datetime) -> dict | None:
    last = None
    for r in rows:
        if _ts(r["last_changed"]) <= t:
            last = r
        else:
            break
    return last


def windows(climate: list[dict], tvs: list[list[dict]], places: dict[str, list[tuple[datetime, str, str]]],
            start: datetime, end: datetime, zone) -> list[dict]:
    out = []
    t = start
    while t < end:
        mid = t + STEP / 2
        loc = mid.astimezone(zone)
        w = {"t": t.isoformat(timespec="minutes"), "hour": loc.hour, "minute": loc.minute, "weekday": loc.weekday()}
        c = state_at(climate, mid)
        if c:
            a = c.get("attributes") or {}
            w["heating"] = 1 if a.get("hvac_action") == "heating" else 0
            for k, name in (("temperature", "setpoint"), ("target_temp_low", "setpoint_low"),
                            ("current_temperature", "inside")):
                if a.get(k) is not None:
                    w[name] = float(a[k])
        states = [state_at(rows, mid) for rows in tvs]
        if any(states):
            w["tv_on"] = 1 if any(s and s.get("state") in TV_ON for s in states) else 0
        for who, evs in places.items():
            last = [e for e in evs if e[0] <= mid]
            if last:
                w[f"{who}_home"] = 1 if (last[-1][1] == "arrived" and last[-1][2] == "home") else 0
        out.append(w)
        t += STEP
    return out


def main() -> int:
    sys.path.insert(0, "/ai/projects/titan-node/scripts")
    import home_assistant as ha
    import household_identity as hh
    import local_time
    import places as P
    zone = local_time.zone()
    states = ha._http("GET", "/api/states", None)
    clim = [s["entity_id"] for s in states if s["entity_id"].startswith("climate.")]
    tvs = [s["entity_id"] for s in states if s["entity_id"].startswith("media_player.")]
    climate = ha_history(ha._http, clim[0], minimal=False) if clim else []
    tv_rows = [ha_history(ha._http, e) for e in tvs]
    adults = [p["id"] for p in hh.get_store().people() if p.get("role") in ("admin", "adult")]
    alias = {pid: f"adult{i + 1}" for i, pid in enumerate(adults)}
    con = sqlite3.connect(f"file:{P.DB_PATH}?mode=ro", uri=True)
    pl: dict[str, list] = {}
    for person, event, place, at in con.execute("SELECT person, event, place, at FROM place_events ORDER BY at"):
        if person in alias:     # adults' own events only; kids excluded (10-02)
            pl.setdefault(alias[person], []).append((_ts(at), event, str(place or "").lower()))
    firsts = [_ts(r["last_changed"]) for rows in [climate, *tv_rows] for r in rows[:1]]
    if not firsts:
        print(json.dumps({"windows": 0, "reason": "no history"}))
        return 0
    start = min(firsts).replace(minute=0, second=0, microsecond=0)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    ws = windows(climate, tv_rows, pl, start, end, zone)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT, 0o700)
    path = OUT / f"windows-{end:%Y%m%d}.jsonl"
    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for w in ws:
            f.write(json.dumps(w) + "\n")
    cutoff = time.time() - KEEP_DAYS * 86400
    purged = 0
    for old in OUT.glob("windows-*.jsonl"):
        if old.stat().st_mtime < cutoff:
            old.unlink()
            purged += 1
    print(json.dumps({"windows": len(ws), "days": round((end - start).total_seconds() / 86400, 1), "file": str(path),
                      "adults_with_places": len(pl), "place_events": sum(len(v) for v in pl.values()),
                      "tv_entities": len(tv_rows), "purged": purged}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
