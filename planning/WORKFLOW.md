# Lean Editing Workflow — Cost Optimization

## Principles
- **Grep-first**: Always search before reading. Exact locations → minimal context loads.
- **Batch reads**: One read = multiple edits extracted from that single pull.
- **Session notes**: Write findings immediately to `WORKFLOW-SESSION.md` to prevent rework.
- **No spot-checks**: Trust edits work unless user requests verification.
- **Compress output**: Replace summaries with "✅ Done" + issue list.
- **Ask scope upfront**: "Which 3 issues are must-fix?"

## Sequence for Future Sessions

### Step 1: Intake (1 grep)
```bash
grep -n "search_term" Perception\ Is\ How\ We\ Experience.md
```
Write results to `WORKFLOW-SESSION.md` immediately.

### Step 2: Context Pull (1 read only)
Read the exact region containing all target edits (e.g., lines 100-150).
Store findings in `WORKFLOW-SESSION.md`.

### Step 3: Batch Edits (1 edit per target)
From that single read, execute all edits sequentially without re-reading.

### Step 4: Done
No verification. Move to next issue.

## Session Template

Create `WORKFLOW-SESSION.md` at session start:

```markdown
# Session: [Date] — [Goal]

## Scope (User-Confirmed)
- [ ] Issue A (MUST)
- [ ] Issue B (MUST)
- [ ] Issue C (NICE)

## Findings (grep results)
File: Perception Is How We Experience.md
- Line 156: Contains "Hebbian" — change to "contrastive"
- Line 863: "updating synaptic weights using Hebbian" — scope to contrastive

## Context Pull (lines 150-870)
[Read once, extract all needed edits]

## Edits Applied
- [ ] Edit 1: Line 156 replace X with Y
- [ ] Edit 2: Line 863 replace A with B
- [ ] Edit 3: Line 880 replace C with D

## Status
✅ All edits applied. No verification run.
```

## Example: M4 Redux (Old vs. New)

**OLD (verbose, multiple reads)**
1. Read file to find Hebbian references
2. Grep for "Hebbian"
3. Read again to understand context
4. Edit
5. Read again to verify
6. Write 5-line summary
**Cost**: ~4 reads, 1 verify read = high token burn

**NEW (one context pull)**
1. Grep "Hebbian" → locations: 156, 863, 880, 913
2. Read lines 150-930 (one pull)
3. Make 4 edits from that read (no re-reading)
4. Write "✅ Changed Hebbian→contrastive in 4 places" to session notes
5. Done
**Cost**: 1 grep + 1 read + 4 edits = ~60% token savings

## Rules for Next Session

1. **User specifies scope first**: "Fix M4 and M5 only" (not "do Phase 2")
2. **No verification unless asked**: "Verify M4 after" if needed
3. **Grep before every read**: Never read blind
4. **One read = multiple edits**: Extract all needed targets, then execute
5. **Session notes are canonical**: Update immediately, reference before re-reading
6. **Summaries are 2 lines max**: "✅ M4 done (4 edits, contrastive paradigm). ✅ M5 done (MIME table added)."

## Estimated Token Savings
- Skip verification: -30%
- Grep-first approach: -20%
- Batch edits from one read: -25%
- Compressed communication: -15%
- **Total**: ~60% reduction (e.g., $0.59 → ~$0.24 for same work)
