# Replay Through Towers

**Status**
- Label: **Evidenced**
- Relation to current repo: strong
- Evidence status: strongest current mechanistic result
- Patent relevance: high

## Concept

Replay must reach the towers and shared backbone, not only late classifier heads, because towers encode the meaning of each modality and the backbone learns the mapping between them.

## Why It Matters

If replay only reaches late heads:
- tower semantics drift
- shared cross-modal relationships drift
- old tasks degrade sharply

If replay reaches raw inputs and the full feature path:
- modality semantics are reinforced
- shared mappings are reinforced
- forgetting is reduced

## Relation to Current Repo

Strongly evidenced in the current continual-learning lane:
- feature-only replay fails badly
- raw-input replay improves retention materially
- tuned replay-centric path remains the strongest validated result

## Source Artifacts

- [/Users/zachshallbetter/Downloads/replay_data_flow_interactive.html](/Users/zachshallbetter/Downloads/replay_data_flow_interactive.html)
- [/Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md](/Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md)
