# Frame-Graph Exploration — ARC-AGI-3 agent, ablations and raw results

Code and raw evaluation output behind the ARC Prize 2026 Paper Track writeup
**"Frame-Graph Exploration: A Training-Free Baseline for ARC-AGI-3"**
by Haoyuan Luo (Kaggle: `lucaluo925`).

- Writeup: https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/frame-graph-exploration-a-training-free-baseline
- Submitted notebook: https://www.kaggle.com/code/lucaluo925/arc3-graph-explorer-v1 (ARC-AGI-3 public leaderboard score 0.26)

The agent treats each level as an unknown deterministic state machine. It hashes
each frame into a graph node, masks border clocks so step counters do not make
every frame unique, orders candidate actions by tier and by a learned per-class
prior, and replans by BFS over the edges it has already seen. It uses no model,
no GPU and no training.

## What is here

    agent/my_agent.py        the submitted agent, unchanged
    agent/ablation_agent.py  wrapper that disables one component at a time
    scripts/eval.py          local evaluation over the public games
    results/*.txt            per-variant summary, one line per game
    results/per_game/        raw output per game, per variant

## Ablation summary

2,000 actions per game, shared across the levels of that game, on the 25 public
games shipped with the toolkit.

| Variant | Mean score | Levels solved |
|---|---|---|
| full | 0.308 | 15/183 |
| no_ticker (no clock masking) | 0.024 | 5/183 |
| no_tiers (flat candidate list) | 0.305 | 15/183 |
| no_prior (constant prior 0.5) | 0.122 | 13/183 |

Each line in `results/full_2000.txt` reads: game id, levels solved, actions
used, scorecard score, wall-clock time, then one tuple per level giving actions
used, the human baseline, and the score for that level.

## Reproducing

`scripts/eval.py` expects the official ARC-AGI-3-Agents framework at
`vendor/ARC-AGI-3-Agents` and the public game files. That framework is not
copied into this repository; it is MIT-licensed, Copyright 2025 ARC Prize, and
is available from the ARC Prize organisation.

    python scripts/eval.py --steps 2000 -v
    ARC_ABLATION=no_ticker python scripts/eval.py --steps 2000 -v

`ARC_ABLATION` takes `full`, `no_ticker`, `no_tiers` or `no_prior`.

## Licence

Apache-2.0, see `LICENSE`. This covers the code and results in this repository.
