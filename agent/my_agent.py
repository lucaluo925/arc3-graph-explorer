"""ARC-AGI-3 agent: training-free graph exploration.

Idea (per level):
  * Each distinct (masked) frame is a node. Each tried action is an edge.
  * Candidate actions per node = available keyboard actions + one click per
    connected colour component (ACTION6), ranked into priority tiers.
  * Explore untried actions of the best tier; when the current node is
    exhausted, walk (BFS over known edges) to the nearest node that still has
    untried actions; if unreachable, RESET the level and walk from the root.
  * Level-up -> new graph. Action-effect statistics carry over between levels.
  * Edge "timer bars" (small changes on the frame border) are learned and
    masked out of the state hash so step counters don't explode the graph.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

# ---------------------------------------------------------------------------
TIME_BUDGET_S = float(os.getenv("ARC_AGENT_TIME_BUDGET", 7.5 * 3600))
MAX_CLICK_CANDIDATES = 80
BORDER = 3  # band (in px) where timer bars live
UNDO_ID = 7


def _components(grid: np.ndarray):
    """4-connected same-colour components. Returns list of (color, cells ndarray)."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            c = grid[y, x]
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((int(c), np.array(cells)))
    return comps


class Node:
    __slots__ = ("key", "actions", "tried", "edges", "depth", "grid")

    def __init__(self, key: str, actions: list, depth: int, grid=None) -> None:
        self.key = key
        self.grid = grid
        self.actions = actions  # list of (tier, token)
        self.tried: set = set()
        self.edges: dict = {}  # token -> next key ("DEAD" for game over)
        self.depth = depth

    def untried(self, tier: int):
        for t, tok in self.actions:
            if t <= tier and tok not in self.tried:
                return tok
        return None

    def min_untried_tier(self):
        best = None
        for t, tok in self.actions:
            if tok not in self.tried and (best is None or t < best):
                best = t
        return best


class MyAgent(Agent):
    MAX_ACTIONS = 1_000_000  # real limit is the time budget

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.t0 = time.time()
        self.ticker = np.zeros((64, 64), dtype=bool)  # learned volatile pixels
        self.eff_stats: dict = {}  # action-class -> [changed, tried]
        self._new_level(0)
        self.prev_key = None
        self.prev_token = None
        self.prev_grid = None
        self.plan: deque = deque()
        self.just_reset = False

    # ------------------------------------------------------------------ utils
    def _new_level(self, lvl: int) -> None:
        self.level = lvl
        self.nodes: dict = {}
        self.root_key = None
        self.plan = deque()
        self.level_started = True  # no action taken yet in this level

    def _mask(self, grid: np.ndarray) -> np.ndarray:
        if not self.ticker.any() or grid.shape != self.ticker.shape:
            return grid
        g = grid.copy()
        g[self.ticker] = 255
        return g

    def _key(self, grid: np.ndarray) -> str:
        return hashlib.md5(self._mask(grid).tobytes()).hexdigest()

    def _rehash(self) -> None:
        """Ticker mask changed: recompute node keys and merge duplicates."""
        remap = {k: self._key(n.grid) for k, n in self.nodes.items()}
        remap["DEAD"] = "DEAD"
        new: dict = {}
        for k, n in self.nodes.items():
            nk = remap[k]
            n.edges = {t: remap.get(v, v) for t, v in n.edges.items()}
            if nk in new:
                m = new[nk]
                m.tried |= n.tried
                for t, v in n.edges.items():
                    m.edges.setdefault(t, v)
                m.depth = min(m.depth, n.depth)
            else:
                n.key = nk
                new[nk] = n
        self.nodes = new
        if self.root_key is not None:
            self.root_key = remap.get(self.root_key, self.root_key)
        if self.prev_key is not None:
            self.prev_key = remap.get(self.prev_key, self.prev_key)

    def _learn_ticker(self, a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape or a.shape != (64, 64):
            return
        before = int(self.ticker.sum())
        d = np.argwhere(a != b)
        if 0 < len(d) <= 4:
            h, w = a.shape
            if all(y < BORDER or y >= h - BORDER or x < BORDER or x >= w - BORDER for y, x in d):
                for y, x in d:
                    # mask the whole border line containing the pixel
                    if y < BORDER or y >= h - BORDER:
                        self.ticker[y, :] = True
                    else:
                        self.ticker[:, x] = True
            if int(self.ticker.sum()) != before:
                self._rehash()

    def _action_class(self, tok, grid) -> tuple:
        if tok[0] == 6:
            return (6, int(grid[tok[2], tok[1]]))  # colour clicked
        return (tok[0],)

    def _prior(self, cls) -> float:
        ch, tr = self.eff_stats.get(cls, (0, 0))
        return (ch + 1.0) / (tr + 2.0)

    def _candidates(self, grid: np.ndarray, avail: list) -> list:
        cands = []
        for a in avail:
            if a in (0, 6, UNDO_ID):
                continue
            cands.append((0, (a,)))
        if 6 in avail:
            m = self._mask(grid)
            comps = _components(m)
            colours, counts = np.unique(m, return_counts=True)
            bg = set(int(c) for c, n in zip(colours, counts) if n > 0.25 * m.size)
            scored = []
            for c, cells in comps:
                if c == 255:
                    continue
                n = len(cells)
                cy, cx = cells.mean(0)
                i = int(np.argmin(((cells - [cy, cx]) ** 2).sum(1)))
                y, x = int(cells[i][0]), int(cells[i][1])
                tier = 1
                if c not in bg and 2 <= n <= 400:
                    tier = 0
                elif c not in bg and n == 1:
                    tier = 1
                elif c in bg:
                    tier = 2
                scored.append((tier, -self._prior((6, c)), n, (6, x, y)))
            scored.sort()
            cands += [(t, tok) for t, _, _, tok in scored[:MAX_CLICK_CANDIDATES]]
        # keyboard actions: order by prior effectiveness
        cands.sort(key=lambda tc: (tc[0], -self._prior((tc[1][0],)) if tc[1][0] != 6 else 0))
        return cands

    def _to_action(self, tok) -> GameAction:
        act = GameAction.from_id(tok[0])
        if tok[0] == 6:
            act.set_data({"x": int(tok[1]), "y": int(tok[2])})
        return act

    def _bfs(self, start: str, tier: int):
        """Shortest known path from start to a node with untried actions <= tier."""
        if start not in self.nodes:
            return None
        q = deque([start])
        prev = {start: None}
        while q:
            k = q.popleft()
            node = self.nodes[k]
            if node.untried(tier) is not None:
                path = []
                while prev[k] is not None:
                    pk, tok = prev[k]
                    path.append(tok)
                    k = pk
                return list(reversed(path))
            for tok, nk in node.edges.items():
                if nk in self.nodes and nk not in prev and nk != k:
                    prev[nk] = (k, tok)
                    q.append(nk)
        return None

    def _global_min_tier(self):
        tiers = [n.min_untried_tier() for n in self.nodes.values()]
        tiers = [t for t in tiers if t is not None]
        return min(tiers) if tiers else None

    # --------------------------------------------------------------- framework
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return time.time() - self.t0 > TIME_BUDGET_S

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        act = self._choose(latest_frame)
        self.just_reset = act == GameAction.RESET
        return act

    def _choose(self, f: FrameData) -> GameAction:
        if f.state is GameState.NOT_PLAYED or not f.frame:
            self.prev_key = None
            return GameAction.RESET
        grid = np.array(f.frame[-1], dtype=np.int16)

        # level transition
        if f.levels_completed != self.level:
            self._new_level(f.levels_completed)
            self.prev_key = None
            self.prev_grid = None

        if self.prev_grid is not None:
            self._learn_ticker(self.prev_grid, grid)

        dead = f.state is GameState.GAME_OVER
        key = "DEAD" if dead else self._key(grid)

        # record edge from previous step
        if self.prev_key is not None and self.prev_token is not None and self.prev_key in self.nodes:
            pn = self.nodes[self.prev_key]
            pn.tried.add(self.prev_token)
            if pn.edges.get(self.prev_token) not in (None, key):
                self.plan.clear()  # nondeterminism: replan
            pn.edges[self.prev_token] = key
            cls = self._action_class(self.prev_token, self.prev_grid) if self.prev_grid is not None else None
            if cls is not None:
                st = self.eff_stats.setdefault(cls, [0, 0])
                st[1] += 1
                if key != self.prev_key:
                    st[0] += 1

        self.prev_grid = grid
        if dead:
            self.plan.clear()
            self.prev_key = None
            self.prev_token = None
            return GameAction.RESET

        if key not in self.nodes:
            depth = 0 if self.root_key is None else (self.nodes[self.prev_key].depth + 1 if self.prev_key in self.nodes else 0)
            self.nodes[key] = Node(key, self._candidates(grid, list(f.available_actions)), depth, grid)
            if self.root_key is None:
                self.root_key = key
        node = self.nodes[key]

        # follow plan if still valid
        tok = None
        if self.plan:
            nxt = self.plan.popleft()
            if nxt in node.edges or nxt in [t for _, t in node.actions]:
                tok = nxt
            else:
                self.plan.clear()
        if tok is None:
            tier = self._global_min_tier()
            if tier is None:
                # fully explored: nothing left; do random-ish retry
                tier = 99
                for n in self.nodes.values():
                    n.tried.clear()
            tok = node.untried(tier)
            if tok is None:
                path = self._bfs(key, tier)
                if path:
                    tok = path[0]
                    self.plan = deque(path[1:])
                elif self.root_key is not None and key != self.root_key and not self.level_started_guard():
                    path = self._bfs(self.root_key, tier)
                    self.plan = deque(path or [])
                    self.prev_key = None
                    self.prev_token = None
                    return GameAction.RESET
                else:
                    # stuck at root with nothing reachable: try any untried action
                    tok = node.untried(99) or node.actions[0][1] if node.actions else (1,)
        self.prev_key = key
        self.prev_token = tok
        self.level_started = False
        return self._to_action(tok)

    def level_started_guard(self) -> bool:
        # RESET with zero actions in the level triggers a FULL game reset.
        return self.level_started or self.just_reset
