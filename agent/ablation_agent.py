"""Ablation variants of agent/my_agent.py for the Paper Track. Does NOT modify the submission agent.
Select with env ARC_ABLATION = full | no_ticker | no_tiers | no_prior
"""
import importlib.util, os
from pathlib import Path
_p = Path(__file__).resolve().parents[1] / "agent" / "my_agent.py"
_s = importlib.util.spec_from_file_location("base_agent", _p); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
MODE = os.getenv("ARC_ABLATION", "full")

class MyAgent(_m.MyAgent):
    def _learn_ticker(self, a, b):
        if MODE == "no_ticker":
            return
        return super()._learn_ticker(a, b)

    def _candidates(self, grid, avail):
        c = super()._candidates(grid, avail)
        if MODE == "no_tiers":
            c = [(0, tok) for _, tok in c]
        return c

    def _prior(self, cls):
        if MODE == "no_prior":
            return 0.5
        return super()._prior(cls)
