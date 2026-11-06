"""Parallel local eval: python scripts/eval.py --steps 2000 [--games a,b]"""
import argparse, importlib.util, logging, sys, time, os, json
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

def run(args):
    gid, steps, agent_path = args
    logging.disable(logging.CRITICAL)
    import arc_agi
    from arc_agi import OperationMode
    spec = importlib.util.spec_from_file_location("ua", agent_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(gid)
    Cls = m.MyAgent; Cls.MAX_ACTIONS = steps
    a = Cls(card_id="x", game_id=gid, agent_name="e", ROOT_URL="http://localhost", record=False, arc_env=env)
    t = time.time(); a.main(); dt = time.time() - t
    sc = arc.get_scorecard()
    env_sc = [e for e in sc.environments if e.id.startswith(gid)] if hasattr(sc, "environments") else []
    d = json.loads(sc.model_dump_json())
    score = d.get("score"); lv = None
    try:
        e = d["environments"][0]["runs"][0]
        lv = list(zip(e.get("level_actions") or [], e.get("level_baseline_actions") or [], [round(x,1) for x in (e.get("level_scores") or [])]))
        score = e["score"]
    except Exception:
        pass
    return gid, a.frames[-1].levels_completed, a.frames[-1].win_levels, a.action_counter, score, lv, dt

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--games", default=None); p.add_argument("--agent", default=str(ROOT / "agent" / "my_agent.py"))
    p.add_argument("-v", action="store_true")
    a = p.parse_args()
    games = a.games.split(",") if a.games else sorted(d.name for d in (ROOT / "environment_files").iterdir())
    with Pool(min(len(games), os.cpu_count())) as pool:
        res = pool.map(run, [(g, a.steps, a.agent) for g in games])
    tot = 0
    for gid, lc, wl, n, s, lv, dt in res:
        tot += s or 0
        print(f"{gid:6} levels {lc}/{wl}  actions {n:6}  score {s:6.2f}  {dt:5.1f}s  " + (str([x for x in lv if x[0]]) if a.v and lv else ""))
    print(f"MEAN SCORE {tot/len(res):.3f}  levels {sum(r[1] for r in res)}/{sum(r[2] for r in res)}")
