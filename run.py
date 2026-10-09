"""Run the agent system on a task and keep a full record of what happened.

  python run.py                      # task = task5_prompt.txt
Needs: GEMINI_API_KEY, GEMINI_MODEL, SILVERBOX_MAT (path to SNLS80mV.mat).

Each run gets its own folder runs/<timestamp>/ containing the scripts the executor wrote,
transcript.md (readable), transcript.json (messages, decisions, tool calls, final status).
"""
import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from graph import MAX_DECISIONS, build_graph, verified_rmse


def save(run_dir, state, final=None):
    """Write transcript.json and transcript.md. Called after every step, so a crash keeps the log."""
    out = {"messages": state.get("messages", []), "decisions": state.get("decisions", []),
           "tool_calls": [{k: v for k, v in r.items()} for r in state.get("tool_calls", [])],
           "final": final}
    Path(run_dir, "transcript.json").write_text(json.dumps(out, indent=2))
    md = ["# Transcript\n"]
    for m in state.get("messages", []):
        md.append(f"## Turn {m['turn']} - {m['role']}\n\n{m['text']}\n")
    md.append("## Orchestrator decisions\n")
    for d in state.get("decisions", []):
        flag = "" if d["accepted"] else "  **(REJECTED)**"
        md.append(f"- {d['decision']}. -> {d['next_role']}{flag}: {d['reason']}")
    md.append("\n## Tool calls\n")
    for r in state.get("tool_calls", []):
        md.append(f"### attempt_{r['attempt']}.py - exit code {r['exit_code']}, {r['seconds']} s\n"
                  f"stdout (end):\n```\n{r['stdout'][-3000:]}\n```\nstderr (end):\n```\n{r['stderr'][-3000:]}\n```\n")
    if final:
        md.append(f"\n## Final status\n\n```\n{json.dumps(final, indent=2)}\n```\n")
    Path(run_dir, "transcript.md").write_text("\n".join(md))


def final_status(state):
    rmse = verified_rmse(state)
    return {"verified_test_rmse": rmse, "verified": rmse is not None,
            "decisions_used": state.get("n_decisions", 0),
            "ended_by": state.get("ended_by", "error"),
            "executor_attempts": len(state.get("tool_calls", []))}


def run_graph(graph, task_text, run_dir):
    """Stream the graph, print a live trace, save the log after every step."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    state = {"messages": [{"turn": 1, "role": "user", "text": task_text}], "decisions": [],
             "tool_calls": [], "n_decisions": 0, "run_dir": str(Path(run_dir).resolve())}
    shown = 0
    try:
        for state in graph.stream(state, {"recursion_limit": 100}, stream_mode="values"):
            for m in state["messages"][shown:]:      # print only the new turns
                one_line = " ".join(m["text"].split())
                print(f"[turn {m['turn']:2d}] {m['role']:<12} {one_line[:170]}", flush=True)
            shown = len(state["messages"])
            save(run_dir, state)
    except Exception:
        final = dict(final_status(state), error=traceback.format_exc()[-2000:])
        save(run_dir, state, final)
        raise
    final = final_status(state)
    save(run_dir, state, final)
    print("\nFINAL:", json.dumps(final))
    return state, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task5_prompt.txt")
    ap.add_argument("--max-decisions", type=int, default=MAX_DECISIONS)
    args = ap.parse_args()

    mat = os.environ.get("SILVERBOX_MAT")
    if not mat or not Path(mat).exists():
        raise SystemExit("Set SILVERBOX_MAT to the path of SNLS80mV.mat")
    from llm import make_llm                     # imported here: needs the key and model env vars
    graph = build_graph(make_llm(), args.max_decisions)
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"run folder: {run_dir}")
    run_graph(graph, Path(args.task).read_text(), run_dir)


if __name__ == "__main__":
    main()
