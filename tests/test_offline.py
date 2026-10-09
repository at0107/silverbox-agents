"""Offline checks of the system's own logic. NO API key and NO network needed:
a scripted fake model stands in for Gemini; the graph and the executor tool are the real ones.

  cd <some folder without a .env> && python /path/to/tests/test_offline.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["GEMINI_API_KEY"] = "SECRET-FOR-TEST"       # a fake value, to prove the child never sees it

from langchain_core.messages import AIMessage

import tools
from graph import Decision, build_graph
from run import run_graph


class FakeLLM:
    """Pretends to be the chat model. `decisions` scripts the orchestrator, `codes` the executor."""
    def __init__(self, decisions, codes=()):
        self.decisions, self.codes = list(decisions), list(codes)
    def with_structured_output(self, schema):
        outer = self
        class S:
            def invoke(self, messages):
                role, instr = outer.decisions.pop(0) if len(outer.decisions) > 1 else outer.decisions[0]
                return Decision(next_role=role, instruction=instr, reason="scripted")
        return S()
    def bind_tools(self, tools_, **kw):
        outer = self
        class T:
            def invoke(self, messages):
                code = outer.codes.pop(0)
                return AIMessage(content="", tool_calls=[{"name": "RunPython", "args": {"code": code}, "id": "x"}])
        return T()
    def invoke(self, messages):                         # scientist / general
        return AIMessage(content="1. do the thing\n2. print the result")


GOOD = "print('FREE-RUN TRAIN RMSE: 0.5')\nprint('FREE-RUN TEST RMSE: 0.25')\n"
CRASH = "raise RuntimeError('boom')\n"
FAKE_BUT_FAILS = "print('FREE-RUN TEST RMSE: 0.0001')\nraise SystemExit(3)\n"
ENVCHECK = ("import os\nprint('KEY_VISIBLE' if 'GEMINI_API_KEY' in os.environ else 'KEY_HIDDEN')\n"
            "print('FREE-RUN TEST RMSE: 0.3')\n")


def go(llm, name, max_decisions=8):
    d = Path(tempfile.mkdtemp()) / name
    return run_graph(build_graph(llm, max_decisions), "task text", d)


def test_reject_repair_finish():
    llm = FakeLLM([("finish", "too early"), ("scientist", "plan"), ("executor", "run"),
                   ("executor", "repair"), ("executor", "again"), ("finish", "done")],
                  codes=[CRASH, FAKE_BUT_FAILS, GOOD])
    state, final = go(llm, "t1")
    d = state["decisions"]
    assert d[0]["next_role"] == "finish" and d[0]["accepted"] is False       # early finish rejected
    assert final["verified"] and final["verified_test_rmse"] == 0.25 and final["ended_by"] == "finish"
    assert final["decisions_used"] == 6                                      # rejection counted
    codes = [r["exit_code"] for r in state["tool_calls"]]
    assert codes == [1, 3, 0], codes                                         # crash, fake-but-failed, good
    files = [Path(r["file"]).name for r in state["tool_calls"]]
    assert files == ["attempt_1.py", "attempt_2.py", "attempt_3.py"]         # a NEW file each time
    print("ok: early finish rejected, failed runs do not count, new file per attempt")


def test_cap_without_result():
    state, final = go(FakeLLM([("general", "chat")]), "t2")                  # orchestrator never finishes
    assert final["decisions_used"] == 8 and final["ended_by"] == "cap" and not final["verified"]
    assert "NO verified" in state["messages"][-1]["text"]
    print("ok: stops after exactly 8 decisions and says no verified RMSE")


def test_finish_rejected_until_cap():
    state, final = go(FakeLLM([("finish", "now")]), "t3")                    # keeps asking to finish
    assert final["decisions_used"] == 8 and not final["verified"]
    assert all(not d["accepted"] for d in state["decisions"])
    print("ok: repeated unverified finishes are rejected, run ends at the cap")


def test_child_has_no_key_and_guards():
    state, final = go(FakeLLM([("executor", "run"), ("executor", "run"), ("finish", "x")],
                              codes=[ENVCHECK, "open('/home/alik/silverbox/data.py')"]), "t4")
    assert "KEY_HIDDEN" in state["tool_calls"][0]["stdout"]                  # key removed from child env
    assert state["tool_calls"][1]["exit_code"] == -2                         # forbidden path refused
    print("ok: key hidden from child, forbidden path refused")


def test_timeout():
    tools.TIMEOUT_S = 2
    r = tools.write_and_run(tempfile.mkdtemp(), 1, "import time\nprint('start', flush=True)\ntime.sleep(30)\n")
    assert r["exit_code"] == -1 and "TIMEOUT" in r["stderr"] and "start" in r["stdout"]
    tools.TIMEOUT_S = 300
    print("ok: timeout kills the script and keeps partial output")


if __name__ == "__main__":
    for t in (test_reject_repair_finish, test_cap_without_result, test_finish_rejected_until_cap,
              test_child_has_no_key_and_guards, test_timeout):
        t()
    print("ALL OFFLINE TESTS PASSED")
