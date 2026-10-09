"""The LangGraph StateGraph: four nodes (orchestrator, scientist, general, executor).

The shared state IS the dialogue: a list of turns {turn, role, text}. Every node reads the
turns already there and appends its own. The orchestrator picks the next edge; each specialist
hands the turn straight back to the orchestrator.
"""
import operator
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import prompts
from llm import call
from tools import printed_test_rmse, tail, write_and_run

MAX_DECISIONS = 8


# ----------------------------------------------------------------------------- state
class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]     # the dialogue (appended to by every node)
    decisions: Annotated[list, operator.add]    # every orchestrator decision, incl. rejected ones
    tool_calls: Annotated[list, operator.add]   # every executor tool call with its real result
    n_decisions: int
    next: str                                   # routing key set by the orchestrator
    run_dir: str
    ended_by: str                               # "finish" or "cap"


class Decision(BaseModel):
    """What the orchestrator must return on every turn."""
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(description="What that role should do now (1-4 sentences).")
    reason: str = Field(description="ONE sentence: why this role, why now.")


class RunPython(BaseModel):
    """Tool for the executor: write the code to a new attempt_<n>.py file and run it."""
    code: str = Field(description="The COMPLETE Python script.")


# ----------------------------------------------------------------------------- helpers
def as_text(content):
    """Gemini may return a string or a list of parts; always give back plain text."""
    if isinstance(content, str):
        return content
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)


def turn(state, role, text, offset=0):
    return {"turn": len(state.get("messages", [])) + 1 + offset, "role": role, "text": text}


def render(state):
    """The dialogue as one text block, each turn labelled with the role that produced it."""
    return "\n\n".join(f"[turn {m['turn']} | {m['role']}]\n{m['text']}" for m in state["messages"])


def verified_rmse(state):
    """The latest test RMSE that a script REALLY printed (checked on the stored tool result,
    never on what any model said)."""
    found = [printed_test_rmse(r) for r in state.get("tool_calls", [])]
    found = [x for x in found if x is not None]
    return found[-1] if found else None


# ----------------------------------------------------------------------------- nodes
def make_nodes(llm, max_decisions=MAX_DECISIONS):
    decide = llm.with_structured_output(Decision)

    def orchestrator(state):
        n = state.get("n_decisions", 0)
        rmse = verified_rmse(state)

        # (a) Hard cap, enforced in code: after max_decisions decisions we stop, honestly.
        if n >= max_decisions:
            text = (f"Decision cap ({max_decisions}) reached. "
                    + (f"A verified test RMSE was printed: {rmse}." if rmse is not None
                       else "NO verified test RMSE was produced; the run ends unsuccessful."))
            return {"messages": [turn(state, "system", text)], "next": "end", "ended_by": "cap"}

        status = (f"Decisions used so far: {n} of {max_decisions}. "
                  f"Code-verified test RMSE from a real run: {rmse if rmse is not None else 'none yet'}.")
        d = call(decide, [SystemMessage(prompts.ORCHESTRATOR.format(max_decisions=max_decisions)),
                          HumanMessage(render(state) + "\n\n" + status + "\nMake your decision.")])
        n += 1
        rec = {"decision": n, "next_role": d.next_role, "instruction": d.instruction,
               "reason": d.reason, "accepted": True}

        # (b) 'finish' only counts if the CODE can verify a real printed test RMSE.
        if d.next_role == "finish":
            if rmse is None:
                rec["accepted"] = False
                note = ("FINISH REJECTED by the code: no executor script has exited with code 0 "
                        "and printed 'FREE-RUN TEST RMSE: <number>'. Choose another role. "
                        f"(Orchestrator's reason was: {d.reason})")
                return {"messages": [turn(state, "system", note)], "decisions": [rec],
                        "n_decisions": n, "next": "orchestrator"}
            text = f"FINISH (verified test RMSE {rmse}). Reason: {d.reason}"
            return {"messages": [turn(state, "orchestrator", text)], "decisions": [rec],
                    "n_decisions": n, "next": "end", "ended_by": "finish"}

        text = f"-> {d.next_role}: {d.instruction}\n(reason: {d.reason})"
        return {"messages": [turn(state, "orchestrator", text)], "decisions": [rec],
                "n_decisions": n, "next": d.next_role}

    def specialist(role, system_prompt):
        """Scientist and general agent: plain text answers, NO tools."""
        def node(state):
            msg = call(llm, [SystemMessage(system_prompt),
                             HumanMessage(render(state) + f"\n\nIt is your turn, {role}. "
                                          "Follow the orchestrator's latest instruction.")])
            return {"messages": [turn(state, role, as_text(msg.content))]}
        return node

    # The executor is the ONLY node that gets a tool.
    executor_llm = llm.bind_tools([RunPython], tool_choice="any")

    def executor(state):
        prev = state.get("tool_calls", [])
        context = render(state)
        if prev:   # show the previous script so a failed one can be repaired
            context += f"\n\n[source of your previous script {prev[-1]['file'].split('/')[-1]}]\n"
            context += tail(prev[-1]["code"], 20_000)
        msg = call(executor_llm, [SystemMessage(prompts.EXECUTOR),
                                  HumanMessage(context + "\n\nIt is your turn, executor. "
                                               "Call run_python once with the full script.")])
        if not msg.tool_calls:      # the model answered in words instead of calling the tool
            text = "(executor made no tool call) " + as_text(msg.content)
            return {"messages": [turn(state, "executor", text)]}

        code = msg.tool_calls[0]["args"].get("code", "")
        rec = write_and_run(state["run_dir"], len(prev) + 1, code)   # the tool: write + run
        printed = printed_test_rmse(rec)
        text = (f"Attempt {rec['attempt']} ({rec['file'].split('/')[-1]}): exit code {rec['exit_code']}, "
                f"{rec['seconds']} s.\n--- stdout (end) ---\n{tail(rec['stdout'], 3000)}\n"
                f"--- stderr (end) ---\n{tail(rec['stderr'], 3000)}\n"
                f"Test RMSE printed by this run: {printed if printed is not None else 'none'}")
        return {"messages": [turn(state, "executor", text)], "tool_calls": [rec]}

    return orchestrator, specialist("scientist", prompts.SCIENTIST), \
        specialist("general", prompts.GENERAL), executor


# ----------------------------------------------------------------------------- graph
def build_graph(llm, max_decisions=MAX_DECISIONS):
    orchestrator, scientist, general, executor = make_nodes(llm, max_decisions)
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("scientist", scientist)
    g.add_node("general", general)
    g.add_node("executor", executor)
    g.set_entry_point("orchestrator")
    # the orchestrator chooses the next edge (or loops on itself after a rejected finish)
    g.add_conditional_edges("orchestrator", lambda s: s["next"],
                            {"scientist": "scientist", "general": "general", "executor": "executor",
                             "orchestrator": "orchestrator", "end": END})
    for role in ("scientist", "general", "executor"):
        g.add_edge(role, "orchestrator")          # specialists always hand the turn back
    return g.compile()
