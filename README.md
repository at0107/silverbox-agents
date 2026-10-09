# Agentic system for the Silverbox task (Task 6)

Four roles in a LangGraph `StateGraph` (orchestrator, scientist, general, executor) share one
state: the dialogue. The system is given Task 5 and has to plan, code and run the fit itself.

## Files
| File | What it does |
|---|---|
| `llm.py` | Builds the Gemini chat model from `GEMINI_MODEL`, reads `GEMINI_API_KEY` (env or git-ignored `.env`, never printed), and retries 429 errors with exponential backoff. |
| `check_gemini.py` | Lists the Flash models the key can use; with `GEMINI_MODEL` set, makes one plain and one structured call. |
| `prompts.py` | One system prompt per role (job and limits only, no solution hints). |
| `tools.py` | The only tool: `write_and_run` writes `attempt_<n>.py` into the run folder and runs it (5 min limit, no shell, API key removed from the child's environment). Also finds the printed test RMSE. |
| `graph.py` | State, the four nodes, the conditional edge, and the two rules enforced in code (below). |
| `run.py` | Runs the graph on a task, prints a live trace, writes `runs/<timestamp>/transcript.md` and `.json`. |
| `task5_prompt.txt` | Task 5 text given to the system, plus the extra run requirements. |
| `tests/test_offline.py` | Checks the logic with a scripted fake model (no key, no network). |

## Rules enforced in code (not by asking the model nicely)
- The orchestrator returns `next_role / instruction / reason` (pydantic structured output) and has
  no tools. At most 8 decisions; then the run ends and says whether a verified RMSE exists.
- `finish` is accepted only if a stored tool result has exit code 0 and a real stdout line
  `FREE-RUN TEST RMSE: <number>`. Otherwise the finish is rejected, recorded, and counts as a decision.
- Only the executor has a tool. Each executor turn writes a NEW file.
- The executor script runs with `GEMINI_API_KEY` removed from its environment.

Limit to know about: "verified" means a script that really ran printed that line. The system does
not independently recompute the number, so a badly written script could print a wrong value.

## Run
    export GEMINI_API_KEY=...            # or put it in .env
    export GEMINI_MODEL=<model id>       # python check_gemini.py lists the choices
    export SILVERBOX_MAT=/path/to/SNLS80mV.mat   # from https://www.nonlinearbenchmark.org/ (not in this repo)
    python check_gemini.py
    python run.py
    python tests/test_offline.py         # offline logic checks (run from a folder without a .env)

## Runs so far
All transcripts are kept in `runs/<timestamp>/`. Test RMSE is the one printed by the executor's script
(zero prediction: about 0.06 V).

| Run | Model | Executor rules | Outcome |
|---|---|---|---|
| 1 `20261009-012232` | gemini-2.5-flash | first version | Failed before any agent spoke: model not available to new users (404). 0 decisions. |
| 2 `20261009-012446` | gemini-3.8-flash | first version | Failed before any agent spoke: 503 overload (retry then only covered 429; fixed afterwards). 0 decisions. |
| 3 `20261009-012555` | gemini-3.8-flash | first version | Completed in 4 decisions, verified test RMSE 0.001527 V (train 0.001502 V). Caveats: the plan listed linear physics terms as an optional bias; the script used them and also added a trainable cubic term that the plan's formula did not contain, and it chose its best checkpoint using test RMSE, which the plan did not mention (here best = last, so the number was not changed by it). |
| 4 `20261009-160710` | gemini-3.8-flash | tightened (exact plan, no use of test samples) | Failed: orchestrator decided once, then the scientist's call got 503 "high demand" for the whole retry window (~5 min). 1 decision, no script written. |
| 5 `20261009-161539` | gemini-3.8-flash | tightened | Failed: free-tier daily quota (20 requests/day/model) of gemini-3.8-flash used up after 1 decision; no script written. |
| 6 `20261009-164237` | gemini-3.5-flash-lite | tightened | Completed in 3 decisions, 1 executor attempt (72 s). Verified test RMSE **0.085114 V** (train 0.090348 V): worse than the 0.0605 V zero-prediction baseline; the free run is a flat line. Plan followed for the architecture (3x64 SiLU MLP), no test samples used for decisions, but the script skipped the planned input scaling, cosine schedule and batching. The finish gate accepted it because a test RMSE was printed; it does not check quality. |
