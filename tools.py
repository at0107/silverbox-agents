"""The ONE tool in the system, used only by the executor role.

write_and_run(...) does two things:
  (i)  writes the given Python source into the run directory as attempt_<n>.py
  (ii) runs it with the same Python that runs this program, with a 5-minute limit,
       and returns the exit code, standard output and standard error.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = 300
KEEP_CHARS = 20_000          # how much of each stream we keep in the stored record
# Environment variables the child process must NOT see (it has no business with the API key)
SECRET_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL")
# The executor must not peek at the earlier, hand-written solution to this task.
FORBIDDEN = ("/home/alik/silverbox/", "silverbox_pinn", "/home/alik/silverbox ")

# The line a script must print for the run to count as having a test RMSE.
RMSE_RE = re.compile(r"^FREE-RUN TEST RMSE:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$", re.M)


def tail(text, n):
    """Keep the END of a long text (that is where tracebacks and final numbers are)."""
    return text if len(text) <= n else f"[... {len(text) - n} chars cut ...]\n" + text[-n:]


def write_and_run(run_dir, attempt, code):
    """Write attempt_<attempt>.py into run_dir and run it. Returns a record (dict)."""
    run_dir = Path(run_dir)
    path = run_dir / f"attempt_{attempt}.py"
    rec = {"attempt": attempt, "file": str(path), "code": code}

    # Guard: refuse code that points at the earlier solution (it would be cheating).
    bad = [s for s in FORBIDDEN if s in code]
    if bad:
        rec.update(exit_code=-2, stdout="", seconds=0.0,
                   stderr=f"REFUSED: the script mentions a forbidden path/name {bad}. "
                          "Write the solution from scratch; read data only via SILVERBOX_MAT.")
        return rec

    path.write_text(code)                                    # (i) write
    env = {k: v for k, v in os.environ.items() if k not in SECRET_VARS}
    t0 = time.time()
    try:                                                     # (ii) run, no shell, cwd = run dir
        p = subprocess.run([sys.executable, path.name], cwd=run_dir, env=env,
                           capture_output=True, text=True, timeout=TIMEOUT_S)
        rec.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        dec = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
        rec.update(exit_code=-1, stdout=dec(e.stdout),
                   stderr=dec(e.stderr) + f"\nTIMEOUT: killed after {TIMEOUT_S} s")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["stdout"] = tail(rec["stdout"], KEEP_CHARS)
    rec["stderr"] = tail(rec["stderr"], KEEP_CHARS)
    return rec


def printed_test_rmse(rec):
    """Return the test RMSE a run really printed, or None.
    Counts only if the process exited with code 0 and its real stdout has the line."""
    if rec.get("exit_code") != 0:
        return None
    m = RMSE_RE.findall(rec.get("stdout", ""))
    return float(m[-1]) if m else None
