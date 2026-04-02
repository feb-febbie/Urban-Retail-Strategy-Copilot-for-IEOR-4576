"""
Sandboxed Python execution tool for the Lead Strategist agent.

The agent writes pandas/geopandas/matplotlib code as a string.
This module executes it in a controlled namespace pre-loaded with the
panel dataframe and common libraries, then returns stdout + any named outputs.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from typing import Any

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")  # non-interactive backend for server use


def execute_python(code: str, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Execute analyst-written Python code with a pre-loaded namespace.

    Pre-loaded variables available in the code:
        df      – the panel DataFrame (zip_code, total_exits, complaints, …)
        pd      – pandas
        np      – numpy
        plt     – matplotlib.pyplot
        gpd     – geopandas
        stats   – scipy.stats

    To return outputs from your code, assign to these names:
        result          – dict: any JSON-serialisable summary
        candidates      – list[dict]: ranked ZIP code candidates
        fig             – matplotlib Figure: will be saved as PNG

    Returns dict with keys:
        success         – bool
        stdout          – captured print output
        result          – value of `result` if set
        candidates      – value of `candidates` if set
        visualization_b64 – base64-encoded PNG if `fig` was set
        error           – error message if success=False
        traceback       – full traceback if success=False
    """
    _df = df.copy() if df is not None else pd.DataFrame()

    def _safe_norm(s, index=None):
        """
        Min-max normalize to [0, 1].
        Always returns a pd.Series aligned to the given index (or s.index).
        Handles zero-variance (all same value) with epsilon, and scalar inputs.
        """
        if not isinstance(s, pd.Series):
            idx = index if index is not None else (range(len(_df)) if not _df.empty else range(0))
            s = pd.Series(float(s) if s is not None else 0.0, index=idx)
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "gpd": gpd,
        "stats": stats,
        "df": _df,
        "norm": _safe_norm,   # safe normalizer — always returns Series, handles zero-variance
        # Outputs the agent can set
        "result": None,
        "candidates": None,
        "fig": None,
    }

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    output: dict[str, Any] = {"success": False}

    try:
        exec(compile(code, "<agent_code>", "exec"), namespace)  # noqa: S102

        output["success"] = True
        output["stdout"] = buf.getvalue()

        # Harvest named outputs
        if namespace["result"] is not None:
            try:
                output["result"] = json.loads(json.dumps(namespace["result"], default=str))
            except Exception:
                output["result"] = str(namespace["result"])

        if namespace["candidates"] is not None:
            try:
                output["candidates"] = json.loads(
                    json.dumps(namespace["candidates"], default=str)
                )
            except Exception:
                output["candidates"] = namespace["candidates"]

        if namespace["fig"] is not None:
            fig = namespace["fig"]
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=120)
            plt.close(fig)
            img_buf.seek(0)
            output["visualization_b64"] = base64.b64encode(img_buf.read()).decode()

    except Exception as exc:
        output["success"] = False
        output["error"] = str(exc)
        output["traceback"] = traceback.format_exc()
        output["stdout"] = buf.getvalue()

    finally:
        sys.stdout = old_stdout

    return output


def format_tool_result(exec_result: dict[str, Any]) -> str:
    """
    Format the execution result as a string to feed back to the LLM
    as a tool_result message.
    """
    lines = []
    if exec_result.get("success"):
        lines.append("✅ Code executed successfully.")
        if exec_result.get("stdout"):
            lines.append(f"\nSTDOUT:\n{exec_result['stdout'].strip()}")
        if exec_result.get("result"):
            lines.append(f"\nRESULT (dict):\n{json.dumps(exec_result['result'], indent=2)}")
        if exec_result.get("candidates"):
            lines.append(
                f"\nCANDIDATES (top {min(5, len(exec_result['candidates']))} of {len(exec_result['candidates'])}):\n"
                + json.dumps(exec_result["candidates"][:5], indent=2, default=str)
            )
        if exec_result.get("visualization_b64"):
            lines.append("\n📊 Visualization generated and stored.")
    else:
        lines.append("❌ Code execution failed.")
        lines.append(f"Error: {exec_result.get('error', 'Unknown error')}")
        lines.append(f"Traceback:\n{exec_result.get('traceback', '')}")
        if exec_result.get("stdout"):
            lines.append(f"Partial stdout:\n{exec_result['stdout']}")

    return "\n".join(lines)
