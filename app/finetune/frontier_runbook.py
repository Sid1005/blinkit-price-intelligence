"""Frontier fine-tuning pipeline — structurally complete (week 6).

There is no OpenAI/Anthropic/Google chat-FT key here (only Groq + HF), and Groq does
not expose a customer fine-tuning endpoint, so the *frontier* fine-tune cannot run in
this environment. This module builds the exact JSONL training file in OpenAI's
chat fine-tune format, emits the precise external commands (mirrored in RUNBOOK.md),
and records W&B-compatible job metadata so the user can wire it with their own key.

Everything up to the credentialed API call is real and runnable.
"""
from __future__ import annotations

import json

from app import config
from app.finetune import dataset
from app.monitoring import experiment_tracking as tracking

SYS = ("You are India Commerce SignalForge's analyst. Given an Indian shopping or "
       "complaint snippet (English or Hinglish), output the single commerce-signal label.")


def build_openai_jsonl(out_name: str = "frontier_ft_train.jsonl") -> str:
    """Convert the curated dataset into OpenAI chat fine-tune format."""
    rows = dataset.load("train")
    fp = config.DATA_DIR / out_name
    with fp.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": r["text"]},
                {"role": "assistant", "content": r["label"]},
            ]}, ensure_ascii=False) + "\n")
    return str(fp)


def record_job_metadata() -> dict:
    """Log the (deferred) frontier-FT job shape to the experiment tracker."""
    jsonl = build_openai_jsonl()
    run = tracking.start_run(name="frontier-ft-runbook",
                             config_dict={"provider": "openai", "base_model": "gpt-4.1-nano",
                                          "train_file": jsonl, "status": "deferred_no_key",
                                          "format": "openai_chat_jsonl"},
                             tags=["week6", "frontier_ft", "runbook"])
    tracking.log_artifact(run, jsonl, type="dataset", name="frontier_ft_train")
    tracking.finish_run(run, status="deferred",
                        summary={"reason": "no frontier FT key in this environment"})
    return {"jsonl": jsonl, "run_id": run.run_id, "backend": run.backend}


RUNBOOK_STEPS = """\
# Frontier fine-tune (EXTERNAL — requires an OpenAI key, NOT available here)
# 1. export OPENAI_API_KEY=sk-...
# 2. openai api files.create -f {jsonl} -p fine-tune
# 3. openai api fine_tuning.jobs.create -t <file_id> -m gpt-4.1-nano
# 4. Poll: openai api fine_tuning.jobs.list
# 5. Set the resulting model id in an env (FRONTIER_FT_MODEL) and route to it.
# (Optional) Add WANDB integration: the OpenAI<->W&B sync logs train/val loss to W&B.
"""


if __name__ == "__main__":
    meta = record_job_metadata()
    print("Wrote", meta["jsonl"])
    print(RUNBOOK_STEPS.format(jsonl=meta["jsonl"]))
