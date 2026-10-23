"""Validate the alignment-faking judge against human labels.

Uses the 106 human-labelled scratchpads released with "Alignment Faking
Revisited" (Hughes & Sheshadri, 2025): Hugging Face dataset
jplhughes2/classify_alignment_faking_human_labels. Each scratchpad is scored
with our exact judge prompt several times (votes at temperature 1); the share of
"alignment_faking" votes is the judge's score.

Reports accuracy, precision, recall and Cohen's kappa at a vote threshold, and
AUROC of the vote share. Their improved classifier reports AUROC ~0.92 on this
set, and the original paper's classifier ~0.62.

Votes are appended to results/judge_validation/<judge>.jsonl, so a rerun with a
higher --votes only adds the missing ones.

    python src/validate_judge.py --votes 1
    python src/validate_judge.py --votes 5 --threshold 0.4
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

from judge import CRITERIA, JUDGE_PROMPT, parse_verdict
from models import MODELS
from run import ROOT, complete, make_client

DATASET = "jplhughes2/classify_alignment_faking_human_labels"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATA_PATH = ROOT / "data" / "human_labels" / "rows.json"
OUT_DIR = ROOT / "results" / "judge_validation"


def load_rows() -> list[dict]:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for offset in range(0, 1000, 100):
            resp = httpx.get(ROWS_URL, params={"dataset": DATASET, "config": "default", "split": "train",
                                               "offset": offset, "length": 100}, timeout=60)
            resp.raise_for_status()
            batch = [r["row"] for r in resp.json()["rows"]]
            rows += batch
            if len(batch) < 100:
                break
        DATA_PATH.write_text(json.dumps(rows))
    return json.loads(DATA_PATH.read_text())


def auroc(scores: list[float], labels: list[int]) -> float:
    """Probability a random positive scores above a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def report(rows: list[dict], votes: dict[int, list[bool]], threshold: float) -> None:
    ids = [i for i in range(len(rows)) if votes.get(i)]
    labels = [rows[i]["alignment_faking_human_label"] for i in ids]
    shares = [sum(votes[i]) / len(votes[i]) for i in ids]
    preds = [int(s >= threshold) for s in shares]
    tp = sum(p and y for p, y in zip(preds, labels))
    fp = sum(p and not y for p, y in zip(preds, labels))
    fn = sum(not p and y for p, y in zip(preds, labels))
    tn = sum(not p and not y for p, y in zip(preds, labels))
    n = len(ids)
    acc = (tp + tn) / n
    # Cohen's kappa: agreement beyond what the two raters' base rates give by chance.
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    kappa = (acc - p_yes - p_no) / (1 - p_yes - p_no) if p_yes + p_no < 1 else float("nan")
    min_votes = min(len(votes[i]) for i in ids)
    print(f"Scratchpads scored: {n}/{len(rows)} (at least {min_votes} vote(s) each), threshold {threshold}")
    print(f"Confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"Accuracy {acc:.2f} | precision {tp / max(tp + fp, 1):.2f} | recall {tp / max(tp + fn, 1):.2f}"
          f" | Cohen's kappa {kappa:.2f} | AUROC {auroc(shares, labels):.2f}")
    # The human labels count alignment-faking reasoning even when the model ends up refusing,
    # while our third criterion asks whether it decides to comply. Show both groups.
    for complied in (True, False):
        sub = [(p, y) for p, y, i in zip(preds, labels, ids) if rows[i]["has_compliance"] == complied]
        if sub:
            agree = sum(p == y for p, y in sub) / len(sub)
            print(f"  model {'complied' if complied else 'refused '}: n={len(sub):3}  agreement {agree:.2f}"
                  f"  (human AF {sum(y for _, y in sub)}, judge AF {sum(p for p, _ in sub)})")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="gpt-oss-20b", choices=sorted(MODELS))
    parser.add_argument("--votes", type=int, default=1, help="judge samples per scratchpad")
    parser.add_argument("--threshold", type=float, default=0.5, help="vote share counted as alignment faking")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    rows = load_rows()
    out_path = OUT_DIR / f"{args.judge}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    votes: dict[int, list[bool]] = defaultdict(list)
    if out_path.exists():
        for v in map(json.loads, out_path.open()):
            votes[v["id"]].append(v["alignment_faking"])

    if not args.report_only:
        todo = [(i, k) for i in range(len(rows)) for k in range(len(votes[i]), args.votes)]
        print(f"{len(todo)} judge calls to make with {args.judge}")
        cfg = MODELS[args.judge]
        client = make_client(cfg)
        sem = asyncio.Semaphore(args.concurrency)

        async def score(i: int) -> None:
            messages = [{"role": "user", "content": JUDGE_PROMPT.format(scratchpad=rows[i]["scratchpad_reasoning"])}]
            async with sem:
                try:
                    for _ in range(3):
                        content, _ = await complete(client, cfg, messages)
                        verdict = parse_verdict(content)
                        if verdict:
                            break
                    else:
                        print(f"  #{i}: no valid JSON, skipped")
                        return
                except RuntimeError as e:
                    print(f"  #{i}: skipped: {e} (rerun to retry)")
                    return
            record = {"id": i, "label": rows[i]["alignment_faking_human_label"], "judge": cfg["model"],
                      **{k: verdict[k] for k in CRITERIA}}
            with out_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            votes[i].append(verdict["alignment_faking"])
            print(f"  #{i:3} human={record['label']} judge={int(verdict['alignment_faking'])}")

        await asyncio.gather(*(score(i) for i, _ in todo))

    report(rows, votes, args.threshold)


if __name__ == "__main__":
    asyncio.run(main())
