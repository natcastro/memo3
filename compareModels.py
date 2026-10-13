"""
MEMO EXTENSION 5 (whole file): step 6, two embedding models x three thresholds.

Usage:  python compareModels.py tesla10K.pdf

Same chunks, same script 3 reference labels, same percentile thresholds for every
model (each model has its own score scale, so thresholds are "send the top X% of
chunks"). Reports the three numbers the memo asks for, per model:
  1. recall of LLM-confirmed content the filter keeps
  2. chunks sent to the LLM per filing
  3. results at three thresholds
plus the keyword filter as the baseline and the embedding time per model.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from docAnalyzerCommon import chunk_text, embedding_scores, extract_text_from_pdf, keyword_hit

MODELS = ["embeddinggemma-2:270m", "embeddinggemma:300m"]
SEND_FRACTIONS = [0.50, 0.25, 0.10]  # share of chunks sent to the LLM

pdf = sys.argv[1] if len(sys.argv) > 1 else "tesla10K.pdf"
report = Path(f"files/script3_gap_report_{Path(pdf).stem}.json")

rows = json.loads(report.read_text())["all_rows"]
chunks = chunk_text(extract_text_from_pdf(pdf))
assert len(chunks) == len(rows), "chunk count differs from the script 3 report"

kw = np.array([keyword_hit(c) for c in chunks])
loose = np.array([r["llm_found_something"] for r in rows])
strict = np.array([
    bool(r["extraction"]) and any(v for k, v in r["extraction"].items() if k != "mitigation")
    for r in rows
])


def recall(keep, truth):
    return float((keep & truth).sum() / max(truth.sum(), 1))


results = [{
    "filter": "keywords", "send_fraction": float(kw.mean()), "chunks_sent": int(kw.sum()),
    "recall_strict": recall(kw, strict), "recall_loose": recall(kw, loose),
}]

for model in MODELS:
    start = time.time()
    scores = np.array(embedding_scores(chunks, model)[0])
    seconds = time.time() - start  # cached runs are near-instant; delete files/embedding_cache to time fresh
    for frac in SEND_FRACTIONS:
        thr = np.percentile(scores, 100 * (1 - frac))
        keep = scores >= thr
        results.append({
            "filter": model, "send_fraction": frac, "threshold": round(float(thr), 4),
            "chunks_sent": int(keep.sum()), "recall_strict": recall(keep, strict),
            "recall_loose": recall(keep, loose), "embed_seconds": round(seconds, 1),
        })

print(f"{pdf}: {len(chunks)} chunks | strict truth {strict.sum()} | loose truth {loose.sum()}\n")
print(f"{'filter':<24}{'sent':>6}{'sent%':>8}{'threshold':>11}{'recall strict':>15}{'recall loose':>14}{'embed s':>9}")
for r in results:
    print(f"{r['filter']:<24}{r['chunks_sent']:>6}{r['chunks_sent'] / len(chunks):>8.1%}"
          f"{r.get('threshold', float('nan')):>11.3f}{r['recall_strict']:>15.1%}"
          f"{r['recall_loose']:>14.1%}{r.get('embed_seconds', float('nan')):>9}")

out = Path(f"files/compare_models_{Path(pdf).stem}.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
