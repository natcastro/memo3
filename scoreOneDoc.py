"""
MEMO EXTENSION 4 (whole file): step 5, look at the embedding score distribution.

Usage:  python scoreOneDoc.py tesla10K.pdf [model]

Embeds every chunk of one 10-K, then compares the scores against what script 3
found (files/script3_gap_report_<name>.json). Two definitions of "real content":
  - loose : the LLM returned anything at all (inflated: phi4-mini fills `mitigation`)
  - strict: the LLM returned something in a category other than `mitigation`
No LLM calls here, only embeddings.
"""
import json
import sys
from pathlib import Path

import numpy as np

from docAnalyzerCommon import (
    EMBED_MODEL, chunk_text, embedding_scores, extract_text_from_pdf, keyword_hit,
)

pdf = sys.argv[1] if len(sys.argv) > 1 else "tesla10K.pdf"
model = sys.argv[2] if len(sys.argv) > 2 else EMBED_MODEL
report = Path(f"files/script3_gap_report_{Path(pdf).stem}.json")

rows = json.loads(report.read_text())["all_rows"]
chunks = chunk_text(extract_text_from_pdf(pdf))
assert len(chunks) == len(rows), "chunk count differs from the script 3 report"

scores, cats = embedding_scores(chunks, model)
scores = np.array(scores)

kw = np.array([keyword_hit(c) for c in chunks])
loose = np.array([r["llm_found_something"] for r in rows])
strict = np.array([
    bool(r["extraction"]) and any(v for k, v in r["extraction"].items() if k != "mitigation")
    for r in rows
])


def auc(pos_mask):
    """P(random positive chunk scores higher than random negative chunk)."""
    pos, neg = scores[pos_mask], scores[~pos_mask]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


print(f"{pdf} | model={model} | chunks={len(chunks)}")
print(f"keyword filter keeps {kw.sum()} | loose truth {loose.sum()} | strict truth {strict.sum()}\n")

print("Score distribution (all chunks):")
for p in (0, 10, 25, 50, 75, 90, 95, 100):
    print(f"  p{p:<3} {np.percentile(scores, p):.3f}")

print(f"\nMean score  strict-positive: {scores[strict].mean():.3f}   other chunks: {scores[~strict].mean():.3f}")
print(f"Separation (AUC; 0.5 = no better than chance, 1.0 = perfect)")
print(f"  vs strict truth: {auc(strict):.3f}")
print(f"  vs loose truth : {auc(loose):.3f}")

print("\nCandidate thresholds (by percentile of this document's scores):")
print(f"{'pct':>5} {'thr':>7} {'sent':>5} {'sent%':>6} {'recall strict':>14} {'recall loose':>13}")
for p in (50, 75, 90, 95):
    thr = np.percentile(scores, p)
    keep = scores >= thr
    print(f"{p:>5} {thr:7.3f} {keep.sum():5d} {keep.mean():6.1%} "
          f"{(keep & strict).sum() / max(strict.sum(), 1):14.1%} "
          f"{(keep & loose).sum() / max(loose.sum(), 1):13.1%}")
print(f"{'kw':>5} {'--':>7} {kw.sum():5d} {kw.mean():6.1%} "
      f"{(kw & strict).sum() / max(strict.sum(), 1):14.1%} "
      f"{(kw & loose).sum() / max(loose.sum(), 1):13.1%}   <- keyword filter")

print("\nTop 10 chunks by score:")
for i in np.argsort(-scores)[:10]:
    print(f"  chunk {i:3d} score={scores[i]:.3f} cat={cats[i]:<17} kw={int(kw[i])} strict={int(strict[i])} "
          f"| {chunks[i][:90].replace(chr(10), ' ')}")
