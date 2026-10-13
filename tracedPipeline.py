"""
MEMO EXTENSION 6 (whole file): step 7, traced run of the embedding-prefilter pipeline.

Usage:  python tracedPipeline.py tesla10K.pdf [model] [fraction]
        e.g. python tracedPipeline.py tesla10K.pdf embeddinggemma-2:270m 0.10

Needs LANGSMITH_TRACING=true and LANGSMITH_API_KEY in the environment. One run =
one trace: the embedding prefilter picks the top `fraction` of chunks, then script
2's validated extractor runs on only those. The LLM calls show up as child runs
automatically (LangChain traces itself). Chunk text is NOT sent as the prefilter's
input, only a summary, so the trace stays small.
"""
import json
import sys
from pathlib import Path

import numpy as np
from langsmith import traceable

from docAnalyzerCommon import chunk_text, embedding_scores, extract_text_from_pdf
from script2PydanticStructured import MODEL_NAME as LLM_NAME, extract_with_retry

pdf = sys.argv[1] if len(sys.argv) > 1 else "tesla10K.pdf"
embed_model = sys.argv[2] if len(sys.argv) > 2 else "embeddinggemma-2:270m"
fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

chunks = chunk_text(extract_text_from_pdf(pdf))


@traceable(name="embedding_prefilter", run_type="chain")
def prefilter(pdf_name: str, model: str, send_fraction: float) -> dict:
    scores = np.array(embedding_scores(chunks, model)[0])
    threshold = float(np.percentile(scores, 100 * (1 - send_fraction)))
    kept = [int(i) for i in np.where(scores >= threshold)[0]]
    return {"total_chunks": len(chunks), "threshold": round(threshold, 4), "kept_indices": kept}


@traceable(name="lab3_embedding_pipeline", run_type="chain")
def pipeline(pdf_name: str, model: str, send_fraction: float, llm: str) -> dict:
    selected = prefilter(pdf_name, model, send_fraction)
    found = 0
    failed = 0
    for i in selected["kept_indices"]:
        disclosure, _, _ = extract_with_retry(chunks[i])
        if disclosure is None:
            failed += 1
        elif not disclosure.is_empty():
            found += 1
    return {
        "chunks_total": selected["total_chunks"],
        "chunks_sent_to_llm": len(selected["kept_indices"]),
        "llm_found_something": found,
        "llm_failed": failed,
        "threshold": selected["threshold"],
    }


if __name__ == "__main__":
    summary = pipeline(pdf, embed_model, fraction, LLM_NAME)
    print(json.dumps(summary, indent=2))
