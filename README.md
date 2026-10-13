# Lab 3: Intelligent Document Analysis (10-K climate disclosures)

Scripts 1-4 extract climate-related disclosures from 10-K filings with a local
LLM (Ollama, `phi4-mini`). This repo is for the memo extension: replacing the
keyword prefilter (`keyword_hit()`) with an embedding-similarity prefilter.

## Setup
1. Put the 10-K PDFs in `10Ks/` (not tracked by git; get them from the course Dropbox).
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Make sure Ollama is running and `ollama pull phi4-mini`.

## Run (from this folder)
- `python script1Baseline.py`
- `python script2PydanticStructured.py`
- `python script3KeywordGap.py`
- `python script4HybridRecovery.py` (needs script 3's output)

Results are written to `files/` (not tracked).
