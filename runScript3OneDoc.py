"""
MEMO EXTENSION 3 (whole file): helper to run script 3 on one 10-K only.

Runs script 3 (keyword gap) on a single 10-K instead of all of them.

Usage:  python runScript3OneDoc.py tesla10K.pdf

Does not change script 3 itself: it reuses script3KeywordGap.run() but limits the
file list to one PDF and writes the report to files/script3_gap_report_<name>.json
so running a second document doesn't overwrite the first.
"""
import sys
from pathlib import Path

import script3KeywordGap as s3

if len(sys.argv) != 2:
    raise SystemExit("Usage: python runScript3OneDoc.py <file.pdf>   e.g. tesla10K.pdf")

pdf = sys.argv[1]
s3.list_pdfs = lambda *args, **kwargs: [pdf]
s3.OUTPUT_FILE = Path(f"files/script3_gap_report_{Path(pdf).stem}.json")

s3.run()
