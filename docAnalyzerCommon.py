"""
Shared helpers for Intelligent Document Analysis lab
"""
import re
import os
# START MEMO EXTENSION 1: imports for the embedding prefilter
import json
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_ollama import OllamaEmbeddings
# END MEMO EXTENSION 1
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

PDF_DIR = "10Ks/"

# Same list as the original lab4 solution script.
KEYWORDS = [
    "climate", "emission", "greenhouse", "ghg", "scope 1", "scope 2", "scope 3",
    "weather", "natural disaster", "act of god", "acts of god", "tornado",
    "hurricane", "earthquake", "wildfire", "drought", "flood", "temperature",
    "sea level", "sea-level", "carbon", "warming", "net zero", "net-zero",
    "methane", "precipitation", "co2", "clean", "green", "earth",
]

EXPECTED_KEYS = [
    "climate_risks", "emissions", "targets_and_goals",
    "governance", "financials", "mitigation",
]

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False,
)


def list_pdfs(pdf_dir: str = PDF_DIR) -> List[str]:
    return sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))


def extract_text_from_pdf(file: str, pdf_dir: str = PDF_DIR) -> str:
    reader = PdfReader(os.path.join(pdf_dir, file))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str) -> List[str]:
    return textSplitter.split_text(text)


def keyword_hit(chunk: str) -> bool:
    return any(kw.casefold() in chunk.casefold() for kw in KEYWORDS)


def matched_keywords(chunk: str) -> List[str]:
    return [kw for kw in KEYWORDS if kw.casefold() in chunk.casefold()]


# START MEMO EXTENSION 2: embedding prefilter (counterpart of keyword_hit())
# One query per SEC category. A chunk is kept if its best cosine similarity against
# any of these clears the threshold. keyword_hit() above is left untouched so the
# two filters can be compared on the same chunks.
EMBED_QUERIES = {
    "climate_risks": "Physical and transition climate risks such as hurricanes, wildfires, "
                     "flooding, drought, or new regulation that could materially affect our "
                     "operations and financial condition",
    "emissions": "Our Scope 1 and Scope 2 greenhouse gas emissions in metric tons of CO2 equivalent",
    "targets_and_goals": "Targets to reduce emissions or achieve net zero by a specific year, "
                         "and progress toward them",
    "governance": "Board oversight and management responsibility for climate-related risks",
    "financials": "Costs, charges, or losses from severe weather events or climate-related "
                  "initiatives recorded in the financial statements",
    "mitigation": "Transition plans, scenario analysis, and investments in technology to "
                  "manage climate risk",
}

EMBED_MODEL = "embeddinggemma-2:270m"
EMBED_THRESHOLD = 0.5  # placeholder: step 5 looks at the real score distribution
EMBED_CACHE_DIR = Path("files/embedding_cache")

_embedders: Dict[str, OllamaEmbeddings] = {}
_memory_cache: Dict[str, Dict[str, List[float]]] = {}


def _cache_path(model: str) -> Path:
    return EMBED_CACHE_DIR / f"{model.replace(':', '_').replace('/', '_')}.json"


def _load_cache(model: str) -> Dict[str, List[float]]:
    if model not in _memory_cache:
        path = _cache_path(model)
        _memory_cache[model] = json.loads(path.read_text()) if path.exists() else {}
    return _memory_cache[model]


def _save_cache(model: str) -> None:
    path = _cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_memory_cache[model]))


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Returns one L2-normalised vector per text. Vectors are cached on disk per
    model, so trying a new threshold never re-embeds anything."""
    cache = _load_cache(model)
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = {k: t for k, t in zip(keys, texts) if k not in cache}

    if missing:
        if model not in _embedders:
            _embedders[model] = OllamaEmbeddings(model=model)
        vectors = _embedders[model].embed_documents(list(missing.values()))
        for k, v in zip(missing, vectors):
            cache[k] = v
        _save_cache(model)

    matrix = np.array([cache[k] for k in keys], dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def embedding_scores(chunks: List[str], model: str = EMBED_MODEL):
    """For each chunk: (best cosine similarity over EMBED_QUERIES, name of the
    best-matching category). Use this to inspect the score distribution."""
    names = list(EMBED_QUERIES)
    sims = embed_texts(chunks, model) @ embed_texts(list(EMBED_QUERIES.values()), model).T
    return sims.max(axis=1), [names[i] for i in sims.argmax(axis=1)]


def embedding_hit(chunk: str, model: str = EMBED_MODEL, threshold: float = EMBED_THRESHOLD) -> bool:
    """Drop-in counterpart of keyword_hit(): True if the chunk is similar enough to
    any climate-disclosure query. For many chunks, call embedding_scores() once
    instead of this, so the embeddings are computed in one batch."""
    scores, _ = embedding_scores([chunk], model)
    return bool(scores[0] >= threshold)
# END MEMO EXTENSION 2


class ClimateDisclosure(BaseModel):
    """Structured climate-disclosure extraction for a single document chunk."""

    climate_risks: List[str] = Field(
        default_factory=list,
        description="Material climate-related threats to operations or financials",
    )
    emissions: List[str] = Field(
        default_factory=list,
        description="GHG emissions disclosures (Scope 1/2/3), with numbers and units when present",
    )
    targets_and_goals: List[str] = Field(
        default_factory=list,
        description="Reduction targets, net-zero commitments, or milestones, with dates",
    )
    governance: List[str] = Field(
        default_factory=list,
        description="Board oversight and management responsibility for climate risk",
    )
    financials: List[str] = Field(
        default_factory=list,
        description="Capital, costs, or R&D spending tied to climate initiatives or impacts",
    )
    mitigation: List[str] = Field(
        default_factory=list,
        description="Transition plans, scenario analysis, frameworks, or technology investments",
    )

    @field_validator(
        "climate_risks", "emissions", "targets_and_goals",
        "governance", "financials", "mitigation",
        mode="after",
    )
    @classmethod
    def drop_parroted_description(cls, items: List[str], info: ValidationInfo) -> List[str]:
        """Strips any list item that's just this field's own schema
        description echoed back as if it were extracted content.
 
        with_structured_output() puts each field's `description=` into the
        JSON schema shown to the model as guidance. A model that isn't
        confident anything in the chunk belongs in a category can parrot
        that description back as a "finding" instead of correctly returning
        an empty list -- that's exactly what happened on the costco10k run
        that prompted this validator. This is cleanup, not a retry trigger:
        it filters the bad item out and otherwise accepts the object. Raising
        here instead (to force extract_with_retry() to retry) would do
        nothing useful at temperature=0 -- the model would just parrot the
        same description again on every attempt.
        """
        field_description = cls.model_fields[info.field_name].description or ""
        target = _normalize(field_description)
        return [item for item in items if _normalize(item) != target]

    def total_items(self) -> int:
        return sum(
            len(v)
            for v in (
                self.climate_risks, self.emissions, self.targets_and_goals,
                self.governance, self.financials, self.mitigation,
            )
        )

    def is_empty(self) -> bool:
        return self.total_items() == 0

def _normalize(text: str) -> str:
    """Lowercases and collapses everything but letters/digits to single
    spaces, so trivial differences in punctuation/whitespace/case don't
    defeat an exact-match comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


EXTRACTION_INSTRUCTIONS = """You are a compliance analyst extracting climate-related data per the SEC's
March 2024 climate-related disclosure rule (Release Nos. 33-11275/34-99678).

Extract only information explicitly present in the text below. If a category has no
relevant data in this text, leave it empty. Use the detailed guidance below to decide
which category (if any) a given statement belongs in — many categories are easy to
confuse with each other, so the "not this" notes matter as much as the "this" notes.

climate_risks: material climate-related risks that have materially impacted, or are
reasonably likely to materially impact, the company's strategy, results of
operations, or financial condition. Covers physical risks (extreme weather, wildfire,
drought, flooding, sea-level rise affecting facilities, operations, or suppliers) and
transition risks (new regulation, carbon pricing, shifting demand, stranded assets,
reputational exposure), over both short-term (next 12 months) and long-term horizons.
Not this: routine risk-factor boilerplate that never reaches a material climate
impact; the actions taken in response (that's mitigation) or their dollar cost
(that's financials).

governance: board-level oversight of climate risk — which board committee or
subcommittee is responsible and how it is kept informed — plus management's role:
who is responsible, their relevant expertise, and how they report up to the board.
This is about oversight structure and reporting lines specifically, not the substance
of what risks exist or what's being done about them.

mitigation: the company's processes for identifying, assessing, and managing material
climate risks, and how those processes feed into its overall risk management system —
transition plans, scenario analysis (e.g. against a 1.5°C or 2°C warming scenario),
internal carbon pricing, adoption of a framework (TCFD, SBTi, GHG Protocol), or
specific technology or operational changes. Mitigation is the process and the action,
not a figure (emissions) or a dated commitment (targets_and_goals) — if a sentence is
really just an emissions number or a target date with no action described, it belongs
in one of those categories instead.

emissions: Scope 1 and Scope 2 greenhouse gas emissions, when material, expressed in
aggregate as CO2e in gross terms (excluding any purchased or generated offsets).
Scope 3 was dropped from the final rule entirely, and this requirement only applies to
Large Accelerated and Accelerated Filers — Smaller Reporting Companies and Emerging
Growth Companies are exempt, so don't expect (or invent) emissions figures for those
companies. Not this: a vague statement about reducing emissions with no figure or
methodology attached.

targets_and_goals: climate-related targets or goals that have materially affected, or
are reasonably likely to materially affect, the business, along with progress toward
them and material expenditures incurred pursuing them. A target only counts here if
it's a specific, material commitment — a date, a percentage, or a defined benchmark.
Not this: a vague aspirational statement ("committed to sustainability") — that
doesn't meet the rule's bar and usually isn't a finding for any category.

financials: the financial-statement effects the rule requires — capitalized costs,
expensed costs, charges, and losses from severe weather events and other natural
conditions, and the financial-statement effects of material climate-related targets
or transition activities, plus any resulting impact on financial estimates and
assumptions, all subject to the rule's materiality thresholds. This is strictly about
dollar impacts that flow through the financial statements, not a general cost or risk
mentioned elsewhere in the narrative disclosure.

Materiality is the operative threshold throughout the rule — if a statement wouldn't
clear a materiality bar on its own, it likely doesn't belong in any of these six
categories. If a single statement could reasonably fit more than one category, put it
in the most specific one that applies (a $50M investment in carbon capture technology
is mitigation, not financials, unless the sentence is really about the cost itself).

Never restate any of the guidance above as if it were a finding, in whole or in part —
not your own category's wording, and not another category's. Only include text that
actually appears (verbatim or closely paraphrased) in the chunk below. If nothing in
the text supports a category, leave it empty — an empty list is a correct, expected
answer, not a failure.

Text:
{text}
"""