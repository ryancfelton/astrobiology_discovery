import hashlib
import html
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import quote, urlsplit, urlunsplit

import arxiv
import numpy as np
import requests
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Astrobiology Discovery Explorer",
    page_icon="🪐",
    layout="wide",
)

MODEL_NAME = "nasa-impact/indus-sde-st-v0.2"
MODEL_URL = "https://huggingface.co/nasa-impact/indus-sde-st-v0.2"
PDS_API_BASE = "https://pds.nasa.gov/api/search/1"
PDS_ARCHIVE_SEARCH_ENDPOINT = "https://pds.nasa.gov/services/search/search"

# Temporary feature flag. Keep all PDS retrieval/parsing code in place, but
# disable the feature while Streamlit Community Cloud resource usage is being
# evaluated. Set this back to True to restore PDS search.
PDS_ENABLED = True

CURRENT_YEAR = datetime.now(timezone.utc).year


# ============================================================
# OPTIONAL ASTROBIOLOGY LENS
# ============================================================

# This is an experimental ranking layer created for this prototype.
# It does not modify or retrain INDUS and is OFF by default.
ASTRO_LENS_WEIGHT = 0.25

ASTROBIOLOGY_LENSES: Dict[str, str] = {
    "Habitability": (
        "planetary habitability, environments capable of supporting life, "
        "liquid water, energy sources, nutrients, environmental limits, "
        "and habitable planetary conditions"
    ),
    "Biosignatures": (
        "biosignatures, life detection, biological gases, atmospheric "
        "disequilibrium, surface biosignatures, false positives, "
        "false negatives, and signs of life"
    ),
    "Origins of Life": (
        "origins of life, prebiotic chemistry, chemical evolution, "
        "abiogenesis, organic chemistry, emergence of metabolism, "
        "and early biochemical systems"
    ),
    "Ocean Worlds": (
        "ocean worlds, Europa, Enceladus, subsurface oceans, "
        "hydrothermal systems, ice-ocean interfaces, water-rock "
        "interaction, and icy moon habitability"
    ),
    "Mars & Ancient Environments": (
        "Mars habitability, ancient aqueous environments, sedimentary "
        "environments, organics, life detection on Mars, and preservation "
        "of biosignatures"
    ),
    "Exoplanets": (
        "exoplanet habitability, terrestrial exoplanets, atmospheric "
        "characterization, biosignature gases, stellar environments, "
        "and remotely detectable signs of life"
    ),
    "Planetary Chemistry": (
        "planetary geochemistry, atmospheric chemistry, photochemistry, "
        "serpentinization, redox disequilibrium, methane chemistry, "
        "and water-rock reactions"
    ),
    "Life in Extremes": (
        "extremophiles, microbial ecology, environmental limits of life, "
        "analog environments, deep biosphere, hydrothermal life, "
        "and microbial metabolism"
    ),
    "Titan & Organic Worlds": (
        "Titan, organic-rich planetary environments, atmospheric haze, "
        "hydrocarbon chemistry, prebiotic organic chemistry, and "
        "habitability beyond liquid-water surface environments"
    ),
}


# ============================================================
# SEARCH HELPERS
# ============================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "about", "can", "could",
    "do", "does", "for", "from", "how", "in", "into", "is", "it", "of",
    "on", "or", "that", "the", "their", "these", "this", "to", "what",
    "when", "where", "which", "with", "would", "relevant", "using", "use",
    "find", "show", "tell", "study", "studies", "research", "data", "dataset",
    "datasets", "material", "science", "scientific",
}

# PDS supports structured target searches with target_name. These aliases let
# the app recognize common astrobiology targets in ordinary-language queries.
PDS_TARGET_ALIASES = {
    "titan": "Titan",
    "europa": "Europa",
    "enceladus": "Enceladus",
    "mars": "Mars",
    "venus": "Venus",
    "earth": "Earth",
    "moon": "Moon",
    "luna": "Moon",
    "ganymede": "Ganymede",
    "callisto": "Callisto",
    "ceres": "Ceres",
    "pluto": "Pluto",
    "triton": "Triton",
    "saturn": "Saturn",
    "jupiter": "Jupiter",
    "mercury": "Mercury",
    "asteroid bennu": "Bennu",
    "bennu": "Bennu",
    "asteroid ryugu": "Ryugu",
    "ryugu": "Ryugu",
}

PDS_DATA_CLASSES = ["bundles", "collections"]
PDS_DOCUMENT_CLASSES = ["documents"]

# The PDS Collection class includes many infrastructure/support collections
# (Context, Document, XML Schema, Miscellaneous, etc.) that are not useful
# as scientific data-discovery results.  Keep only data-like collection types
# behind the "PDS datasets / collections" checkbox.
PDS_SCIENCE_COLLECTION_TYPES = {
    "data",
    "calibration",
    "geometry",
    "spice kernel",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def informative_terms(query: str, max_terms: int = 10) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-\.]*", query.lower())
    terms = []
    for token in tokens:
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def detect_pds_targets(query: str) -> List[str]:
    lower = query.lower()
    found = []
    # Longest aliases first so multiword aliases win before their pieces.
    for alias, canonical in sorted(PDS_TARGET_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            if canonical not in found:
                found.append(canonical)
    return found


def pds_concept_terms(query: str, targets: List[str], max_terms: int = 5) -> List[str]:
    terms = informative_terms(query, max_terms=12)
    target_words = set()
    for target in targets:
        target_words.update(target.lower().split())
    filtered = [term for term in terms if term not in target_words]
    return filtered[:max_terms]


def deduplicate_resources(resources: List[dict]) -> List[dict]:
    seen = set()
    unique = []

    for resource in resources:
        source = resource.get("source", "")
        record_id = clean_text(str(resource.get("record_id", ""))).lower()
        doi = clean_text(str(resource.get("doi", ""))).lower()
        title_key = re.sub(r"[^a-z0-9]", "", resource.get("title", "").lower())[:180]

        if record_id:
            key = f"{source}|id|{record_id}"
        elif doi and doi != "n/a":
            key = f"doi|{doi}"
        else:
            key = f"{source}|title|{title_key}"

        if not title_key or key in seen:
            continue

        seen.add(key)
        unique.append(resource)

    return unique


def resource_sort_key(resource: dict):
    """Stable ordering so equivalent candidate sets produce identical cache keys."""
    return (
        clean_text(str(resource.get("source", ""))).lower(),
        clean_text(str(resource.get("record_id", ""))).lower(),
        clean_text(str(resource.get("doi", ""))).lower(),
        clean_text(str(resource.get("title", ""))).lower(),
        int(resource.get("year") or 0),
    )


def resource_text(resource: dict) -> str:
    title = resource.get("title", "")
    description = resource.get("abstract", "")
    context = resource.get("metadata_context", "")
    return f"{title}. {description}. {context}".strip()


def resource_embedding_key(resource: dict, text: str) -> str:
    """
    Stable cache key for an INDUS document embedding.

    Including both the record identity and actual text means an updated
    abstract/metadata record automatically gets a fresh embedding.
    """
    identity = "|".join(
        [
            MODEL_NAME,
            clean_text(str(resource.get("source", ""))),
            clean_text(str(resource.get("record_id", ""))),
            clean_text(str(resource.get("doi", ""))),
            text,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def strength_label(score: float, values: np.ndarray) -> str:
    if len(values) < 3:
        return "Relevant"
    p70 = float(np.percentile(values, 70))
    p40 = float(np.percentile(values, 40))
    if score >= p70:
        return "Strong"
    if score >= p40:
        return "Moderate"
    return "Possible"


# ============================================================
# PDS VALUE / METADATA HELPERS
# ============================================================


def humanize_pds_identifier(value: str) -> str:
    text = clean_text(str(value))
    if not text.startswith("urn:nasa:pds:context:"):
        return text
    tail = text.rsplit(":", 1)[-1]
    if "." in tail:
        tail = tail.split(".", 1)[-1]
    return tail.replace("_", " ").replace("-", " ").strip().title()


def flatten_values(value) -> List[str]:
    results = []
    if value is None:
        return results

    if isinstance(value, list):
        for item in value:
            results.extend(flatten_values(item))
        return list(dict.fromkeys([x for x in results if x]))

    if isinstance(value, dict):
        for key in ["name", "title", "value"]:
            if key in value:
                nested = flatten_values(value.get(key))
                if nested:
                    results.extend(nested)
        if not results and "id" in value:
            results.append(humanize_pds_identifier(value["id"]))
        return list(dict.fromkeys([x for x in results if x]))

    text = humanize_pds_identifier(str(value))
    return [text] if text else []


def pds_sources(item: dict) -> List[dict]:
    sources = []
    properties = item.get("properties", {})
    metadata = item.get("metadata", {})
    if isinstance(properties, dict):
        sources.append(properties)
    sources.append(item)
    if isinstance(metadata, dict):
        sources.append(metadata)
    return sources


def pds_values(item: dict, keys: List[str]) -> List[str]:
    results = []
    for source in pds_sources(item):
        for key in keys:
            if key in source:
                results.extend(flatten_values(source.get(key)))

    cleaned = []
    for result in results:
        result = clean_text(result)
        if result and result not in cleaned:
            cleaned.append(result)
    return cleaned


def pds_first(item: dict, keys: List[str]) -> str:
    values = pds_values(item, keys)
    return values[0] if values else ""


def is_useless_pds_description(text: str) -> bool:
    lower = text.lower().strip()
    useless_phrases = {
        "migration to pds4",
        "initial version",
        "initial release",
        "updated version",
        "reformatted for pds4",
    }
    return lower in useless_phrases or len(lower) < 15


def get_pds_description(item: dict) -> str:
    priority_groups = [
        ["pds:Citation_Information.pds:description", "citation_description"],
        ["pds:Document.pds:description"],
        ["abstract_text", "data_set_description"],
        ["bundle_description", "collection_description"],
        ["description"],
    ]

    for keys in priority_groups:
        values = pds_values(item, keys)
        good_values = [value for value in values if not is_useless_pds_description(value)]
        if good_values:
            # Prefer a substantive description over a tiny label-like string.
            good_values.sort(key=len, reverse=True)
            return good_values[0]

    return ""


def pds_human_page(product_class: str, lid: str, version: str) -> str:
    if not lid:
        return ""

    identifier = quote(lid, safe="")
    version_piece = f"&version={quote(version, safe='')}" if version else ""
    base = "https://pds.nasa.gov/ds-view/pds/"

    if product_class == "Product_Bundle":
        return base + "viewBundle.jsp?identifier=" + identifier + version_piece
    if product_class == "Product_Collection":
        return base + "viewCollection.jsp?identifier=" + identifier + version_piece
    if product_class == "Product_Document":
        return base + "viewDocument.jsp?identifier=" + identifier + version_piece

    return "https://pds.nasa.gov/services/search/search?q=" + identifier


def pds_result_page(item: dict, fallback_url: str) -> str:
    """Prefer the human-facing result URL returned by the PDS archive search."""
    result_url = pds_first(item, ["resLocation", "res_location"])
    if result_url:
        if result_url.startswith(("http://", "https://")):
            return result_url
        if result_url.startswith("/"):
            return "https://pds.nasa.gov" + result_url
    return fallback_url


def get_pds_file_urls(item: dict) -> List[str]:
    urls = pds_values(
        item,
        [
            "ops:Data_File_Info.ops:file_ref",
            "file_ref_url",
        ],
    )
    return [url for url in urls if url.startswith(("http://", "https://"))]


def get_pds_label_url(item: dict) -> str:
    urls = pds_values(
        item,
        [
            "ops:Label_File_Info.ops:file_ref",
            "label_url",
        ],
    )
    for url in urls:
        if url.startswith(("http://", "https://")):
            return url
    return ""


def preferred_direct_file(file_urls: List[str]) -> str:
    if not file_urls:
        return ""
    for url in file_urls:
        if url.lower().endswith(".pdf"):
            return url
    for url in file_urls:
        if not url.lower().endswith(".xml"):
            return url
    return file_urls[0]


def pds_parent_directory_url(url: str) -> str:
    """
    Convert a concrete PDS file URL into the directory that contains it.

    Collection inventory tables are metadata that enumerate collection
    members. They are not themselves the scientific data resource a user
    expects to open. For aggregate PDS products, browsing the containing
    archive directory is a much more useful access point.
    """
    if not url or not url.startswith(("http://", "https://")):
        return ""

    parts = urlsplit(url)
    path = parts.path or ""

    if path.endswith("/"):
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    if "/" not in path:
        return ""

    directory = path.rsplit("/", 1)[0] + "/"
    return urlunsplit((parts.scheme, parts.netloc, directory, "", ""))


def pds_archive_access_url(
    product_class: str,
    file_urls: List[str],
    label_url: str,
) -> str:
    """
    Choose the user-facing archive access link for a PDS result.

    Documents may sensibly open a PDF directly. Bundles, collections, and data
    sets should instead open an archive directory rather than a collection
    inventory/member-list file.
    """
    if product_class == "Product_Document":
        return preferred_direct_file(file_urls)

    aggregate_classes = {
        "Product_Bundle",
        "Product_Collection",
        "Product_Data_Set",
        "Product_Data_Set_PDS3",
    }

    if product_class in aggregate_classes:
        if label_url:
            directory = pds_parent_directory_url(label_url)
            if directory:
                return directory

        for url in file_urls:
            directory = pds_parent_directory_url(url)
            if directory:
                return directory

    return preferred_direct_file(file_urls)


def pds_is_inventory_or_manifest_url(url: str) -> bool:
    filename = urlsplit(url).path.rsplit("/", 1)[-1].lower()
    return any(
        token in filename
        for token in (
            "inventory",
            "manifest",
            "checksum",
            "member",
        )
    )


def get_pds_collection_type(item: dict) -> str:
    return pds_first(item, ["collection_type", "pds:Collection.pds:collection_type"])


def pds_is_context_registry_product(item: dict) -> bool:
    """
    Reject PDS context-registry/infrastructure records before INDUS ranking.

    The federated PDS index contains collections whose job is to catalogue
    missions, instruments, agencies, targets, etc.  They can match words such
    as Titan or Cassini but they are metadata infrastructure, not the science
    resources this app is trying to surface.
    """
    collection_type = get_pds_collection_type(item).strip().lower()
    if collection_type == "context":
        return True

    identifiers = pds_values(
        item,
        [
            "lid",
            "lidvid",
            "id",
            "pds:Identification_Area.pds:logical_identifier",
        ],
    )

    for identifier in identifiers:
        lower_id = identifier.lower()
        # Covers NASA and partner-agency identifiers such as
        # urn:nasa:pds:context:... and urn:isro:isda:context:...
        if re.search(r":context(?::|$)", lower_id):
            return True

    title = pds_first(
        item,
        ["title", "pds:Identification_Area.pds:title"],
    )
    description = get_pds_description(item)
    combined = f"{title} {description}".lower()

    if "context products" in combined:
        return True
    if "bundle context" in combined:
        return True

    return False


def pds_resource_type(product_class: str, collection_type: str) -> str:
    if product_class == "Product_Document":
        return "PDS Documentation"
    if product_class == "Product_Bundle":
        return "PDS Bundle"
    if product_class == "Product_Collection":
        if collection_type and collection_type.lower() == "document":
            return "PDS Documentation Collection"
        if collection_type and collection_type.lower() == "data":
            return "PDS Dataset / Collection"
        return "PDS Collection"
    if product_class in {"Product_Data_Set", "Product_Data_Set_PDS3"}:
        return "PDS Dataset"
    if product_class == "Product_Resource":
        return "PDS Scientific Resource"
    return "PDS Resource"


# ============================================================
# INDUS MODEL
# ============================================================


@st.cache_resource(show_spinner=False)
def load_indus_model():
    # Explicitly keep CPU inference in FP32. Newer Transformers versions may
    # honor the checkpoint's configured BF16 dtype automatically, which can be
    # extremely slow on CPUs without fast native BF16 support.
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model = model.float()
    return model


def indus_model_dtype(model) -> str:
    try:
        return str(next(model.parameters()).dtype)
    except (StopIteration, AttributeError):
        return "unknown"


@st.cache_resource(show_spinner=False)
def resource_embedding_cache():
    # Shared within the life of this Streamlit process. This avoids recomputing
    # INDUS embeddings for the same papers/resources across related searches.
    return {}


def trim_resource_embedding_cache(cache: dict, max_entries: int = 5000):
    while len(cache) > max_entries:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


@st.cache_resource(show_spinner=False)
def lens_embeddings():
    model = load_indus_model()
    return model.encode(
        list(ASTROBIOLOGY_LENSES.values()),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def rank_with_indus(
    resources: List[dict],
    query: str,
    use_astro_lens: bool = False,
):
    if not resources:
        return [], {}

    diagnostics = {}

    model_load_started = time.perf_counter()
    model = load_indus_model()
    diagnostics["model_load_seconds"] = time.perf_counter() - model_load_started
    diagnostics["model_dtype"] = indus_model_dtype(model)

    texts = [resource_text(resource) for resource in resources]
    embedding_keys = [
        resource_embedding_key(resource, text)
        for resource, text in zip(resources, texts)
    ]

    query_started = time.perf_counter()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    diagnostics["query_embedding_seconds"] = time.perf_counter() - query_started

    docs_started = time.perf_counter()
    embedding_cache = resource_embedding_cache()

    missing_indices = [
        index
        for index, key in enumerate(embedding_keys)
        if key not in embedding_cache
    ]

    if missing_indices:
        missing_texts = [texts[index] for index in missing_indices]
        missing_embeddings = model.encode(
            missing_texts,
            batch_size=12,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for index, embedding in zip(missing_indices, missing_embeddings):
            embedding_cache[embedding_keys[index]] = embedding

        trim_resource_embedding_cache(embedding_cache)

    doc_embeddings = np.stack(
        [embedding_cache[key] for key in embedding_keys],
        axis=0,
    )

    diagnostics["document_embedding_seconds"] = time.perf_counter() - docs_started
    diagnostics["document_count"] = len(texts)
    diagnostics["document_cache_hits"] = len(texts) - len(missing_indices)
    diagnostics["document_cache_misses"] = len(missing_indices)
    diagnostics["document_cache_size"] = len(embedding_cache)

    scoring_started = time.perf_counter()
    query_scores = doc_embeddings @ query_embedding
    diagnostics["cosine_scoring_seconds"] = time.perf_counter() - scoring_started

    if not use_astro_lens:
        ranked = []
        for idx, resource in enumerate(resources):
            item = dict(resource)
            item["query_score"] = float(query_scores[idx])
            item["astro_score"] = None
            item["combined_score"] = float(query_scores[idx])
            item["astro_tags"] = []
            ranked.append(item)
        ranked.sort(key=lambda r: r["combined_score"], reverse=True)
        return ranked, diagnostics

    lens_started = time.perf_counter()
    lens_emb = lens_embeddings()
    all_lens_scores = doc_embeddings @ lens_emb.T
    diagnostics["astro_lens_seconds"] = time.perf_counter() - lens_started

    astro_scores = np.max(all_lens_scores, axis=1)
    combined_scores = (
        (1.0 - ASTRO_LENS_WEIGHT) * query_scores
        + ASTRO_LENS_WEIGHT * astro_scores
    )

    lens_names = list(ASTROBIOLOGY_LENSES.keys())
    ranked = []
    for idx, resource in enumerate(resources):
        lens_order = np.argsort(all_lens_scores[idx])[::-1][:2]
        item = dict(resource)
        item["query_score"] = float(query_scores[idx])
        item["astro_score"] = float(astro_scores[idx])
        item["combined_score"] = float(combined_scores[idx])
        item["astro_tags"] = [lens_names[j] for j in lens_order]
        ranked.append(item)

    ranked.sort(key=lambda r: r["combined_score"], reverse=True)
    return ranked, diagnostics


# ============================================================
# ADS / SCIX
# ============================================================


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def fetch_ads(
    query: str,
    api_key: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:
    if not api_key:
        return []

    endpoint = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "q": query,
        "fq": f"year:[{start_year} TO {end_year}]",
        "fl": "title,author,year,pubdate,abstract,identifier,doi,bibcode,keyword",
        "rows": rows,
        "sort": "score desc",
    }

    response = requests.get(endpoint, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    docs = response.json().get("response", {}).get("docs", [])

    resources = []
    for doc in docs:
        title = doc.get("title", [""])
        title = title[0] if isinstance(title, list) and title else str(title)
        abstract = clean_text(doc.get("abstract", ""))
        if not abstract:
            continue

        bibcode = doc.get("bibcode", "")
        doi_value = doc.get("doi", [""])
        doi = doi_value[0] if isinstance(doi_value, list) and doi_value else str(doi_value or "")
        identifiers = doc.get("identifier", []) or []
        arxiv_id = ""
        for ident in identifiers:
            if str(ident).lower().startswith("arxiv:"):
                arxiv_id = str(ident).split(":", 1)[1]
                break

        resources.append(
            {
                "title": clean_text(title),
                "authors": doc.get("author", []) or [],
                "year": int(doc.get("year") or 0),
                "abstract": abstract,
                "metadata_context": "",
                "source": "ADS",
                "resource_type": "Publication",
                "record_id": bibcode,
                "doi": doi,
                "url": f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract" if bibcode else "",
                "secondary_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            }
        )

    return resources


# ============================================================
# ARXIV
# ============================================================


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def fetch_arxiv(
    query: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:
    terms = informative_terms(query)
    if not terms:
        terms = [query.strip()]

    term_query = " OR ".join(f'all:"{term}"' for term in terms)
    date_query = f"submittedDate:[{start_year}01010000 TO {end_year}12312359]"
    final_query = f"({term_query}) AND {date_query}"

    client = arxiv.Client(
        page_size=min(rows, 100),
        delay_seconds=3.0,
        num_retries=2,
    )

    search = arxiv.Search(
        query=final_query,
        max_results=rows,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    resources = []
    for result in client.results(search):
        abstract = clean_text(result.summary)
        if not abstract:
            continue

        resources.append(
            {
                "title": clean_text(result.title),
                "authors": [author.name for author in result.authors],
                "year": result.published.year,
                "abstract": abstract,
                "metadata_context": "",
                "source": "arXiv",
                "resource_type": "Preprint",
                "record_id": result.entry_id,
                "doi": result.doi or "",
                "url": result.entry_id,
                "secondary_url": result.pdf_url or "",
            }
        )

    return resources


# ============================================================
# NASA NTRS
# ============================================================


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def fetch_ntrs(
    query: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:
    endpoint = "https://ntrs.nasa.gov/api/citations/search"
    params = {
        "q": query,
        "publicationDateFrom": f"{start_year}-01-01",
        "publicationDateTo": f"{end_year}-12-31",
        "page": 1,
        "pageSize": rows,
    }

    response = requests.get(endpoint, params=params, timeout=30)
    response.raise_for_status()
    docs = response.json().get("results", [])

    resources = []
    for doc in docs[:rows]:
        abstract = clean_text(doc.get("abstract", ""))
        if not abstract:
            continue

        authors = []
        for auth_item in doc.get("authorAffiliations", []) or []:
            if isinstance(auth_item, dict):
                meta = auth_item.get("meta", {}) or {}
                author_obj = meta.get("author", {}) or {}
                name = author_obj.get("name") or auth_item.get("name")
                if name:
                    authors.append(name)

        if not authors:
            for author in doc.get("authors", []) or []:
                if isinstance(author, dict) and author.get("name"):
                    authors.append(author["name"])
                elif isinstance(author, str):
                    authors.append(author)

        ntrs_id = str(doc.get("id", ""))
        pub_date = str(doc.get("publicationDate") or doc.get("issued") or doc.get("created") or "")
        year_match = re.match(r"(\d{4})", pub_date)
        year = int(year_match.group(1)) if year_match else 0

        resources.append(
            {
                "title": clean_text(doc.get("title", "")),
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "metadata_context": "",
                "source": "NASA NTRS",
                "resource_type": "Technical Report / NASA Document",
                "record_id": ntrs_id,
                "doi": clean_text(str(doc.get("doi", ""))),
                "url": f"https://ntrs.nasa.gov/citations/{ntrs_id}" if ntrs_id else "",
                "secondary_url": "",
            }
        )

    return resources


# ============================================================
# NASA PDS PARSER
# ============================================================


def parse_pds_record(item: dict, allow_documentation: bool = False):
    product_class = clean_text(str(item.get("type", "")))
    if not product_class:
        product_class = pds_first(
            item,
            ["product_class", "pds:Identification_Area.pds:product_class"],
        )

    # Context registry records are metadata infrastructure, not science results.
    if pds_is_context_registry_product(item):
        return None

    collection_type = get_pds_collection_type(item)
    collection_type_lower = collection_type.strip().lower()

    is_doc = (
        product_class == "Product_Document"
        or (product_class == "Product_Collection" and collection_type_lower == "document")
    )

    if is_doc and not allow_documentation:
        return None

    if product_class == "Product_Collection" and not is_doc:
        # Do not surface Context, Miscellaneous, XML Schema, External, Browse,
        # or other infrastructure/support collections as scientific datasets.
        if collection_type_lower not in PDS_SCIENCE_COLLECTION_TYPES:
            return None

    if product_class not in {
        "Product_Bundle",
        "Product_Collection",
        "Product_Document",
        "Product_Data_Set",
        "Product_Data_Set_PDS3",
    }:
        return None

    lid = pds_first(
        item,
        ["lid", "pds:Identification_Area.pds:logical_identifier"],
    )
    lidvid = pds_first(item, ["lidvid", "id"])
    version = pds_first(
        item,
        ["vid", "pds:Identification_Area.pds:version_id", "version"],
    )

    if not version and "::" in lidvid:
        version = lidvid.rsplit("::", 1)[1]
    if not lid and lidvid:
        lid = lidvid.split("::", 1)[0]

    record_id = lidvid or lid
    title = pds_first(
        item,
        [
            "title",
            "pds:Identification_Area.pds:title",
            "pds:Document.pds:document_name",
        ],
    )
    if not title:
        return None

    description = get_pds_description(item)
    investigations = pds_values(
        item,
        ["pds:Investigation_Area.pds:name", "investigation_name", "investigations"],
    )
    targets = pds_values(
        item,
        ["pds:Target_Identification.pds:name", "target_name", "targets"],
    )
    instruments = pds_values(
        item,
        [
            "pds:Observing_System_Component.pds:name",
            "observing_system_component_name",
            "instrument_name",
            "observing_system_components",
        ],
    )
    authors = pds_values(
        item,
        [
            "pds:Citation_Information.pds:author_list",
            "pds:Document.pds:author_list",
            "citation_author_list",
            "full_name",
        ],
    )

    year_text = pds_first(
        item,
        [
            "pds:Citation_Information.pds:publication_year",
            "pds:Document.pds:publication_date",
            "citation_publication_year",
            "publication_year",
            "data_set_release_date",
        ],
    )
    year_match = re.search(r"\b(?:19|20)\d{2}\b", year_text)
    year = int(year_match.group(0)) if year_match else 0

    doi = pds_first(
        item,
        [
            "pds:Citation_Information.pds:doi",
            "pds:Document.pds:doi",
            "citation_doi",
            "doi",
        ],
    )

    file_urls = get_pds_file_urls(item)
    label_url = get_pds_label_url(item)
    access_url = pds_archive_access_url(
        product_class,
        file_urls,
        label_url,
    )
    human_url = pds_result_page(
        item,
        pds_human_page(product_class, lid, version),
    )

    context_parts = []
    if targets:
        context_parts.append("Targets: " + ", ".join(targets[:8]))
    if investigations:
        context_parts.append("Investigations: " + ", ".join(investigations[:8]))
    if instruments:
        context_parts.append("Observing systems: " + ", ".join(instruments[:8]))

    metadata_context = ". ".join(context_parts)
    if not description:
        description = metadata_context
    if not description:
        return None

    extra_file_urls = [
        url
        for url in file_urls
        if url != access_url and not pds_is_inventory_or_manifest_url(url)
    ][:3]

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": description,
        "metadata_context": metadata_context,
        "source": "NASA PDS",
        "resource_type": pds_resource_type(product_class, collection_type),
        "record_id": record_id,
        "doi": doi,
        "url": human_url,
        "secondary_url": access_url,
        "label_url": label_url,
        "extra_file_urls": extra_file_urls,
        "pds_product_class": product_class,
        "pds_collection_type": collection_type,
        "pds_targets": targets,
        "pds_investigations": investigations,
        "pds_instruments": instruments,
    }


# ============================================================
# NASA PDS STRUCTURED RETRIEVAL
# ============================================================


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def pds_search_class(
    class_name: str,
    *,
    q: str = "",
    keywords: str = "",
    limit: int = 100,
) -> List[dict]:
    """
    Search one PDS high-level class endpoint.

    We use this for bundles and collections. Query strings are wrapped in
    parentheses because the current PDS query parser is most reliable with the
    documented grouped syntax.
    """
    endpoint = f"{PDS_API_BASE}/classes/{class_name}"
    params = {"limit": min(max(limit, 1), 1000)}

    if q:
        q = q.strip()
        if not (q.startswith("(") and q.endswith(")")):
            q = f"({q})"
        params["q"] = q

    if keywords:
        params["keywords"] = keywords

    response = requests.get(
        endpoint,
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def pds_search_products(
    *,
    q: str = "",
    keywords: str = "",
    limit: int = 500,
) -> List[dict]:
    """
    Search the general PDS /products endpoint.

    This endpoint is our compatibility/fallback path and is also used for
    Product_Document records because the dedicated documents class endpoint
    has not behaved consistently in live deployment.
    """
    endpoint = f"{PDS_API_BASE}/products"
    params = {"limit": min(max(limit, 1), 1000)}

    if q:
        q = q.strip()
        if not (q.startswith("(") and q.endswith(")")):
            q = f"({q})"
        params["q"] = q

    if keywords:
        params["keywords"] = keywords

    response = requests.get(
        endpoint,
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def raw_pds_product_class(item: dict) -> str:
    """Read a product class cheaply before doing the full PDS parse."""
    top_type = clean_text(str(item.get("type", "")))
    if top_type:
        return top_type

    return pds_first(
        item,
        [
            "product_class",
            "pds:Identification_Area.pds:product_class",
        ],
    )


def pds_selected_raw_item(
    item: dict,
    *,
    include_data: bool,
    include_documents: bool,
) -> bool:
    """Keep only user-facing scientific PDS resources requested by the user."""
    # The PDS index also contains context registries (missions, instruments,
    # agencies, targets, etc.).  Those can match a planetary query but are not
    # scientific datasets/resources, so reject them before INDUS sees them.
    if pds_is_context_registry_product(item):
        return False

    product_class = raw_pds_product_class(item)
    collection_type = get_pds_collection_type(item).strip().lower()

    if product_class == "Product_Document":
        return include_documents

    if product_class == "Product_Collection":
        if collection_type == "document":
            return include_documents

        if not include_data:
            return False

        # Product_Collection is broader than "dataset": PDS permits Context,
        # Miscellaneous, XML Schema, External, Browse, etc.  Only keep data-like
        # scientific collection types for the datasets/collections checkbox.
        return collection_type in PDS_SCIENCE_COLLECTION_TYPES

    if product_class in {
        "Product_Bundle",
        "Product_Data_Set",
        "Product_Data_Set_PDS3",
    }:
        return include_data

    return False


def pds_document_query(target: str = "") -> str:
    class_clause = (
        'pds:Identification_Area.pds:product_class eq "Product_Document"'
    )

    if target:
        return (
            f'(({class_clause}) and '
            f'(target_name eq "{target}"))'
        )

    return f"({class_clause})"


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def pds_archive_search(
    query: str,
    *,
    rows: int = 250,
    start: int = 0,
) -> List[dict]:
    """
    Search the PDS archive's higher-level Data Set Keyword Search service.

    This service exposes the archive records scientists actually browse and
    supports fields such as target:, product-class:, investigation:, and
    instrument:.  It has proven much more reliable for bundle/collection/data
    set discovery than the beta Search API endpoints used in earlier versions
    of this prototype.
    """
    params = {
        "q": query,
        "rows": min(max(rows, 1), 500),
        "start": max(start, 0),
        # Force the search service to return its Solr-style JSON payload.
        "wt": "json",
    }

    response = requests.get(
        PDS_ARCHIVE_SEARCH_ENDPOINT,
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    docs = payload.get("response", {}).get("docs", [])
    return docs if isinstance(docs, list) else []


def pds_item_matches_targets(item: dict, targets: List[str]) -> bool:
    """
    For a recognized target such as Titan, keep records whose structured
    target metadata contains that body. If target metadata is absent, allow a
    title/description match as a fallback.
    """
    if not targets:
        return True

    wanted = {target.strip().lower() for target in targets if target.strip()}
    item_targets = {
        value.strip().lower()
        for value in pds_values(item, ["target_name", "pds:Target_Identification.pds:name"])
        if value.strip()
    }

    if item_targets:
        return bool(wanted & item_targets)

    fallback_text = " ".join(
        [
            pds_first(item, ["title", "pds:Identification_Area.pds:title"]),
            get_pds_description(item),
        ]
    ).lower()

    return any(re.search(rf"\b{re.escape(target.lower())}\b", fallback_text) for target in targets)


def pds_prefilter_score(
    resource: dict,
    query: str,
    targets: List[str],
    concepts: List[str],
) -> float:
    """
    Cheap lexical/metadata score used only to shrink a large PDS candidate pool
    before the CPU-expensive INDUS embedding step.

    This is NOT the final relevance score. INDUS performs the semantic ranking
    after this prefilter. The goal is simply to avoid embedding hundreds of
    obviously weaker PDS candidates on Streamlit Community Cloud.
    """
    title = clean_text(resource.get("title", "")).lower()
    description = clean_text(resource.get("abstract", "")).lower()
    context = clean_text(resource.get("metadata_context", "")).lower()

    target_metadata = {
        clean_text(value).lower()
        for value in resource.get("pds_targets", [])
        if clean_text(value)
    }

    score = 0.0

    # Structured target metadata is the strongest cheap signal.
    for target in targets:
        target_lower = target.lower()
        if target_lower in target_metadata:
            score += 8.0
        if target_lower in title:
            score += 4.0
        elif target_lower in description:
            score += 2.0

    # Science concepts such as methane should strongly favor resources whose
    # titles/descriptions actually discuss the concept.
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in title:
            score += 6.0
        if concept_lower in description:
            score += 3.0
        if concept_lower in context:
            score += 1.0

    # Include remaining informative query terms as weaker tie-breakers.
    for term in informative_terms(query, max_terms=12):
        term_lower = term.lower()
        if any(term_lower == concept.lower() for concept in concepts):
            continue
        if any(term_lower in target.lower().split() for target in targets):
            continue
        if term_lower in title:
            score += 2.0
        elif term_lower in description:
            score += 1.0

    # Small preference for records with substantive descriptions and useful
    # landing/direct-resource links.
    if len(description) >= 120:
        score += 0.5
    if resource.get("url"):
        score += 0.25
    if resource.get("secondary_url"):
        score += 0.25

    return score


def fetch_pds(
    query: str,
    rows: int,
    include_data: bool,
    include_documents: bool,
) -> List[dict]:
    """
    Build a high-level PDS candidate pool from the public PDS archive keyword
    search, then let INDUS-SDE-ST perform semantic reranking.

    Important design choice:
    ------------------------
    The PDS archive search has useful structured metadata on each result, but
    its fielded query syntax has proven brittle in this prototype.  Therefore
    we deliberately use broad lexical retrieval (for example ``Titan`` and
    ``methane Titan``), then filter the returned records locally using
    ``target_name``, ``product_class``, and ``collection_type``.

    For ``Tell me about methane on Titan`` this means:
      * retrieve a broad Titan pool;
      * retrieve methane + Titan and methane pools;
      * keep genuine Titan bundles/data collections/PDS3 datasets (plus
        documentation if requested);
      * remove Context/XML-schema/support collections;
      * let INDUS rank the surviving scientific resources semantically.
    """
    if not include_data and not include_documents:
        return []

    targets = detect_pds_targets(query)
    concepts = pds_concept_terms(query, targets, max_terms=5)

    # Build a small, prioritized set of lexical searches and stop as soon as
    # enough plausible PDS candidates have been found. This avoids repeatedly
    # hitting the archive for broad fallback searches when the first query was
    # already sufficient.
    search_terms = []

    if targets:
        for target in targets:
            if concepts:
                search_terms.append(f"{concepts[0]} {target}")
            search_terms.append(target)
        if concepts:
            search_terms.append(concepts[0])
    else:
        if query.strip():
            search_terms.append(query.strip())
        search_terms.extend(informative_terms(query, max_terms=2))

    search_terms = list(dict.fromkeys(term for term in search_terms if term))[:3]

    raw_api_hits = 0
    retained_raw = []
    seen_raw = set()
    errors = []
    successful_calls = 0
    attempted_terms = []

    raw_rows_per_query = min(150, max(80, rows * 3))
    target_pool_size = max(60, rows * 2)

    for search_text in search_terms:
        attempted_terms.append(search_text)
        try:
            # PDS is intentionally over-fetched relative to the UI search depth,
            # but the pool is cheaply filtered before anything is sent to INDUS.
            items = pds_archive_search(search_text, rows=raw_rows_per_query)
            successful_calls += 1
            raw_api_hits += len(items)
        except Exception as exc:
            errors.append(f"{search_text}: {exc}")
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            if not pds_item_matches_targets(item, targets):
                continue

            if not pds_selected_raw_item(
                item,
                include_data=include_data,
                include_documents=include_documents,
            ):
                continue

            raw_id = clean_text(
                str(
                    item.get("lidvid")
                    or item.get("lid")
                    or item.get("identifier")
                    or item.get("title")
                    or ""
                )
            )

            if not raw_id:
                continue

            if raw_id in seen_raw:
                continue

            seen_raw.add(raw_id)
            retained_raw.append(item)

        # Once the locally filtered pool is comfortably larger than the number
        # that will reach INDUS, skip broader fallback queries.
        if len(retained_raw) >= target_pool_size:
            break

    parsed = []

    for item in retained_raw:
        record = parse_pds_record(
            item,
            allow_documentation=include_documents,
        )
        if record:
            parsed.append(record)

    parsed = deduplicate_resources(parsed)

    # Cheaply narrow the PDS pool before the expensive embedding step. The UI
    # search-depth value now approximately means how many PDS resources INDUS
    # will actually consider, matching the behavior of the other sources.
    parsed.sort(
        key=lambda resource: pds_prefilter_score(
            resource,
            query,
            targets,
            concepts,
        ),
        reverse=True,
    )

    usable_count = len(parsed)
    indus_candidates = parsed[:rows]

    st.session_state["pds_diagnostics"] = {
        "targets": targets,
        "concepts": concepts,
        "raw_api_hits": raw_api_hits,
        "raw_count": len(retained_raw),
        "parsed_count": usable_count,
        "indus_count": len(indus_candidates),
        "successful_calls": successful_calls,
        "errors": errors,
        "backend": "PDS Data Set Keyword Search",
        "search_terms": attempted_terms,
    }

    return indus_candidates


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🪐 Astrobiology Discovery Explorer")
st.caption(
    "Semantic discovery across scientific literature, planetary data, and NASA "
    "technical resources using NASA-IMPACT's INDUS-SDE-ST model."
)

SOURCE_CHECKBOX_DEFAULTS = {
    "source_ads": True,
    "source_arxiv": True,
    "source_ntrs": True,
    "source_pds_data": True,
    "source_pds_docs": False,
}

for source_key, default_value in SOURCE_CHECKBOX_DEFAULTS.items():
    if source_key not in st.session_state:
        st.session_state[source_key] = default_value

if "source_select_all" not in st.session_state:
    st.session_state["source_select_all"] = all(
        st.session_state[source_key]
        for source_key in SOURCE_CHECKBOX_DEFAULTS
    )


def toggle_all_sources():
    selected = st.session_state["source_select_all"]
    for source_key in SOURCE_CHECKBOX_DEFAULTS:
        st.session_state[source_key] = selected


def sync_select_all_sources():
    st.session_state["source_select_all"] = all(
        st.session_state[source_key]
        for source_key in SOURCE_CHECKBOX_DEFAULTS
    )


with st.sidebar:
    st.header("Discovery settings")

    st.checkbox(
        "Select all sources",
        key="source_select_all",
        on_change=toggle_all_sources,
        help="Check to select every source; uncheck to deselect every source.",
    )

    st.markdown("**Literature & Technical Sources**")
    source_ads = st.checkbox(
        "ADS",
        key="source_ads",
        on_change=sync_select_all_sources,
    )
    source_arxiv = st.checkbox(
        "arXiv",
        key="source_arxiv",
        on_change=sync_select_all_sources,
    )
    source_ntrs = st.checkbox(
        "NASA NTRS",
        key="source_ntrs",
        on_change=sync_select_all_sources,
    )

    st.markdown("**Planetary Data System**")
    source_pds_data = st.checkbox(
        "PDS datasets / collections",
        key="source_pds_data",
        on_change=sync_select_all_sources,
        disabled=not PDS_ENABLED,
        help=(
            "Temporarily disabled to reduce Streamlit compute usage. The PDS "
            "retrieval and parsing code remains in the app and can be re-enabled."
            if not PDS_ENABLED
            else
            "Search PDS bundles and data-like collections. Context registries, "
            "schema collections, miscellaneous infrastructure collections, and "
            "documentation are excluded from this option."
        ),
    )
    source_pds_docs = st.checkbox(
        "PDS documentation",
        key="source_pds_docs",
        on_change=sync_select_all_sources,
        disabled=not PDS_ENABLED,
        help=(
            "Temporarily disabled to reduce Streamlit compute usage."
            if not PDS_ENABLED
            else
            "Also include PDS manuals, user guides, interface documents, and documentation products."
        ),
    )
    if not PDS_ENABLED:
        st.caption("PDS search is temporarily disabled to conserve app compute.")

    year_range = st.slider(
        "Publication years",
        min_value=1990,
        max_value=CURRENT_YEAR,
        value=(2015, CURRENT_YEAR),
    )
    st.caption(
        "Publication years apply to ADS, arXiv, and NTRS. "
        "PDS archive resources are searched independently of this filter."
    )

    candidates_per_source = st.slider(
        "Candidate search depth",
        10,
        100,
        40,
        step=10,
        help=(
            "Controls approximately how many candidates from each selected source "
            "are sent to INDUS for semantic ranking. PDS may retrieve a larger raw "
            "pool internally, but it uses a cheap metadata/text prefilter so only "
            "about this many PDS resources reach the CPU-expensive embedding step."
        ),
    )
    st.caption(
        "Search depth controls roughly how many candidates per source INDUS "
        "embeds and ranks. It does not control how many final results are displayed."
    )
    results_to_show = st.slider(
        "Results to show",
        5,
        30,
        15,
        step=5,
    )

    st.divider()

    use_astro_lens = st.checkbox(
        "Use experimental Astrobiology Lens",
        value=False,
        help=(
            "When unchecked, ranking is based entirely on INDUS-SDE-ST similarity "
            "to your research question. When enabled, 25% of the ranking also "
            "reflects semantic similarity to broad astrobiology themes."
        ),
    )

    if use_astro_lens:
        st.caption(
            "Experimental lens enabled: 75% query similarity + "
            "25% astrobiology-theme similarity."
        )

    st.divider()
    st.caption(
        "Embedding model: "
        f"[`{MODEL_NAME}`]({MODEL_URL})"
    )
    st.caption("Similarity scores are ranking signals, not probabilities.")


query = st.text_area(
    "Ask an astrobiology research question",
    placeholder=(
        "Example: What measurements could distinguish biological from abiotic "
        "methane production on Titan or in ocean-world hydrothermal environments?"
    ),
    height=110,
)

search_clicked = st.button(
    "Explore the science",
    type="primary",
    use_container_width=True,
)


if search_clicked:
    if not query.strip():
        st.warning("Enter a research question first.")
        st.stop()

    ads_api_key = st.secrets.get("ADS_API_KEY", "")
    if source_ads and not ads_api_key:
        st.info(
            "ADS is enabled, but no `ADS_API_KEY` is configured in Streamlit Secrets. "
            "This search will continue with the other selected sources."
        )

    if not any(
        [
            source_ads and bool(ads_api_key),
            source_arxiv,
            source_ntrs,
            source_pds_data,
            source_pds_docs,
        ]
    ):
        st.warning("Enable at least one usable source.")
        st.stop()

    resources: List[dict] = []
    errors = []
    source_counts = {}
    search_started = time.perf_counter()

    with st.status("Gathering candidate science...", expanded=True) as status:
        # ADS, arXiv, and NTRS are independent network calls. Run them in
        # parallel so one slow service does not block the other sources.
        source_jobs = {}

        if source_ads and ads_api_key:
            st.write("Searching ADS publications...")
        if source_arxiv:
            st.write("Searching arXiv preprints...")
        if source_ntrs:
            st.write("Searching NASA NTRS technical material...")

        with ThreadPoolExecutor(max_workers=3) as executor:
            if source_ads and ads_api_key:
                started = time.perf_counter()
                future = executor.submit(
                    fetch_ads,
                    query,
                    ads_api_key,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )
                source_jobs[future] = ("ADS", started)

            if source_arxiv:
                started = time.perf_counter()
                future = executor.submit(
                    fetch_arxiv,
                    query,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )
                source_jobs[future] = ("arXiv", started)

            if source_ntrs:
                started = time.perf_counter()
                future = executor.submit(
                    fetch_ntrs,
                    query,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )
                source_jobs[future] = ("NASA NTRS", started)

            for future in as_completed(source_jobs):
                source_name, started = source_jobs[future]
                elapsed = time.perf_counter() - started
                try:
                    source_results = future.result()
                    resources.extend(source_results)
                    source_counts[source_name] = len(source_results)
                    st.write(
                        f"{source_name}: {len(source_results)} candidates "
                        f"({elapsed:.1f}s)"
                    )
                except Exception as exc:
                    errors.append(f"{source_name}: {exc}")
                    st.write(f"{source_name}: request failed ({elapsed:.1f}s)")

        if PDS_ENABLED and (source_pds_data or source_pds_docs):
            st.write("Searching NASA PDS archive datasets and collections...")
            try:
                pds_results = fetch_pds(
                    query,
                    candidates_per_source,
                    include_data=source_pds_data,
                    include_documents=source_pds_docs,
                )
                resources.extend(pds_results)
                source_counts["NASA PDS"] = len(pds_results)

                diagnostics = st.session_state.get("pds_diagnostics", {})
                targets = diagnostics.get("targets", [])
                raw_api_hits = diagnostics.get("raw_api_hits", 0)
                raw_count = diagnostics.get("raw_count", 0)
                parsed_count = diagnostics.get("parsed_count", 0)
                indus_count = diagnostics.get("indus_count", 0)
                if targets:
                    st.write(
                        "PDS target-aware archive search: "
                        f"{', '.join(targets)} · {raw_api_hits} raw archive hits · "
                        f"{raw_count} high-level target-matched candidates · "
                        f"{parsed_count} usable resources · "
                        f"{indus_count} sent to INDUS"
                    )
                else:
                    st.write(
                        f"PDS archive text discovery: {raw_count} candidates · "
                        f"{parsed_count} usable resources · "
                        f"{indus_count} sent to INDUS"
                    )
            except Exception as exc:
                errors.append(f"NASA PDS: {exc}")

        resources = deduplicate_resources(resources)
        resources.sort(key=resource_sort_key)
        retrieval_elapsed = time.perf_counter() - search_started
        st.write(f"Source retrieval completed in {retrieval_elapsed:.1f}s.")

        if source_counts:
            count_text = " · ".join(
                f"{source}: {count}"
                for source, count in source_counts.items()
            )
            st.write("Candidate records: " + count_text)

        st.write(f"Collected {len(resources)} unique scientific resources overall.")

        if resources:
            if use_astro_lens:
                st.write(
                    "Loading INDUS-SDE-ST and ranking all resource types "
                    "with the optional Astrobiology Lens..."
                )
            else:
                st.write(
                    "Loading INDUS-SDE-ST and ranking all resource types "
                    "by semantic similarity to your question..."
                )

            ranking_started = time.perf_counter()
            ranked, indus_diagnostics = rank_with_indus(
                resources,
                query,
                use_astro_lens=use_astro_lens,
            )
            ranking_elapsed = time.perf_counter() - ranking_started
            st.write(f"INDUS ranking completed in {ranking_elapsed:.1f}s.")

            if indus_diagnostics:
                st.write(
                    "INDUS diagnostics: "
                    f"model load/cache {indus_diagnostics.get('model_load_seconds', 0.0):.2f}s · "
                    f"query embedding {indus_diagnostics.get('query_embedding_seconds', 0.0):.2f}s · "
                    f"{indus_diagnostics.get('document_count', len(resources))} document embeddings "
                    f"{indus_diagnostics.get('document_embedding_seconds', 0.0):.2f}s · "
                    f"document cache {indus_diagnostics.get('document_cache_hits', 0)} hit / "
                    f"{indus_diagnostics.get('document_cache_misses', 0)} new · "
                    f"cosine scoring {indus_diagnostics.get('cosine_scoring_seconds', 0.0):.4f}s · "
                    f"dtype {indus_diagnostics.get('model_dtype', 'unknown')}"
                )
                if use_astro_lens:
                    st.write(
                        "Astrobiology Lens embedding/scoring: "
                        f"{indus_diagnostics.get('astro_lens_seconds', 0.0):.2f}s."
                    )
        else:
            ranked = []

        status.update(label="Discovery complete", state="complete")

    pds_diag = st.session_state.get("pds_diagnostics", {})
    pds_diag_errors = pds_diag.get("errors", [])
    if pds_diag_errors and pds_diag.get("parsed_count", 0) == 0:
        errors.append(
            "NASA PDS did not return any usable high-level resources. "
            f"{len(pds_diag_errors)} PDS request(s) failed."
        )

    if errors:
        with st.expander("Source warnings"):
            for error in errors:
                st.warning(error)

    if not ranked:
        st.warning(
            "No candidate resources with usable scientific metadata were retrieved. "
            "Try a broader question, a wider publication-year range, or different sources."
        )
        st.stop()

    top_results = ranked[:results_to_show]
    score_values = np.array(
        [result["combined_score"] for result in top_results],
        dtype=float,
    )

    if use_astro_lens:
        st.subheader("Astrobiology-ranked scientific resources")
        st.caption(
            f"INDUS reranked {len(ranked)} resources across the selected sources. "
            "The optional Astrobiology Lens is enabled."
        )
    else:
        st.subheader("INDUS semantic results")
        st.caption(
            f"INDUS-SDE-ST reranked {len(ranked)} resources across the selected sources "
            "according to semantic similarity to your research question."
        )

    for rank, resource in enumerate(top_results, start=1):
        label = strength_label(resource["combined_score"], score_values)
        source = resource["source"]
        resource_type = resource.get("resource_type", "Scientific Resource")
        year = resource.get("year") or "—"
        tags = " · ".join(resource.get("astro_tags", []))

        st.markdown(f"### {rank}. {resource['title']}")

        metadata_line = f"**{resource_type}** · {source}"
        if year != "—":
            metadata_line += f" · {year}"
        if use_astro_lens and tags:
            metadata_line += f" · Astrobiology themes: {tags}"
        st.caption(metadata_line)

        authors = resource.get("authors", [])
        if authors:
            short_authors = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
            st.markdown(f"**Authors:** {short_authors}")

        if source == "NASA PDS":
            targets = resource.get("pds_targets", [])
            investigations = resource.get("pds_investigations", [])
            instruments = resource.get("pds_instruments", [])

            if targets:
                st.markdown("**Target:** " + ", ".join(targets[:6]))
            if investigations:
                st.markdown("**Mission / Investigation:** " + ", ".join(investigations[:6]))
            if instruments:
                st.markdown("**Instrument / Observing System:** " + ", ".join(instruments[:6]))

        with st.expander("Description / Abstract"):
            st.write(resource["abstract"])

        col1, col2, col3 = st.columns([1.2, 1.4, 3.0])

        with col1:
            st.metric("Relevance", label)

        with col2:
            st.metric("INDUS cosine similarity", f"{resource['combined_score']:.3f}")

        with col3:
            links = []

            if source == "NASA PDS":
                if resource.get("url"):
                    links.append(f"[View in PDS]({resource['url']})")

                if resource.get("secondary_url"):
                    direct_url = resource["secondary_url"]
                    if direct_url.lower().endswith(".pdf"):
                        links.append(f"[Open PDF]({direct_url})")
                    elif resource.get("pds_product_class") in {
                        "Product_Bundle",
                        "Product_Collection",
                        "Product_Data_Set",
                        "Product_Data_Set_PDS3",
                    }:
                        links.append(f"[Browse archive files]({direct_url})")
                    else:
                        links.append(f"[Open data/resource]({direct_url})")

                extra_files = resource.get("extra_file_urls", [])
                for index, file_url in enumerate(extra_files[:2], start=1):
                    label_text = f"Additional file {index}"
                    if file_url.lower().endswith(".pdf"):
                        label_text = f"Additional PDF {index}"
                    elif file_url.lower().endswith(".csv"):
                        label_text = f"CSV file {index}"
                    links.append(f"[{label_text}]({file_url})")

                if resource.get("label_url"):
                    links.append(f"[PDS metadata label]({resource['label_url']})")
            else:
                if resource.get("url"):
                    links.append(f"[Open {source}]({resource['url']})")
                if resource.get("secondary_url"):
                    links.append(f"[Alternate / PDF]({resource['secondary_url']})")

            if resource.get("doi"):
                links.append(f"DOI: `{resource['doi']}`")

            if links:
                st.markdown(" &nbsp; | &nbsp; ".join(links))

        with st.expander("Why INDUS ranked this result"):
            st.write(
                "Similarity to your research question: "
                f"**{resource['query_score']:.3f}**"
            )

            if use_astro_lens:
                st.write(
                    "Astrobiology-lens similarity: "
                    f"**{resource['astro_score']:.3f}**"
                )
                if tags:
                    st.write(f"Closest astrobiology themes: **{tags}**")
                st.write(
                    "Final ranking uses **75% query similarity + "
                    "25% Astrobiology Lens similarity**."
                )
            else:
                st.write(
                    "The Astrobiology Lens is disabled, so this result is ranked entirely "
                    "from INDUS-SDE-ST similarity to your research question."
                )

            st.caption(
                "These values are cosine-similarity ranking signals in the INDUS embedding "
                "space. They are not calibrated probabilities or expert relevance judgments."
            )

        st.divider()
