import html
import math
import re
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import quote

import arxiv
import numpy as np
import requests
import streamlit as st
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
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
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


def resource_text(resource: dict) -> str:
    title = resource.get("title", "")
    description = resource.get("abstract", "")
    context = resource.get("metadata_context", "")
    return f"{title}. {description}. {context}".strip()


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
        ["pds:Citation_Information.pds:description"],
        ["pds:Document.pds:description"],
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


def get_pds_collection_type(item: dict) -> str:
    return pds_first(item, ["collection_type", "pds:Collection.pds:collection_type"])


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
    return "PDS Resource"


# ============================================================
# INDUS MODEL
# ============================================================


@st.cache_resource(show_spinner=False)
def load_indus_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner=False)
def lens_embeddings():
    model = load_indus_model()
    return model.encode(
        list(ASTROBIOLOGY_LENSES.values()),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def rank_with_indus(
    resources: List[dict],
    query: str,
    use_astro_lens: bool = False,
) -> List[dict]:
    if not resources:
        return []

    model = load_indus_model()
    texts = [resource_text(resource) for resource in resources]

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    doc_embeddings = model.encode(
        texts,
        batch_size=12,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_scores = doc_embeddings @ query_embedding

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
        return ranked

    lens_emb = lens_embeddings()
    all_lens_scores = doc_embeddings @ lens_emb.T
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
    return ranked


# ============================================================
# ADS / SCIX
# ============================================================


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
                "source": "ADS/SciX",
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

    collection_type = get_pds_collection_type(item)
    is_doc = (
        product_class == "Product_Document"
        or (product_class == "Product_Collection" and collection_type.lower() == "document")
    )

    if is_doc and not allow_documentation:
        return None

    if product_class not in {"Product_Bundle", "Product_Collection", "Product_Document"}:
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
        ],
    )

    year_text = pds_first(
        item,
        [
            "pds:Citation_Information.pds:publication_year",
            "pds:Document.pds:publication_date",
            "citation_publication_year",
            "publication_year",
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
    direct_file = preferred_direct_file(file_urls)
    label_url = get_pds_label_url(item)
    human_url = pds_human_page(product_class, lid, version)

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

    extra_file_urls = [url for url in file_urls if url != direct_file][:3]

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
        "secondary_url": direct_file,
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


def pds_search_class(
    class_name: str,
    *,
    q: str = "",
    keywords: str = "",
    limit: int = 100,
) -> List[dict]:
    """
    Search a specific high-level PDS class.

    PDS documents the class endpoints as bundles, collections, documents,
    observationals, and any. q handles structured metadata constraints such as
    target_name eq "Titan"; keywords searches title/description text.
    """
    endpoint = f"{PDS_API_BASE}/classes/{class_name}"
    params = {"limit": min(max(limit, 1), 100)}
    if q:
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


def fetch_pds(
    query: str,
    rows: int,
    include_data: bool,
    include_documents: bool,
) -> List[dict]:
    """
    Build a broad, scientifically plausible PDS candidate pool BEFORE INDUS.

    If the user's query names a recognized planetary target, the app queries
    PDS's structured target_name field directly for bundles/collections (and
    documents if requested). This avoids the old failure mode where the first
    mixed keyword page contained few high-level products.

    INDUS-SDE-ST then reranks this broad candidate pool against the full user
    question, e.g. target-only Titan candidates are reranked for methane.
    """
    classes = []
    if include_data:
        classes.extend(PDS_DATA_CLASSES)
    if include_documents:
        classes.extend(PDS_DOCUMENT_CLASSES)
    classes = list(dict.fromkeys(classes))

    if not classes:
        return []

    targets = detect_pds_targets(query)
    concepts = pds_concept_terms(query, targets, max_terms=5)

    # We intentionally over-fetch from PDS because INDUS needs a meaningful
    # candidate pool to rerank. The final UI still shows only results_to_show.
    target_limit = min(100, max(50, rows * 2))
    concept_limit = min(100, max(30, rows))

    raw_items = []
    seen_raw = set()
    errors = []

    def add_items(items):
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_id = clean_text(str(item.get("id") or item.get("lidvid") or item.get("title") or ""))
            if not raw_id:
                raw_id = str(hash(str(item)))
            if raw_id in seen_raw:
                continue
            seen_raw.add(raw_id)
            raw_items.append(item)

    # 1) STRUCTURED TARGET RETRIEVAL — the important fix.
    #    Example: target_name eq "Titan" directly asks PDS for Titan resources.
    for target in targets:
        q = f'target_name eq "{target}"'
        for class_name in classes:
            try:
                add_items(
                    pds_search_class(
                        class_name,
                        q=q,
                        limit=target_limit,
                    )
                )
            except Exception as exc:
                errors.append(f"{class_name} target={target}: {exc}")

        # 2) TARGET + CONCEPT SEARCHES — useful when descriptions explicitly
        #    mention methane, atmosphere, organics, etc.
        for concept in concepts[:3]:
            for class_name in classes:
                try:
                    add_items(
                        pds_search_class(
                            class_name,
                            q=q,
                            keywords=concept,
                            limit=concept_limit,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{class_name} target={target} keyword={concept}: {exc}")

    # 3) KEYWORD DISCOVERY — catches resources when no target is recognized and
    #    adds cross-target material that is semantically relevant to the query.
    keyword_queries = []
    if query.strip():
        keyword_queries.append(query.strip())
    keyword_queries.extend(concepts if targets else informative_terms(query, max_terms=6))
    keyword_queries = list(dict.fromkeys([q for q in keyword_queries if q]))

    # Avoid making too many network calls while still broadening the pool.
    for keyword in keyword_queries[:5]:
        for class_name in classes:
            try:
                add_items(
                    pds_search_class(
                        class_name,
                        keywords=keyword,
                        limit=concept_limit,
                    )
                )
            except Exception as exc:
                errors.append(f"{class_name} keyword={keyword}: {exc}")

    parsed = []
    for item in raw_items:
        record = parse_pds_record(item, allow_documentation=include_documents)
        if record:
            parsed.append(record)

    parsed = deduplicate_resources(parsed)

    # Store diagnostics for the Streamlit status panel without failing the whole
    # search if one supplemental PDS call errors.
    st.session_state["pds_diagnostics"] = {
        "targets": targets,
        "concepts": concepts,
        "raw_count": len(raw_items),
        "parsed_count": len(parsed),
        "errors": errors,
    }

    # Keep a broad but bounded candidate pool for INDUS reranking.
    return parsed[: min(300, max(rows * 6, rows))]


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🪐 Astrobiology Discovery Explorer")
st.caption(
    "Semantic discovery across scientific literature, planetary data, and NASA "
    "technical resources using NASA-IMPACT's INDUS-SDE-ST model."
)

with st.sidebar:
    st.header("Discovery settings")

    st.markdown("**Literature & Technical Sources**")
    source_ads = st.checkbox("ADS / SciX", value=True)
    source_arxiv = st.checkbox("arXiv", value=True)
    source_ntrs = st.checkbox("NASA NTRS", value=True)

    st.markdown("**Planetary Data System**")
    source_pds_data = st.checkbox(
        "PDS datasets / collections",
        value=True,
        help="Search PDS bundles and collections using structured target metadata plus text discovery.",
    )
    source_pds_docs = st.checkbox(
        "PDS documentation",
        value=False,
        help="Also include PDS manuals, user guides, interface documents, and documentation products.",
    )

    year_range = st.slider(
        "Publication years",
        min_value=1990,
        max_value=CURRENT_YEAR,
        value=(2015, CURRENT_YEAR),
    )
    st.caption(
        "Publication years apply to ADS/SciX, arXiv, and NTRS. "
        "PDS archive resources are searched independently of this filter."
    )

    candidates_per_source = st.slider(
        "Candidates per source",
        10,
        100,
        40,
        step=10,
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
            "ADS/SciX is enabled, but no `ADS_API_KEY` is configured in Streamlit Secrets. "
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

    with st.status("Gathering candidate science...", expanded=True) as status:
        if source_ads and ads_api_key:
            st.write("Searching ADS/SciX publications...")
            try:
                ads_results = fetch_ads(
                    query,
                    ads_api_key,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )
                resources.extend(ads_results)
                source_counts["ADS/SciX"] = len(ads_results)
            except Exception as exc:
                errors.append(f"ADS/SciX: {exc}")

        if source_arxiv:
            st.write("Searching arXiv preprints...")
            try:
                arxiv_results = fetch_arxiv(
                    query,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )
                resources.extend(arxiv_results)
                source_counts["arXiv"] = len(arxiv_results)
            except Exception as exc:
                errors.append(f"arXiv: {exc}")

        if source_ntrs:
            st.write("Searching NASA NTRS technical material...")
            try:
                ntrs_results = fetch_ntrs(
                    query,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )
                resources.extend(ntrs_results)
                source_counts["NASA NTRS"] = len(ntrs_results)
            except Exception as exc:
                errors.append(f"NASA NTRS: {exc}")

        if source_pds_data or source_pds_docs:
            st.write("Searching NASA PDS with structured target and product-class retrieval...")
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
                raw_count = diagnostics.get("raw_count", 0)
                parsed_count = diagnostics.get("parsed_count", 0)
                if targets:
                    st.write(
                        "PDS structured target search: "
                        f"{', '.join(targets)} · {raw_count} raw candidates · "
                        f"{parsed_count} usable high-level resources"
                    )
                else:
                    st.write(
                        f"PDS text discovery: {raw_count} raw candidates · "
                        f"{parsed_count} usable high-level resources"
                    )
            except Exception as exc:
                errors.append(f"NASA PDS: {exc}")

        resources = deduplicate_resources(resources)

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

            ranked = rank_with_indus(
                resources,
                query,
                use_astro_lens=use_astro_lens,
            )
        else:
            ranked = []

        status.update(label="Discovery complete", state="complete")

    pds_diag_errors = st.session_state.get("pds_diagnostics", {}).get("errors", [])
    if pds_diag_errors:
        errors.append(
            f"NASA PDS supplemental calls: {len(pds_diag_errors)} request(s) failed; "
            "other PDS results were retained."
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
