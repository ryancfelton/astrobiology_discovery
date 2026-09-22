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


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Astrobiology Discovery Explorer",
    page_icon="🪐",
    layout="wide",
)

MODEL_NAME = "nasa-impact/indus-sde-st-v0.2"
MODEL_URL = "https://huggingface.co/nasa-impact/indus-sde-st-v0.2"
PDS_SEARCH_ENDPOINT = "https://pds.nasa.gov/api/search/1/products"
CURRENT_YEAR = datetime.now(timezone.utc).year

# The optional Astrobiology Lens is an experimental ranking layer built
# on top of INDUS. It does NOT modify or retrain the INDUS model.
ASTRO_LENS_WEIGHT = 0.25

ASTROBIOLOGY_LENSES: Dict[str, str] = {
    "Habitability": (
        "planetary habitability, environments capable of supporting life, liquid water, "
        "energy sources, nutrients, environmental limits, and habitable planetary conditions"
    ),
    "Biosignatures": (
        "biosignatures, life detection, biological gases, atmospheric disequilibrium, "
        "surface biosignatures, false positives, false negatives, and signs of life"
    ),
    "Origins of Life": (
        "origins of life, prebiotic chemistry, chemical evolution, abiogenesis, organic chemistry, "
        "emergence of metabolism, and early biochemical systems"
    ),
    "Ocean Worlds": (
        "ocean worlds, Europa, Enceladus, subsurface oceans, hydrothermal systems, ice-ocean interfaces, "
        "water-rock interaction, and icy moon habitability"
    ),
    "Mars & Ancient Environments": (
        "Mars habitability, ancient aqueous environments, sedimentary environments, organics, "
        "life detection on Mars, and preservation of biosignatures"
    ),
    "Exoplanets": (
        "exoplanet habitability, terrestrial exoplanets, atmospheric characterization, "
        "biosignature gases, stellar environments, and remotely detectable signs of life"
    ),
    "Planetary Chemistry": (
        "planetary geochemistry, atmospheric chemistry, photochemistry, serpentinization, "
        "redox disequilibrium, methane chemistry, and water-rock reactions"
    ),
    "Life in Extremes": (
        "extremophiles, microbial ecology, environmental limits of life, analog environments, "
        "deep biosphere, hydrothermal life, and microbial metabolism"
    ),
    "Titan & Organic Worlds": (
        "Titan, organic-rich planetary environments, atmospheric haze, hydrocarbon chemistry, "
        "prebiotic organic chemistry, and habitability beyond liquid-water surface environments"
    ),
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "their", "these", "this", "to", "what", "when", "where", "which", "with", "would",
    "relevant", "using", "use", "find", "show", "tell", "study", "studies", "research",
}

# Higher-level PDS products are much more useful in a discovery interface than
# thousands of individual observational files. We keep bundles, collections,
# documents, resources, and legacy PDS3 data-set records.
PDS_HIGH_LEVEL_TYPES = {
    "Product_Bundle",
    "Product_Collection",
    "Product_Document",
    "Product_Resource",
    "Product_Data_Set_PDS3",
    "Product_Data_Set",
}

PDS_FIELDS = ",".join(
    [
        "lid",
        "lidvid",
        "title",
        "product_class",
        "description",
        "pds:Identification_Area.pds:title",
        "pds:Identification_Area.pds:product_class",
        "pds:Citation_Information.pds:description",
        "pds:Citation_Information.pds:publication_year",
        "pds:Citation_Information.pds:doi",
        "pds:Modification_Detail.pds:description",
        "pds:Investigation_Area.pds:name",
        "pds:Target_Identification.pds:name",
        "pds:Observing_System_Component.pds:name",
        "pds:Observing_System_Component.pds:description",
        "ops:Label_File_Info.ops:file_ref",
    ]
)


# -----------------------------
# Generic helpers
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def informative_terms(query: str, max_terms: int = 10) -> List[str]:
    """
    Pull a small number of useful words from the user's question.

    These terms are only used to create broad searches against source APIs.
    INDUS-SDE-ST performs the semantic reranking afterward.
    """
    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9+\-\.]*",
        query.lower(),
    )

    terms = []

    for token in tokens:
        if len(token) < 3 or token in STOPWORDS:
            continue

        if token not in terms:
            terms.append(token)

        if len(terms) >= max_terms:
            break

    return terms


def humanize_pds_identifier(value: str) -> str:
    """Turn a PDS context LID into a friendlier label when no name is returned."""
    text = clean_text(str(value))

    if not text.startswith("urn:nasa:pds:context:"):
        return text

    tail = text.rsplit(":", 1)[-1]

    if "." in tail:
        tail = tail.split(".", 1)[-1]

    return tail.replace("_", " ").replace("-", " ").strip().title()


def first_scalar(value):
    """Return a readable scalar from common PDS JSON list/dict/scalar shapes."""
    if value is None:
        return ""

    if isinstance(value, list):
        for item in value:
            result = first_scalar(item)

            if result:
                return result

        return ""

    if isinstance(value, dict):
        for key in ("name", "title", "value", "id", "href"):
            if key in value:
                result = first_scalar(value.get(key))

                if result:
                    return result

        return ""

    return humanize_pds_identifier(str(value))


def all_scalars(value) -> List[str]:
    """Flatten a nested API value into a small list of readable strings."""
    results: List[str] = []

    if value is None:
        return results

    if isinstance(value, list):
        for item in value:
            results.extend(all_scalars(item))

        return list(dict.fromkeys([x for x in results if x]))

    if isinstance(value, dict):
        preferred = []

        for key in ("name", "title", "value", "id"):
            if key in value:
                preferred.extend(all_scalars(value.get(key)))

        return list(dict.fromkeys([x for x in preferred if x]))

    text = humanize_pds_identifier(str(value))

    return [text] if text else []


def first_present(record: dict, keys: List[str]) -> str:
    for key in keys:
        if key in record:
            value = first_scalar(record.get(key))

            if value:
                return value

    return ""


def list_present(record: dict, keys: List[str]) -> List[str]:
    values: List[str] = []

    for key in keys:
        if key in record:
            values.extend(all_scalars(record.get(key)))

    return list(dict.fromkeys([x for x in values if x]))


def resource_type_from_pds(product_type: str) -> str:
    if product_type == "Product_Bundle":
        return "PDS Bundle"

    if product_type == "Product_Collection":
        return "PDS Dataset / Collection"

    if product_type == "Product_Document":
        return "PDS Documentation"

    if product_type == "Product_Resource":
        return "PDS Resource"

    if "Data_Set" in product_type:
        return "PDS Dataset"

    return "PDS Resource"


def deduplicate_resources(resources: List[dict]) -> List[dict]:
    seen = set()
    unique = []

    for resource in resources:
        source = resource.get("source", "")
        record_id = clean_text(
            str(resource.get("record_id", ""))
        ).lower()

        doi = clean_text(
            str(resource.get("doi", ""))
        ).lower()

        title_key = re.sub(
            r"[^a-z0-9]",
            "",
            resource.get("title", "").lower(),
        )[:180]

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
    """
    Text passed into INDUS-SDE-ST for each retrieved scientific resource.
    """
    title = resource.get("title", "")
    abstract = resource.get("abstract", "")
    metadata_context = resource.get("metadata_context", "")

    return f"{title}. {abstract}. {metadata_context}".strip()


def strength_label(
    score: float,
    values: np.ndarray,
) -> str:
    """
    Relative label based on the current result set.

    These are not calibrated probabilities.
    """
    if len(values) < 3:
        return "Relevant"

    p70 = float(
        np.percentile(
            values,
            70,
        )
    )

    p40 = float(
        np.percentile(
            values,
            40,
        )
    )

    if score >= p70:
        return "Strong"

    if score >= p40:
        return "Moderate"

    return "Possible"


# -----------------------------
# INDUS model
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_indus_model():
    """
    Load NASA-IMPACT's public INDUS-SDE-ST model from Hugging Face.
    """
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


@st.cache_resource(show_spinner=False)
def lens_embeddings():
    """
    Convert the optional astrobiology concept descriptions into INDUS embeddings.

    This only runs if the user explicitly enables the Astrobiology Lens.
    """
    model = load_indus_model()

    texts = list(
        ASTROBIOLOGY_LENSES.values()
    )

    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def rank_with_indus(
    resources: List[dict],
    query: str,
    use_astro_lens: bool = False,
) -> List[dict]:
    """
    Rank candidate scientific resources using INDUS-SDE-ST.

    Default behavior:
        Rank only according to INDUS semantic similarity between the user's
        research question and each resource's descriptive text.

    Optional Astrobiology Lens:
        If enabled, also compare each resource to a small set of astrobiology
        concept descriptions and combine that signal with question similarity.
    """
    if not resources:
        return []

    model = load_indus_model()

    texts = [
        resource_text(resource)
        for resource in resources
    ]

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

    # Normalized vectors -> dot product equals cosine similarity.
    query_scores = (
        doc_embeddings
        @ query_embedding
    )

    if not use_astro_lens:
        ranked = []

        for idx, resource in enumerate(resources):
            item = dict(resource)

            item["query_score"] = float(
                query_scores[idx]
            )

            item["astro_score"] = None

            item["combined_score"] = float(
                query_scores[idx]
            )

            item["astro_tags"] = []

            ranked.append(item)

        ranked.sort(
            key=lambda r: r["combined_score"],
            reverse=True,
        )

        return ranked

    lens_emb = lens_embeddings()

    all_lens_scores = (
        doc_embeddings
        @ lens_emb.T
    )

    astro_scores = np.max(
        all_lens_scores,
        axis=1,
    )

    combined_scores = (
        (1.0 - ASTRO_LENS_WEIGHT)
        * query_scores
        + ASTRO_LENS_WEIGHT
        * astro_scores
    )

    lens_names = list(
        ASTROBIOLOGY_LENSES.keys()
    )

    ranked = []

    for idx, resource in enumerate(resources):
        lens_order = np.argsort(
            all_lens_scores[idx]
        )[::-1][:2]

        item = dict(resource)

        item["query_score"] = float(
            query_scores[idx]
        )

        item["astro_score"] = float(
            astro_scores[idx]
        )

        item["combined_score"] = float(
            combined_scores[idx]
        )

        item["astro_tags"] = [
            lens_names[j]
            for j in lens_order
        ]

        ranked.append(item)

    ranked.sort(
        key=lambda r: r["combined_score"],
        reverse=True,
    )

    return ranked


# -----------------------------
# Source retrieval: ADS / SciX
# -----------------------------
def fetch_ads(
    query: str,
    api_key: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:
    if not api_key:
        return []

    endpoint = (
        "https://api.adsabs.harvard.edu/"
        "v1/search/query"
    )

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "q": query,
        "fq": (
            f"year:[{start_year} "
            f"TO {end_year}]"
        ),
        "fl": (
            "title,author,year,pubdate,"
            "abstract,identifier,doi,"
            "bibcode,keyword"
        ),
        "rows": rows,
        "sort": "score desc",
    }

    response = requests.get(
        endpoint,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    docs = (
        response
        .json()
        .get("response", {})
        .get("docs", [])
    )

    resources = []

    for doc in docs:
        title = doc.get(
            "title",
            [""],
        )

        if isinstance(
            title,
            list,
        ) and title:
            title = title[0]

        else:
            title = str(title)

        abstract = clean_text(
            doc.get(
                "abstract",
                "",
            )
        )

        if not abstract:
            continue

        bibcode = doc.get(
            "bibcode",
            "",
        )

        doi_value = doc.get(
            "doi",
            [""],
        )

        if (
            isinstance(
                doi_value,
                list,
            )
            and doi_value
        ):
            doi = doi_value[0]

        else:
            doi = str(
                doi_value
                or ""
            )

        identifiers = (
            doc.get(
                "identifier",
                [],
            )
            or []
        )

        arxiv_id = ""

        for ident in identifiers:
            if str(ident).lower().startswith(
                "arxiv:"
            ):
                arxiv_id = (
                    str(ident)
                    .split(
                        ":",
                        1,
                    )[1]
                )

                break

        resources.append(
            {
                "title": clean_text(title),
                "authors": (
                    doc.get(
                        "author",
                        [],
                    )
                    or []
                ),
                "year": int(
                    doc.get(
                        "year",
                    )
                    or 0
                ),
                "abstract": abstract,
                "metadata_context": "",
                "source": "ADS/SciX",
                "resource_type": "Publication",
                "record_id": bibcode,
                "doi": doi,
                "url": (
                    "https://ui.adsabs.harvard.edu/"
                    f"abs/{bibcode}/abstract"
                    if bibcode
                    else ""
                ),
                "secondary_url": (
                    f"https://arxiv.org/abs/{arxiv_id}"
                    if arxiv_id
                    else ""
                ),
            }
        )

    return resources


# -----------------------------
# Source retrieval: arXiv
# -----------------------------
def fetch_arxiv(
    query: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:
    terms = informative_terms(query)

    if not terms:
        terms = [
            query.strip()
        ]

    term_query = " OR ".join(
        f'all:"{term}"'
        for term in terms
    )

    date_query = (
        f"submittedDate:["
        f"{start_year}01010000 "
        f"TO "
        f"{end_year}12312359]"
    )

    final_query = (
        f"({term_query}) "
        f"AND {date_query}"
    )

    client = arxiv.Client(
        page_size=min(
            rows,
            100,
        ),
        delay_seconds=3.0,
        num_retries=2,
    )

    search = arxiv.Search(
        query=final_query,
        max_results=rows,
        sort_by=(
            arxiv
            .SortCriterion
            .Relevance
        ),
    )

    resources = []

    for result in client.results(search):
        abstract = clean_text(
            result.summary
        )

        if not abstract:
            continue

        resources.append(
            {
                "title": clean_text(
                    result.title
                ),
                "authors": [
                    author.name
                    for author
                    in result.authors
                ],
                "year": (
                    result
                    .published
                    .year
                ),
                "abstract": abstract,
                "metadata_context": "",
                "source": "arXiv",
                "resource_type": "Preprint",
                "record_id": result.entry_id,
                "doi": (
                    result.doi
                    or ""
                ),
                "url": result.entry_id,
                "secondary_url": (
                    result.pdf_url
                    or ""
                ),
            }
        )

    return resources


# -----------------------------
# Source retrieval: NASA NTRS
# -----------------------------
def fetch_ntrs(
    query: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:
    endpoint = (
        "https://ntrs.nasa.gov/"
        "api/citations/search"
    )

    params = {
        "q": query,
        "publicationDateFrom": (
            f"{start_year}-01-01"
        ),
        "publicationDateTo": (
            f"{end_year}-12-31"
        ),
        "page": 1,
        "pageSize": rows,
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    docs = (
        response
        .json()
        .get(
            "results",
            [],
        )
    )

    resources = []

    for doc in docs[:rows]:
        abstract = clean_text(
            doc.get(
                "abstract",
                "",
            )
        )

        if not abstract:
            continue

        authors = []

        for auth_item in (
            doc.get(
                "authorAffiliations",
                [],
            )
            or []
        ):
            if isinstance(
                auth_item,
                dict,
            ):
                meta = (
                    auth_item.get(
                        "meta",
                        {},
                    )
                    or {}
                )

                author_obj = (
                    meta.get(
                        "author",
                        {},
                    )
                    or {}
                )

                name = (
                    author_obj.get(
                        "name"
                    )
                    or auth_item.get(
                        "name"
                    )
                )

                if name:
                    authors.append(name)

        if not authors:
            for author in (
                doc.get(
                    "authors",
                    [],
                )
                or []
            ):
                if (
                    isinstance(
                        author,
                        dict,
                    )
                    and author.get(
                        "name"
                    )
                ):
                    authors.append(
                        author["name"]
                    )

                elif isinstance(
                    author,
                    str,
                ):
                    authors.append(author)

        ntrs_id = str(
            doc.get(
                "id",
                "",
            )
        )

        pub_date = str(
            doc.get(
                "publicationDate"
            )
            or doc.get(
                "issued"
            )
            or doc.get(
                "created"
            )
            or ""
        )

        year_match = re.match(
            r"(\d{4})",
            pub_date,
        )

        year = (
            int(
                year_match.group(1)
            )
            if year_match
            else 0
        )

        resources.append(
            {
                "title": clean_text(
                    doc.get(
                        "title",
                        "",
                    )
                ),
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "metadata_context": "",
                "source": "NASA NTRS",
                "resource_type": (
                    "Technical Report / "
                    "NASA Document"
                ),
                "record_id": ntrs_id,
                "doi": clean_text(
                    str(
                        doc.get(
                            "doi",
                            "",
                        )
                    )
                ),
                "url": (
                    "https://ntrs.nasa.gov/"
                    f"citations/{ntrs_id}"
                    if ntrs_id
                    else ""
                ),
                "secondary_url": "",
            }
        )

    return resources


# -----------------------------
# Source retrieval: NASA PDS
# -----------------------------
def parse_pds_record(
    item: dict,
):
    """
    Convert one PDS Search API result into the common resource format used by
    the rest of the app.

    The PDS API has changed its response representation over time, so this
    parser intentionally checks both compact application/json names and PDS4
    metadata property names.
    """
    product_type = first_present(
        item,
        [
            "type",
            "product_class",
            (
                "pds:Identification_Area."
                "pds:product_class"
            ),
        ],
    )

    # Keep higher-level archive resources rather than individual data files.
    if (
        product_type
        and product_type
        not in PDS_HIGH_LEVEL_TYPES
    ):
        return None

    record_id = first_present(
        item,
        [
            "id",
            "lidvid",
            "lid",
            (
                "pds:Identification_Area."
                "pds:logical_identifier"
            ),
        ],
    )

    title = first_present(
        item,
        [
            "title",
            (
                "pds:Identification_Area."
                "pds:title"
            ),
        ],
    )

    if not title:
        return None

    description_parts = []

    for key in [
        "description",
        (
            "pds:Citation_Information."
            "pds:description"
        ),
        (
            "pds:Modification_Detail."
            "pds:description"
        ),
        (
            "pds:Observing_System_Component."
            "pds:description"
        ),
    ]:
        if key in item:
            description_parts.extend(
                all_scalars(
                    item.get(key)
                )
            )

    description_parts = list(
        dict.fromkeys(
            [
                x
                for x
                in description_parts
                if x
            ]
        )
    )

    targets = list_present(
        item,
        [
            "targets",
            (
                "pds:Target_Identification."
                "pds:name"
            ),
        ],
    )

    investigations = list_present(
        item,
        [
            "investigations",
            (
                "pds:Investigation_Area."
                "pds:name"
            ),
        ],
    )

    instruments = list_present(
        item,
        [
            "observing_system_components",
            (
                "pds:Observing_System_Component."
                "pds:name"
            ),
        ],
    )

    metadata_bits = []

    if product_type:
        metadata_bits.append(
            "PDS product type: "
            f"{product_type}"
        )

    if targets:
        metadata_bits.append(
            "Targets: "
            + ", ".join(
                targets[:8]
            )
        )

    if investigations:
        metadata_bits.append(
            "Investigations: "
            + ", ".join(
                investigations[:8]
            )
        )

    if instruments:
        metadata_bits.append(
            "Observing systems: "
            + ", ".join(
                instruments[:8]
            )
        )

    metadata_context = ". ".join(
        metadata_bits
    )

    abstract = ". ".join(
        description_parts
    )

    # If no prose description is returned, structured target/mission/instrument
    # metadata still gives INDUS useful scientific context.
    if not abstract:
        abstract = metadata_context

    if not abstract:
        return None

    year_text = first_present(
        item,
        [
            (
                "pds:Citation_Information."
                "pds:publication_year"
            ),
            "publication_year",
        ],
    )

    year_match = re.search(
        r"\b(19|20)\d{2}\b",
        year_text,
    )

    year = (
        int(
            year_match.group(0)
        )
        if year_match
        else 0
    )

    doi = first_present(
        item,
        [
            (
                "pds:Citation_Information."
                "pds:doi"
            ),
            "doi",
        ],
    )

    label_ref = first_present(
        item,
        [
            (
                "ops:Label_File_Info."
                "ops:file_ref"
            ),
            "href",
        ],
    )

    if (
        label_ref.startswith(
            "http://"
        )
        or label_ref.startswith(
            "https://"
        )
    ):
        secondary_url = label_ref

    else:
        secondary_url = ""

    api_url = ""

    if record_id:
        api_url = (
            "https://pds.nasa.gov/"
            "api/search/1/products/"
            + quote(
                record_id,
                safe=":",
            )
        )

    return {
        "title": title,
        "authors": [],
        "year": year,
        "abstract": abstract,
        "metadata_context": metadata_context,
        "source": "NASA PDS",
        "resource_type": (
            resource_type_from_pds(
                product_type
            )
        ),
        "record_id": record_id,
        "doi": doi,
        "url": api_url,
        "secondary_url": secondary_url,
        "pds_product_type": product_type,
        "pds_targets": targets,
        "pds_investigations": investigations,
        "pds_instruments": instruments,
    }


def fetch_pds(
    query: str,
    rows: int,
) -> List[dict]:
    """
    Search NASA PDS for higher-level scientific resources.

    PDS keyword search operates over product title/description. We make several
    broad keyword calls using informative terms from the user's question, merge
    the candidate pool, keep higher-level archive resources, and then let
    INDUS-SDE-ST perform the semantic ranking.
    """
    terms = informative_terms(
        query,
        max_terms=6,
    )

    if not terms:
        terms = [
            query.strip()
        ]

    # Over-fetch because many PDS keyword hits may be individual products that
    # we intentionally discard in favor of bundles/collections/documents.
    per_term_limit = max(
        20,
        min(
            100,
            math.ceil(
                (rows * 3)
                / max(
                    len(terms),
                    1,
                )
            ),
        ),
    )

    collected: List[dict] = []
    seen_ids = set()

    for term in terms:
        params = {
            "keywords": term,
            "fields": PDS_FIELDS,
            "limit": per_term_limit,
        }

        response = requests.get(
            PDS_SEARCH_ENDPOINT,
            params=params,
            headers={
                "Accept": "application/json"
            },
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        items = payload.get(
            "data",
            [],
        )

        if not isinstance(
            items,
            list,
        ):
            continue

        for item in items:
            if not isinstance(
                item,
                dict,
            ):
                continue

            parsed = parse_pds_record(
                item
            )

            if not parsed:
                continue

            record_id = parsed.get(
                "record_id",
                "",
            )

            dedupe_key = (
                record_id
                or parsed[
                    "title"
                ].lower()
            )

            if dedupe_key in seen_ids:
                continue

            seen_ids.add(
                dedupe_key
            )

            collected.append(
                parsed
            )

    # The individual keyword calls are only candidate generation.
    # INDUS will perform the final relevance ordering afterward.
    return collected[
        :max(
            rows * 3,
            rows,
        )
    ]


# -----------------------------
# UI
# -----------------------------
st.title(
    "🪐 Astrobiology Discovery Explorer"
)

st.caption(
    "Experimental semantic discovery across scientific literature, "
    "planetary data, and NASA technical resources using "
    "NASA-IMPACT's INDUS-SDE-ST model."
)


with st.sidebar:
    st.header(
        "Discovery settings"
    )

    source_ads = st.checkbox(
        "ADS / SciX",
        value=True,
    )

    source_arxiv = st.checkbox(
        "arXiv",
        value=True,
    )

    source_ntrs = st.checkbox(
        "NASA NTRS",
        value=True,
    )

    source_pds = st.checkbox(
        "NASA Planetary Data System (PDS)",
        value=True,
    )

    year_range = st.slider(
        "Publication years",
        min_value=1990,
        max_value=CURRENT_YEAR,
        value=(
            2015,
            CURRENT_YEAR,
        ),
    )

    st.caption(
        "The publication-year filter applies to ADS/SciX, arXiv, "
        "and NTRS. PDS archive resources are searched independently "
        "of publication year."
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
            "Optional experimental ranking layer. "
            "When unchecked, results are ranked only by "
            "INDUS-SDE-ST similarity to your research question. "
            "When enabled, 25% of the ranking also reflects "
            "semantic similarity to broad astrobiology themes."
        ),
    )

    if use_astro_lens:
        st.caption(
            "Experimental lens enabled: "
            "75% query similarity + "
            "25% astrobiology-theme similarity."
        )

    st.divider()

    st.caption(
        "Embedding model: "
        f"[`{MODEL_NAME}`]"
        f"({MODEL_URL})"
    )

    st.caption(
        "Similarity scores are ranking signals, "
        "not probabilities."
    )


query = st.text_area(
    "Ask an astrobiology research question",
    placeholder=(
        "Example: What measurements could distinguish biological "
        "from abiotic methane production in hydrothermal environments "
        "on ocean worlds?"
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
        st.warning(
            "Enter a research question first."
        )

        st.stop()

    ads_api_key = st.secrets.get(
        "ADS_API_KEY",
        "",
    )

    if source_ads and not ads_api_key:
        st.info(
            "ADS/SciX is enabled, but no `ADS_API_KEY` is configured "
            "in Streamlit Secrets. This search will continue with "
            "the other selected sources."
        )

    if not any(
        [
            source_ads
            and bool(
                ads_api_key
            ),
            source_arxiv,
            source_ntrs,
            source_pds,
        ]
    ):
        st.warning(
            "Enable at least one usable source."
        )

        st.stop()

    resources: List[dict] = []
    errors = []
    source_counts = {}

    with st.status(
        "Gathering candidate science...",
        expanded=True,
    ) as status:

        if (
            source_ads
            and ads_api_key
        ):
            st.write(
                "Searching ADS/SciX publications..."
            )

            try:
                ads_results = fetch_ads(
                    query,
                    ads_api_key,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )

                resources.extend(
                    ads_results
                )

                source_counts[
                    "ADS/SciX"
                ] = len(
                    ads_results
                )

            except Exception as exc:
                errors.append(
                    f"ADS/SciX: {exc}"
                )

        if source_arxiv:
            st.write(
                "Searching arXiv preprints..."
            )

            try:
                arxiv_results = fetch_arxiv(
                    query,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )

                resources.extend(
                    arxiv_results
                )

                source_counts[
                    "arXiv"
                ] = len(
                    arxiv_results
                )

            except Exception as exc:
                errors.append(
                    f"arXiv: {exc}"
                )

        if source_ntrs:
            st.write(
                "Searching NASA NTRS technical material..."
            )

            try:
                ntrs_results = fetch_ntrs(
                    query,
                    year_range[0],
                    year_range[1],
                    candidates_per_source,
                )

                resources.extend(
                    ntrs_results
                )

                source_counts[
                    "NASA NTRS"
                ] = len(
                    ntrs_results
                )

            except Exception as exc:
                errors.append(
                    f"NASA NTRS: {exc}"
                )

        if source_pds:
            st.write(
                "Searching NASA PDS datasets "
                "and archive resources..."
            )

            try:
                pds_results = fetch_pds(
                    query,
                    candidates_per_source,
                )

                resources.extend(
                    pds_results
                )

                source_counts[
                    "NASA PDS"
                ] = len(
                    pds_results
                )

            except Exception as exc:
                errors.append(
                    f"NASA PDS: {exc}"
                )

        resources = deduplicate_resources(
            resources
        )

        if source_counts:
            count_text = " · ".join(
                f"{source}: {count}"
                for source, count
                in source_counts.items()
            )

            st.write(
                f"Candidate records: {count_text}"
            )

        st.write(
            f"Collected {len(resources)} unique "
            "scientific resources overall."
        )

        if resources:
            if use_astro_lens:
                st.write(
                    "Loading INDUS-SDE-ST and ranking all resource "
                    "types together with the optional Astrobiology Lens..."
                )

            else:
                st.write(
                    "Loading INDUS-SDE-ST and ranking all resource "
                    "types together by semantic similarity to your question..."
                )

            ranked = rank_with_indus(
                resources,
                query,
                use_astro_lens=use_astro_lens,
            )

        else:
            ranked = []

        status.update(
            label="Discovery complete",
            state="complete",
        )

    if errors:
        with st.expander(
            "Source warnings"
        ):
            for error in errors:
                st.warning(
                    error
                )

    if not ranked:
        st.warning(
            "No candidate resources with usable descriptive metadata "
            "were retrieved. Try a broader question, a wider "
            "publication-year range, or different sources."
        )

        st.stop()

    top_results = ranked[
        :results_to_show
    ]

    score_values = np.array(
        [
            result[
                "combined_score"
            ]
            for result
            in top_results
        ],
        dtype=float,
    )

    if use_astro_lens:
        st.subheader(
            "Astrobiology-ranked scientific resources"
        )

        st.caption(
            f"INDUS reranked {len(ranked)} resources across "
            "the selected sources. The optional Astrobiology Lens "
            "is enabled."
        )

    else:
        st.subheader(
            "INDUS semantic results"
        )

        st.caption(
            f"INDUS-SDE-ST reranked {len(ranked)} resources "
            "across the selected sources according to semantic "
            "similarity to your research question."
        )

    for rank, resource in enumerate(
        top_results,
        start=1,
    ):
        label = strength_label(
            resource[
                "combined_score"
            ],
            score_values,
        )

        source = resource[
            "source"
        ]

        resource_type = resource.get(
            "resource_type",
            "Scientific Resource",
        )

        year = (
            resource.get(
                "year"
            )
            or "—"
        )

        tags = " · ".join(
            resource.get(
                "astro_tags",
                [],
            )
        )

        st.markdown(
            f"### {rank}. {resource['title']}"
        )

        metadata_line = (
            f"**{resource_type}** "
            f"· {source}"
        )

        if year != "—":
            metadata_line += (
                f" · {year}"
            )

        if (
            use_astro_lens
            and tags
        ):
            metadata_line += (
                " · Astrobiology themes: "
                f"{tags}"
            )

        st.caption(
            metadata_line
        )

        authors = resource.get(
            "authors",
            [],
        )

        if authors:
            short_authors = (
                ", ".join(
                    authors[:5]
                )
                + (
                    " et al."
                    if len(authors) > 5
                    else ""
                )
            )

            st.markdown(
                f"**Authors:** {short_authors}"
            )

        # PDS-specific metadata is useful enough to show directly.
        if source == "NASA PDS":
            pds_targets = resource.get(
                "pds_targets",
                [],
            )

            pds_investigations = resource.get(
                "pds_investigations",
                [],
            )

            pds_instruments = resource.get(
                "pds_instruments",
                [],
            )

            pds_bits = []

            if pds_targets:
                pds_bits.append(
                    "**Targets:** "
                    + ", ".join(
                        pds_targets[:6]
                    )
                )

            if pds_investigations:
                pds_bits.append(
                    "**Investigations:** "
                    + ", ".join(
                        pds_investigations[:6]
                    )
                )

            if pds_instruments:
                pds_bits.append(
                    "**Observing systems:** "
                    + ", ".join(
                        pds_instruments[:6]
                    )
                )

            if pds_bits:
                st.markdown(
                    "  \n".join(
                        pds_bits
                    )
                )

        with st.expander(
            "Description / Abstract"
        ):
            st.write(
                resource[
                    "abstract"
                ]
            )

        col1, col2, col3 = st.columns(
            [
                1.2,
                1.2,
                2.6,
            ]
        )

        with col1:
            st.metric(
                "Relevance",
                label,
            )

        with col2:
            st.metric(
                "INDUS cosine similarity",
                (
                    f"{resource['combined_score']:.3f}"
                ),
            )

        with col3:
            links = []

            if resource.get(
                "url"
            ):
                if source == "NASA PDS":
                    links.append(
                        "[Open PDS record]"
                        f"({resource['url']})"
                    )

                else:
                    links.append(
                        f"[Open {source}]"
                        f"({resource['url']})"
                    )

            if resource.get(
                "secondary_url"
            ):
                if source == "NASA PDS":
                    links.append(
                        "[Archive / label link]"
                        f"({resource['secondary_url']})"
                    )

                else:
                    links.append(
                        "[Alternate / PDF]"
                        f"({resource['secondary_url']})"
                    )

            if resource.get(
                "doi"
            ):
                links.append(
                    f"DOI: `{resource['doi']}`"
                )

            if links:
                st.markdown(
                    " &nbsp; | &nbsp; ".join(
                        links
                    )
                )

        with st.expander(
            "Why INDUS ranked this result"
        ):
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
                    st.write(
                        "Closest astrobiology themes: "
                        f"**{tags}**"
                    )

                st.write(
                    "Final ranking uses **75% query similarity "
                    "+ 25% Astrobiology Lens similarity**."
                )

            else:
                st.write(
                    "The Astrobiology Lens is disabled, so this result "
                    "is ranked entirely from INDUS-SDE-ST similarity "
                    "to your research question."
                )

            st.caption(
                "These values are cosine-similarity ranking signals "
                "in the INDUS embedding space. They are not calibrated "
                "probabilities or expert relevance judgments."
            )

        st.divider()
