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

PDS_SEARCH_ENDPOINT = "https://pds.nasa.gov/api/search/1/products"

CURRENT_YEAR = datetime.now(timezone.utc).year


# ============================================================
# OPTIONAL ASTROBIOLOGY LENS
# ============================================================

# This is an experimental ranking layer that WE created.
# It does not modify or retrain INDUS.
#
# Default application behavior leaves this disabled.
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
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "do", "does", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "their", "these", "this", "to", "what", "when",
    "where", "which", "with", "would", "relevant", "using",
    "use", "find", "show", "tell", "study", "studies",
    "research",
}


# Higher-level PDS products that are useful to scientists.
PDS_DATA_TYPES = {
    "Product_Bundle",
    "Product_Collection",
    "Product_Data_Set_PDS3",
    "Product_Data_Set",
    "Product_Resource",
}

PDS_DOCUMENT_TYPES = {
    "Product_Document",
}


# Fields requested from the PDS Search API.
# The API sometimes returns many additional properties too.
PDS_FIELDS = ",".join(
    [
        "lid",
        "lidvid",
        "vid",
        "title",
        "product_class",
        "description",

        "pds:Identification_Area.pds:title",
        "pds:Identification_Area.pds:product_class",

        "pds:Citation_Information.pds:description",
        "pds:Citation_Information.pds:publication_year",
        "pds:Citation_Information.pds:doi",
        "pds:Citation_Information.pds:author_list",

        "pds:Document.pds:description",
        "pds:Document.pds:publication_date",
        "pds:Document.pds:doi",
        "pds:Document.pds:author_list",

        "pds:Investigation_Area.pds:name",
        "pds:Target_Identification.pds:name",
        "pds:Observing_System_Component.pds:name",

        "ops:Data_File_Info.ops:file_ref",
        "ops:Data_File_Info.ops:file_name",

        "ops:Label_File_Info.ops:file_ref",
        "ops:Label_File_Info.ops:file_name",
    ]
)


# ============================================================
# GENERIC TEXT HELPERS
# ============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def informative_terms(
    query: str,
    max_terms: int = 10,
) -> List[str]:

    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9+\-\.]*",
        query.lower(),
    )

    terms = []

    for token in tokens:

        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        if token not in terms:
            terms.append(token)

        if len(terms) >= max_terms:
            break

    return terms


# ============================================================
# PDS VALUE / METADATA HELPERS
# ============================================================

def humanize_pds_identifier(
    value: str,
) -> str:

    text = clean_text(
        str(value)
    )

    if not text.startswith(
        "urn:nasa:pds:context:"
    ):
        return text

    tail = text.rsplit(
        ":",
        1,
    )[-1]

    if "." in tail:
        tail = tail.split(
            ".",
            1,
        )[-1]

    tail = (
        tail
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    return tail.title()


def flatten_values(
    value,
) -> List[str]:

    results = []

    if value is None:
        return results

    if isinstance(
        value,
        list,
    ):

        for item in value:
            results.extend(
                flatten_values(item)
            )

        return list(
            dict.fromkeys(
                [
                    x
                    for x in results
                    if x
                ]
            )
        )

    if isinstance(
        value,
        dict,
    ):

        # Prefer human-readable values if they exist.
        for key in [
            "name",
            "title",
            "value",
        ]:

            if key in value:

                nested = flatten_values(
                    value.get(key)
                )

                if nested:
                    results.extend(
                        nested
                    )

        # Fall back to PDS identifier.
        if not results and "id" in value:

            results.append(
                humanize_pds_identifier(
                    value["id"]
                )
            )

        return list(
            dict.fromkeys(
                [
                    x
                    for x in results
                    if x
                ]
            )
        )

    text = humanize_pds_identifier(
        str(value)
    )

    if text:
        return [text]

    return []


def pds_sources(
    item: dict,
) -> List[dict]:
    """
    PDS Search results often contain useful science metadata
    inside item["properties"].

    We check properties first because those values are often
    richer than the simplified top-level representation.
    """

    properties = item.get(
        "properties",
        {},
    )

    metadata = item.get(
        "metadata",
        {},
    )

    sources = []

    if isinstance(
        properties,
        dict,
    ):
        sources.append(
            properties
        )

    sources.append(
        item
    )

    if isinstance(
        metadata,
        dict,
    ):
        sources.append(
            metadata
        )

    return sources


def pds_values(
    item: dict,
    keys: List[str],
) -> List[str]:

    results = []

    for source in pds_sources(
        item
    ):

        for key in keys:

            if key not in source:
                continue

            results.extend(
                flatten_values(
                    source.get(key)
                )
            )

    cleaned = []

    for result in results:

        result = clean_text(
            result
        )

        if (
            result
            and result not in cleaned
        ):
            cleaned.append(
                result
            )

    return cleaned


def pds_first(
    item: dict,
    keys: List[str],
) -> str:

    values = pds_values(
        item,
        keys,
    )

    if values:
        return values[0]

    return ""


# ============================================================
# PDS DESCRIPTION PARSING
# ============================================================

def is_useless_pds_description(
    text: str,
) -> bool:

    lower = text.lower().strip()

    useless_phrases = {
        "migration to pds4",
        "initial version",
        "initial release",
        "updated version",
        "reformatted for pds4",
    }

    if lower in useless_phrases:
        return True

    if len(lower) < 15:
        return True

    return False


def get_pds_description(
    item: dict,
) -> str:
    """
    Prefer actual citation/document descriptions.

    Do NOT feed registry maintenance notes like
    "Migration to PDS4" into INDUS as the primary description.
    """

    priority_groups = [

        [
            "pds:Citation_Information.pds:description",
        ],

        [
            "pds:Document.pds:description",
        ],

        [
            "description",
        ],
    ]

    for keys in priority_groups:

        values = pds_values(
            item,
            keys,
        )

        good_values = [
            value
            for value in values
            if not is_useless_pds_description(
                value
            )
        ]

        if good_values:

            # Prefer reasonably descriptive text,
            # but avoid joining every duplicated description.
            good_values.sort(
                key=len,
                reverse=True,
            )

            return good_values[0]

    return ""


# ============================================================
# PDS URL HELPERS
# ============================================================

def pds_human_page(
    product_class: str,
    lid: str,
    version: str,
) -> str:

    if not lid:
        return ""

    identifier = quote(
        lid,
        safe="",
    )

    version_piece = ""

    if version:

        version_piece = (
            "&version="
            + quote(
                version,
                safe="",
            )
        )

    base = (
        "https://pds.nasa.gov/"
        "ds-view/pds/"
    )

    if product_class == "Product_Bundle":

        return (
            base
            + "viewBundle.jsp?identifier="
            + identifier
            + version_piece
        )

    if product_class == "Product_Collection":

        return (
            base
            + "viewCollection.jsp?identifier="
            + identifier
            + version_piece
        )

    if product_class == "Product_Document":

        return (
            base
            + "viewDocument.jsp?identifier="
            + identifier
            + version_piece
        )

    if (
        product_class
        in {
            "Product_Data_Set",
            "Product_Data_Set_PDS3",
        }
    ):

        return (
            base
            + "viewDataset.jsp?identifier="
            + identifier
        )

    # Generic human-facing PDS search fallback.
    return (
        "https://pds.nasa.gov/"
        "services/search/search?q="
        + identifier
    )


def get_pds_file_urls(
    item: dict,
) -> List[str]:

    urls = pds_values(
        item,
        [
            "ops:Data_File_Info.ops:file_ref",
        ],
    )

    return [
        url
        for url in urls
        if url.startswith(
            ("http://", "https://")
        )
    ]


def get_pds_label_url(
    item: dict,
) -> str:

    urls = pds_values(
        item,
        [
            "ops:Label_File_Info.ops:file_ref",
            "label_url",
        ],
    )

    for url in urls:

        if url.startswith(
            ("http://", "https://")
        ):
            return url

    return ""


def preferred_direct_file(
    file_urls: List[str],
) -> str:
    """
    Prefer a useful scientific/document file over an XML label.
    """

    if not file_urls:
        return ""

    # First choice: PDF
    for url in file_urls:

        if url.lower().endswith(
            ".pdf"
        ):
            return url

    # Next choice: non-XML data resource
    for url in file_urls:

        if not url.lower().endswith(
            ".xml"
        ):
            return url

    return file_urls[0]


# ============================================================
# PDS RESOURCE CLASSIFICATION
# ============================================================

def get_pds_collection_type(
    item: dict,
) -> str:

    return pds_first(
        item,
        [
            "collection_type",
            "pds:Collection.pds:collection_type",
        ],
    )


def pds_is_documentation(
    product_class: str,
    collection_type: str,
) -> bool:

    if product_class == "Product_Document":
        return True

    if (
        product_class == "Product_Collection"
        and collection_type
        and collection_type.lower()
        == "document"
    ):
        return True

    return False


def pds_resource_type(
    product_class: str,
    collection_type: str,
) -> str:

    if product_class == "Product_Document":
        return "PDS Documentation"

    if product_class == "Product_Bundle":
        return "PDS Bundle"

    if product_class == "Product_Collection":

        if (
            collection_type
            and collection_type.lower()
            == "document"
        ):
            return "PDS Documentation Collection"

        return "PDS Dataset / Collection"

    if (
        product_class
        in {
            "Product_Data_Set",
            "Product_Data_Set_PDS3",
        }
    ):
        return "PDS Dataset"

    if product_class == "Product_Resource":
        return "PDS Scientific Resource"

    return "PDS Resource"


# ============================================================
# RESOURCE DEDUPLICATION
# ============================================================

def deduplicate_resources(
    resources: List[dict],
) -> List[dict]:

    seen = set()
    unique = []

    for resource in resources:

        source = resource.get(
            "source",
            "",
        )

        record_id = clean_text(
            str(
                resource.get(
                    "record_id",
                    "",
                )
            )
        ).lower()

        doi = clean_text(
            str(
                resource.get(
                    "doi",
                    "",
                )
            )
        ).lower()

        title_key = re.sub(
            r"[^a-z0-9]",
            "",
            resource.get(
                "title",
                "",
            ).lower(),
        )[:180]

        if record_id:

            key = (
                f"{source}|id|"
                f"{record_id}"
            )

        elif (
            doi
            and doi != "n/a"
        ):

            key = (
                f"doi|{doi}"
            )

        else:

            key = (
                f"{source}|title|"
                f"{title_key}"
            )

        if not title_key:
            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            resource
        )

    return unique


# ============================================================
# TEXT FED TO INDUS
# ============================================================

def resource_text(
    resource: dict,
) -> str:
    """
    This is the scientific text actually fed to INDUS.

    We intentionally do NOT include registry/provenance metadata
    such as checksums, harvest timestamps, package IDs, etc.
    """

    title = resource.get(
        "title",
        "",
    )

    description = resource.get(
        "abstract",
        "",
    )

    context = resource.get(
        "metadata_context",
        "",
    )

    return (
        f"{title}. "
        f"{description}. "
        f"{context}"
    ).strip()


# ============================================================
# RELATIVE DISPLAY LABEL
# ============================================================

def strength_label(
    score: float,
    values: np.ndarray,
) -> str:

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


# ============================================================
# INDUS MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_indus_model():

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


@st.cache_resource(
    show_spinner=False
)
def lens_embeddings():

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

    if not resources:
        return []

    model = load_indus_model()

    texts = [
        resource_text(
            resource
        )
        for resource
        in resources
    ]

    # Encode the user's research question.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # Encode all candidate scientific resources.
    doc_embeddings = model.encode(
        texts,
        batch_size=12,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Because embeddings are normalized,
    # dot product = cosine similarity.
    query_scores = (
        doc_embeddings
        @ query_embedding
    )

    # --------------------------------------------------------
    # DEFAULT: PURE INDUS QUERY SIMILARITY
    # --------------------------------------------------------

    if not use_astro_lens:

        ranked = []

        for idx, resource in enumerate(
            resources
        ):

            item = dict(
                resource
            )

            item[
                "query_score"
            ] = float(
                query_scores[idx]
            )

            item[
                "astro_score"
            ] = None

            item[
                "combined_score"
            ] = float(
                query_scores[idx]
            )

            item[
                "astro_tags"
            ] = []

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda r: r[
                "combined_score"
            ],
            reverse=True,
        )

        return ranked

    # --------------------------------------------------------
    # OPTIONAL ASTROBIOLOGY LENS
    # --------------------------------------------------------

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
        (
            1.0
            - ASTRO_LENS_WEIGHT
        )
        * query_scores
        +
        ASTRO_LENS_WEIGHT
        * astro_scores
    )

    lens_names = list(
        ASTROBIOLOGY_LENSES.keys()
    )

    ranked = []

    for idx, resource in enumerate(
        resources
    ):

        lens_order = np.argsort(
            all_lens_scores[idx]
        )[::-1][:2]

        item = dict(
            resource
        )

        item[
            "query_score"
        ] = float(
            query_scores[idx]
        )

        item[
            "astro_score"
        ] = float(
            astro_scores[idx]
        )

        item[
            "combined_score"
        ] = float(
            combined_scores[idx]
        )

        item[
            "astro_tags"
        ] = [
            lens_names[j]
            for j in lens_order
        ]

        ranked.append(
            item
        )

    ranked.sort(
        key=lambda r: r[
            "combined_score"
        ],
        reverse=True,
    )

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

    endpoint = (
        "https://api.adsabs.harvard.edu/"
        "v1/search/query"
    )

    headers = {
        "Authorization":
        f"Bearer {api_key}"
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
        .get(
            "response",
            {},
        )
        .get(
            "docs",
            [],
        )
    )

    resources = []

    for doc in docs:

        title = doc.get(
            "title",
            [""],
        )

        if (
            isinstance(
                title,
                list,
            )
            and title
        ):

            title = title[0]

        else:

            title = str(
                title
            )

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

            if (
                str(
                    ident
                )
                .lower()
                .startswith(
                    "arxiv:"
                )
            ):

                arxiv_id = (
                    str(
                        ident
                    )
                    .split(
                        ":",
                        1,
                    )[1]
                )

                break

        resources.append(
            {
                "title":
                clean_text(
                    title
                ),

                "authors":
                (
                    doc.get(
                        "author",
                        [],
                    )
                    or []
                ),

                "year":
                int(
                    doc.get(
                        "year"
                    )
                    or 0
                ),

                "abstract":
                abstract,

                "metadata_context":
                "",

                "source":
                "ADS/SciX",

                "resource_type":
                "Publication",

                "record_id":
                bibcode,

                "doi":
                doi,

                "url":
                (
                    "https://ui.adsabs.harvard.edu/"
                    f"abs/{bibcode}/abstract"
                    if bibcode
                    else ""
                ),

                "secondary_url":
                (
                    "https://arxiv.org/abs/"
                    f"{arxiv_id}"
                    if arxiv_id
                    else ""
                ),
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

    terms = informative_terms(
        query
    )

    if not terms:

        terms = [
            query.strip()
        ]

    term_query = " OR ".join(
        f'all:"{term}"'
        for term
        in terms
    )

    date_query = (
        "submittedDate:["
        f"{start_year}01010000 "
        "TO "
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

    for result in client.results(
        search
    ):

        abstract = clean_text(
            result.summary
        )

        if not abstract:
            continue

        resources.append(
            {
                "title":
                clean_text(
                    result.title
                ),

                "authors":
                [
                    author.name
                    for author
                    in result.authors
                ],

                "year":
                result.published.year,

                "abstract":
                abstract,

                "metadata_context":
                "",

                "source":
                "arXiv",

                "resource_type":
                "Preprint",

                "record_id":
                result.entry_id,

                "doi":
                result.doi
                or "",

                "url":
                result.entry_id,

                "secondary_url":
                result.pdf_url
                or "",
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

    endpoint = (
        "https://ntrs.nasa.gov/"
        "api/citations/search"
    )

    params = {
        "q":
        query,

        "publicationDateFrom":
        f"{start_year}-01-01",

        "publicationDateTo":
        f"{end_year}-12-31",

        "page":
        1,

        "pageSize":
        rows,
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

            if not isinstance(
                auth_item,
                dict,
            ):
                continue

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
                authors.append(
                    name
                )

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
                        author[
                            "name"
                        ]
                    )

                elif isinstance(
                    author,
                    str,
                ):

                    authors.append(
                        author
                    )

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
                year_match.group(
                    1
                )
            )
            if year_match
            else 0
        )

        resources.append(
            {
                "title":
                clean_text(
                    doc.get(
                        "title",
                        "",
                    )
                ),

                "authors":
                authors,

                "year":
                year,

                "abstract":
                abstract,

                "metadata_context":
                "",

                "source":
                "NASA NTRS",

                "resource_type":
                (
                    "Technical Report / "
                    "NASA Document"
                ),

                "record_id":
                ntrs_id,

                "doi":
                clean_text(
                    str(
                        doc.get(
                            "doi",
                            "",
                        )
                    )
                ),

                "url":
                (
                    "https://ntrs.nasa.gov/"
                    f"citations/{ntrs_id}"
                    if ntrs_id
                    else ""
                ),

                "secondary_url":
                "",
            }
        )

    return resources


# ============================================================
# NASA PDS RECORD PARSER
# ============================================================

def parse_pds_record(
    item: dict,
    include_data: bool,
    include_documents: bool,
):
    """
    Convert a raw PDS Search API result into the same resource
    format used by ADS, arXiv, and NTRS.
    """

    product_class = pds_first(
        item,
        [
            "product_class",
            (
                "pds:Identification_Area."
                "pds:product_class"
            ),
            "type",
        ],
    )

    # Top-level "type" may be the cleanest result.
    if item.get(
        "type"
    ):

        product_class = clean_text(
            str(
                item[
                    "type"
                ]
            )
        )

    collection_type = get_pds_collection_type(
        item
    )

    is_documentation = pds_is_documentation(
        product_class,
        collection_type,
    )

    # --------------------------------------------------------
    # APPLY USER'S PDS RESOURCE-TYPE SETTINGS
    # --------------------------------------------------------

    if is_documentation:

        if not include_documents:
            return None

    else:

        if not include_data:
            return None

        if (
            product_class
            and product_class
            not in PDS_DATA_TYPES
        ):
            return None

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    lid = pds_first(
        item,
        [
            "lid",
            (
                "pds:Identification_Area."
                "pds:logical_identifier"
            ),
        ],
    )

    lidvid = pds_first(
        item,
        [
            "lidvid",
            "id",
        ],
    )

    version = pds_first(
        item,
        [
            "vid",
            (
                "pds:Identification_Area."
                "pds:version_id"
            ),
            "version",
        ],
    )

    if (
        not version
        and "::"
        in lidvid
    ):

        version = (
            lidvid
            .rsplit(
                "::",
                1,
            )[1]
        )

    if (
        not lid
        and lidvid
    ):

        lid = (
            lidvid
            .split(
                "::",
                1,
            )[0]
        )

    record_id = (
        lidvid
        or lid
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = pds_first(
        item,
        [
            "title",
            (
                "pds:Identification_Area."
                "pds:title"
            ),
            (
                "pds:Document."
                "pds:document_name"
            ),
        ],
    )

    if not title:
        return None

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    description = get_pds_description(
        item
    )

    # --------------------------------------------------------
    # MISSION / TARGET / INSTRUMENT
    # --------------------------------------------------------

    investigations = pds_values(
        item,
        [
            (
                "pds:Investigation_Area."
                "pds:name"
            ),
            "investigations",
        ],
    )

    targets = pds_values(
        item,
        [
            (
                "pds:Target_Identification."
                "pds:name"
            ),
            "targets",
        ],
    )

    instruments = pds_values(
        item,
        [
            (
                "pds:Observing_System_Component."
                "pds:name"
            ),
            "observing_system_components",
        ],
    )

    # --------------------------------------------------------
    # AUTHORS
    # --------------------------------------------------------

    authors = pds_values(
        item,
        [
            (
                "pds:Citation_Information."
                "pds:author_list"
            ),
            (
                "pds:Document."
                "pds:author_list"
            ),
        ],
    )

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    year_text = pds_first(
        item,
        [
            (
                "pds:Citation_Information."
                "pds:publication_year"
            ),
            (
                "pds:Document."
                "pds:publication_date"
            ),
            "publication_year",
        ],
    )

    year_match = re.search(
        r"\b(?:19|20)\d{2}\b",
        year_text,
    )

    year = (
        int(
            year_match.group(0)
        )
        if year_match
        else 0
    )

    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    doi = pds_first(
        item,
        [
            (
                "pds:Citation_Information."
                "pds:doi"
            ),
            (
                "pds:Document."
                "pds:doi"
            ),
            "doi",
        ],
    )

    # --------------------------------------------------------
    # FILE LINKS
    # --------------------------------------------------------

    file_urls = get_pds_file_urls(
        item
    )

    direct_file = preferred_direct_file(
        file_urls
    )

    label_url = get_pds_label_url(
        item
    )

    # --------------------------------------------------------
    # HUMAN-FACING PDS PAGE
    # --------------------------------------------------------

    human_url = pds_human_page(
        product_class,
        lid,
        version,
    )

    # --------------------------------------------------------
    # SCIENCE CONTEXT SENT TO INDUS
    # --------------------------------------------------------

    context_parts = []

    if targets:

        context_parts.append(
            "Targets: "
            + ", ".join(
                targets[:8]
            )
        )

    if investigations:

        context_parts.append(
            "Investigations: "
            + ", ".join(
                investigations[:8]
            )
        )

    if instruments:

        context_parts.append(
            "Observing systems: "
            + ", ".join(
                instruments[:8]
            )
        )

    metadata_context = ". ".join(
        context_parts
    )

    # If no prose description exists,
    # the structured science metadata still gives INDUS
    # something useful to work with.
    if not description:

        description = metadata_context

    if not description:
        return None

    resource_type = pds_resource_type(
        product_class,
        collection_type,
    )

    extra_file_urls = [
        url
        for url in file_urls
        if url != direct_file
    ][:3]

    return {
        "title":
        title,

        "authors":
        authors,

        "year":
        year,

        "abstract":
        description,

        "metadata_context":
        metadata_context,

        "source":
        "NASA PDS",

        "resource_type":
        resource_type,

        "record_id":
        record_id,

        "doi":
        doi,

        # Human-facing PDS page.
        "url":
        human_url,

        # Direct PDF/data file if available.
        "secondary_url":
        direct_file,

        "label_url":
        label_url,

        "extra_file_urls":
        extra_file_urls,

        "pds_product_class":
        product_class,

        "pds_collection_type":
        collection_type,

        "pds_targets":
        targets,

        "pds_investigations":
        investigations,

        "pds_instruments":
        instruments,
    }


# ============================================================
# NASA PDS SEARCH
# ============================================================

def fetch_pds(
    query: str,
    rows: int,
    include_data: bool,
    include_documents: bool,
) -> List[dict]:

    terms = informative_terms(
        query,
        max_terms=6,
    )

    # Try the complete question once too.
    search_queries = [
        query.strip()
    ]

    search_queries.extend(
        terms
    )

    search_queries = list(
        dict.fromkeys(
            [
                q
                for q
                in search_queries
                if q
            ]
        )
    )

    if not search_queries:
        return []

    # Over-fetch because we intentionally discard lots of
    # low-level PDS products.
    per_query_limit = max(
        20,
        min(
            100,
            math.ceil(
                (
                    rows
                    * 4
                )
                /
                len(
                    search_queries
                )
            ),
        ),
    )

    collected = []
    seen = set()

    for search_text in search_queries:

        params = {
            "keywords":
            search_text,

            "fields":
            PDS_FIELDS,

            "limit":
            per_query_limit,
        }

        response = requests.get(
            PDS_SEARCH_ENDPOINT,
            params=params,
            headers={
                "Accept":
                "application/json"
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
                item,
                include_data=include_data,
                include_documents=include_documents,
            )

            if not parsed:
                continue

            key = (
                parsed.get(
                    "record_id"
                )
                or parsed[
                    "title"
                ].lower()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            collected.append(
                parsed
            )

    # Let INDUS decide the final ordering.
    # We keep a larger candidate pool than the requested display count.
    return collected[
        :max(
            rows * 4,
            rows,
        )
    ]


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "🪐 Astrobiology Discovery Explorer"
)

st.caption(
    "Experimental semantic discovery across scientific literature, "
    "planetary data, and NASA technical resources using "
    "NASA-IMPACT's INDUS-SDE-ST model."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Discovery settings"
    )

    st.markdown(
        "**Literature & Technical Sources**"
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

    st.markdown(
        "**Planetary Data System**"
    )

    source_pds_data = st.checkbox(
        "PDS datasets / collections",
        value=True,
        help=(
            "Search PDS bundles, collections, datasets, "
            "and higher-level scientific resources."
        ),
    )

    source_pds_docs = st.checkbox(
        "PDS documentation",
        value=False,
        help=(
            "Also include PDS manuals, user guides, "
            "interface documents, and documentation collections."
        ),
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
            "When unchecked, ranking is based entirely on "
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


# ============================================================
# QUERY BOX
# ============================================================

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


# ============================================================
# SEARCH
# ============================================================

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

    if (
        source_ads
        and not ads_api_key
    ):

        st.info(
            "ADS/SciX is enabled, but no `ADS_API_KEY` is configured "
            "in Streamlit Secrets. This search will continue with "
            "the other selected sources."
        )

    if not any(
        [
            (
                source_ads
                and bool(
                    ads_api_key
                )
            ),
            source_arxiv,
            source_ntrs,
            source_pds_data,
            source_pds_docs,
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

        # ----------------------------------------------------
        # ADS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ARXIV
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # NTRS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PDS
        # ----------------------------------------------------

        if (
            source_pds_data
            or source_pds_docs
        ):

            if (
                source_pds_data
                and source_pds_docs
            ):

                st.write(
                    "Searching NASA PDS datasets, collections, "
                    "and documentation..."
                )

            elif source_pds_data:

                st.write(
                    "Searching NASA PDS datasets "
                    "and collections..."
                )

            else:

                st.write(
                    "Searching NASA PDS documentation..."
                )

            try:

                pds_results = fetch_pds(
                    query,
                    candidates_per_source,
                    include_data=source_pds_data,
                    include_documents=source_pds_docs,
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

        # ----------------------------------------------------
        # DEDUPE
        # ----------------------------------------------------

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
                "Candidate records: "
                + count_text
            )

        st.write(
            f"Collected {len(resources)} unique "
            "scientific resources overall."
        )

        # ----------------------------------------------------
        # INDUS RANKING
        # ----------------------------------------------------

        if resources:

            if use_astro_lens:

                st.write(
                    "Loading INDUS-SDE-ST and ranking all resource "
                    "types with the optional Astrobiology Lens..."
                )

            else:

                st.write(
                    "Loading INDUS-SDE-ST and ranking all resource "
                    "types by semantic similarity to your question..."
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


    # ========================================================
    # WARNINGS
    # ========================================================

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
            "No candidate resources with usable scientific metadata "
            "were retrieved. Try a broader question, a wider "
            "publication-year range, or different sources."
        )

        st.stop()


    # ========================================================
    # RESULT SET
    # ========================================================

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
            f"INDUS reranked {len(ranked)} resources across the "
            "selected sources. The optional Astrobiology Lens is enabled."
        )

    else:

        st.subheader(
            "INDUS semantic results"
        )

        st.caption(
            f"INDUS-SDE-ST reranked {len(ranked)} resources across "
            "the selected sources according to semantic similarity "
            "to your research question."
        )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

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


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # AUTHORS
        # ----------------------------------------------------

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
                    if len(
                        authors
                    ) > 5
                    else ""
                )
            )

            st.markdown(
                f"**Authors:** "
                f"{short_authors}"
            )


        # ----------------------------------------------------
        # PDS SCIENCE METADATA
        # ----------------------------------------------------

        if source == "NASA PDS":

            targets = resource.get(
                "pds_targets",
                [],
            )

            investigations = resource.get(
                "pds_investigations",
                [],
            )

            instruments = resource.get(
                "pds_instruments",
                [],
            )

            if targets:

                st.markdown(
                    "**Target:** "
                    + ", ".join(
                        targets[:6]
                    )
                )

            if investigations:

                st.markdown(
                    "**Mission / Investigation:** "
                    + ", ".join(
                        investigations[:6]
                    )
                )

            if instruments:

                st.markdown(
                    "**Instrument / Observing System:** "
                    + ", ".join(
                        instruments[:6]
                    )
                )


        # ----------------------------------------------------
        # DESCRIPTION
        # ----------------------------------------------------

        with st.expander(
            "Description / Abstract"
        ):

            st.write(
                resource[
                    "abstract"
                ]
            )


        # ----------------------------------------------------
        # SCORE / LINKS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(
            [
                1.2,
                1.4,
                3.0,
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

            # -----------------------------------------------
            # PDS LINKS
            # -----------------------------------------------

            if source == "NASA PDS":

                if resource.get(
                    "url"
                ):

                    links.append(
                        "[View in PDS]"
                        f"({resource['url']})"
                    )

                if resource.get(
                    "secondary_url"
                ):

                    direct_url = resource[
                        "secondary_url"
                    ]

                    if direct_url.lower().endswith(
                        ".pdf"
                    ):

                        links.append(
                            "[Open PDF]"
                            f"({direct_url})"
                        )

                    else:

                        links.append(
                            "[Open data/resource]"
                            f"({direct_url})"
                        )

                extra_files = resource.get(
                    "extra_file_urls",
                    [],
                )

                for index, file_url in enumerate(
                    extra_files[:2],
                    start=1,
                ):

                    if file_url.lower().endswith(
                        ".pdf"
                    ):

                        label_text = (
                            f"Additional PDF {index}"
                        )

                    elif file_url.lower().endswith(
                        ".csv"
                    ):

                        label_text = (
                            f"CSV file {index}"
                        )

                    else:

                        label_text = (
                            f"Additional file {index}"
                        )

                    links.append(
                        f"[{label_text}]"
                        f"({file_url})"
                    )

                if resource.get(
                    "label_url"
                ):

                    links.append(
                        "[PDS metadata label]"
                        f"({resource['label_url']})"
                    )

            # -----------------------------------------------
            # NON-PDS LINKS
            # -----------------------------------------------

            else:

                if resource.get(
                    "url"
                ):

                    links.append(
                        f"[Open {source}]"
                        f"({resource['url']})"
                    )

                if resource.get(
                    "secondary_url"
                ):

                    links.append(
                        "[Alternate / PDF]"
                        f"({resource['secondary_url']})"
                    )


            # -----------------------------------------------
            # DOI
            # -----------------------------------------------

            if resource.get(
                "doi"
            ):

                links.append(
                    "DOI: "
                    f"`{resource['doi']}`"
                )


            if links:

                st.markdown(
                    " &nbsp; | &nbsp; ".join(
                        links
                    )
                )


        # ----------------------------------------------------
        # INDUS EXPLANATION
        # ----------------------------------------------------

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
