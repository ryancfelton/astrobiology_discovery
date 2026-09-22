import html
import re
from datetime import datetime, timezone
from typing import Dict, List

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
    "relevant", "using", "use",
}


# -----------------------------
# Helpers
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
    INDUS-SDE-ST performs the actual semantic reranking afterward.
    """
    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9+\-\.]*",
        query.lower()
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


def deduplicate_papers(papers: List[dict]) -> List[dict]:
    seen = set()
    unique = []

    for paper in papers:
        doi = clean_text(str(paper.get("doi", ""))).lower()

        title_key = re.sub(
            r"[^a-z0-9]",
            "",
            paper.get("title", "").lower()
        )[:180]

        key = doi if doi and doi != "n/a" else title_key

        if not key or key in seen:
            continue

        seen.add(key)
        unique.append(paper)

    return unique


def paper_text(paper: dict) -> str:
    """
    Text passed into INDUS-SDE-ST for each retrieved scientific record.
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    return f"{title}. {abstract}".strip()


def strength_label(score: float, values: np.ndarray) -> str:
    """
    Relative label based on the current result set.

    These are not calibrated probabilities.
    """
    if len(values) < 3:
        return "Relevant"

    p70 = float(np.percentile(values, 70))
    p40 = float(np.percentile(values, 40))

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

    SentenceTransformer downloads the model weights the first time the
    Streamlit instance needs them and then keeps the loaded model cached.
    """
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )


@st.cache_resource(show_spinner=False)
def lens_embeddings():
    """
    Convert the optional astrobiology concept descriptions into INDUS
    embeddings.

    This only runs if the user explicitly enables the Astrobiology Lens.
    """
    model = load_indus_model()

    texts = list(ASTROBIOLOGY_LENSES.values())

    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def rank_with_indus(
    papers: List[dict],
    query: str,
    use_astro_lens: bool = False,
) -> List[dict]:
    """
    Rank candidate scientific records using INDUS-SDE-ST.

    Default behavior:
        Rank only according to INDUS semantic similarity between the
        user's research question and each title/abstract.

    Optional Astrobiology Lens:
        If enabled, also compare each record to a small set of
        astrobiology concept descriptions and combine that signal with
        the question similarity.

    The optional lens does not retrain or modify INDUS.
    """
    if not papers:
        return []

    model = load_indus_model()

    texts = [
        paper_text(paper)
        for paper in papers
    ]

    # Convert the user's research question into an INDUS embedding.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # Convert every candidate paper title + abstract into INDUS embeddings.
    doc_embeddings = model.encode(
        texts,
        batch_size=12,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Because the vectors were normalized above, this dot product is
    # equivalent to cosine similarity.
    query_scores = doc_embeddings @ query_embedding

    # ---------------------------------------------------------
    # DEFAULT:
    # Use only INDUS similarity to the user's research question.
    # ---------------------------------------------------------
    if not use_astro_lens:
        ranked = []

        for idx, paper in enumerate(papers):
            item = dict(paper)

            item["query_score"] = float(query_scores[idx])
            item["astro_score"] = None
            item["combined_score"] = float(query_scores[idx])
            item["astro_tags"] = []

            ranked.append(item)

        ranked.sort(
            key=lambda p: p["combined_score"],
            reverse=True
        )

        return ranked

    # ---------------------------------------------------------
    # OPTIONAL EXPERIMENTAL ASTROBIOLOGY LENS
    # ---------------------------------------------------------
    lens_emb = lens_embeddings()

    all_lens_scores = doc_embeddings @ lens_emb.T

    # For each document, use its highest similarity to any of the
    # predefined astrobiology themes.
    astro_scores = np.max(
        all_lens_scores,
        axis=1
    )

    # Experimental weighting:
    # 75% similarity to the actual user question
    # 25% similarity to the Astrobiology Lens
    combined_scores = (
        (1.0 - ASTRO_LENS_WEIGHT) * query_scores
        + ASTRO_LENS_WEIGHT * astro_scores
    )

    lens_names = list(ASTROBIOLOGY_LENSES.keys())

    ranked = []

    for idx, paper in enumerate(papers):
        lens_order = np.argsort(
            all_lens_scores[idx]
        )[::-1][:2]

        item = dict(paper)

        item["query_score"] = float(query_scores[idx])
        item["astro_score"] = float(astro_scores[idx])
        item["combined_score"] = float(combined_scores[idx])

        item["astro_tags"] = [
            lens_names[j]
            for j in lens_order
        ]

        ranked.append(item)

    ranked.sort(
        key=lambda p: p["combined_score"],
        reverse=True
    )

    return ranked


# -----------------------------
# Source retrieval
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

    endpoint = "https://api.adsabs.harvard.edu/v1/search/query"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "q": query,
        "fq": f"year:[{start_year} TO {end_year}]",
        "fl": (
            "title,author,year,pubdate,abstract,"
            "identifier,doi,bibcode,keyword"
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

    papers = []

    for doc in docs:
        title = doc.get("title", [""])

        if isinstance(title, list) and title:
            title = title[0]
        else:
            title = str(title)

        abstract = clean_text(
            doc.get("abstract", "")
        )

        if not abstract:
            continue

        bibcode = doc.get(
            "bibcode",
            ""
        )

        doi_value = doc.get(
            "doi",
            [""]
        )

        if isinstance(doi_value, list) and doi_value:
            doi = doi_value[0]
        else:
            doi = str(doi_value or "")

        identifiers = (
            doc.get("identifier", [])
            or []
        )

        arxiv_id = ""

        for ident in identifiers:
            if str(ident).lower().startswith("arxiv:"):
                arxiv_id = str(ident).split(
                    ":",
                    1
                )[1]
                break

        papers.append(
            {
                "title": clean_text(title),
                "authors": doc.get("author", []) or [],
                "year": int(doc.get("year") or 0),
                "abstract": abstract,
                "source": "ADS/SciX",
                "doi": doi,
                "url": (
                    f"https://ui.adsabs.harvard.edu/"
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

    return papers


def fetch_arxiv(
    query: str,
    start_year: int,
    end_year: int,
    rows: int,
) -> List[dict]:

    terms = informative_terms(query)

    if not terms:
        terms = [query.strip()]

    # Create a deliberately broad candidate set.
    #
    # INDUS-SDE-ST performs the semantic ranking after retrieval.
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
        f"({term_query}) AND {date_query}"
    )

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

    papers = []

    for result in client.results(search):
        abstract = clean_text(
            result.summary
        )

        if not abstract:
            continue

        papers.append(
            {
                "title": clean_text(result.title),
                "authors": [
                    author.name
                    for author in result.authors
                ],
                "year": result.published.year,
                "abstract": abstract,
                "source": "arXiv",
                "doi": result.doi or "",
                "url": result.entry_id,
                "secondary_url": result.pdf_url or "",
            }
        )

    return papers


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
        "publicationDateFrom": f"{start_year}-01-01",
        "publicationDateTo": f"{end_year}-12-31",
        "page": 1,
        "pageSize": rows,
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    docs = data.get(
        "results",
        []
    )

    papers = []

    for doc in docs[:rows]:
        abstract = clean_text(
            doc.get("abstract", "")
        )

        if not abstract:
            continue

        authors = []

        for auth_item in (
            doc.get(
                "authorAffiliations",
                []
            )
            or []
        ):
            if isinstance(auth_item, dict):

                meta = (
                    auth_item.get(
                        "meta",
                        {}
                    )
                    or {}
                )

                author_obj = (
                    meta.get(
                        "author",
                        {}
                    )
                    or {}
                )

                name = (
                    author_obj.get("name")
                    or auth_item.get("name")
                )

                if name:
                    authors.append(name)

        if not authors:
            for author in (
                doc.get("authors", [])
                or []
            ):
                if (
                    isinstance(author, dict)
                    and author.get("name")
                ):
                    authors.append(
                        author["name"]
                    )

                elif isinstance(author, str):
                    authors.append(author)

        ntrs_id = str(
            doc.get(
                "id",
                ""
            )
        )

        pub_date = str(
            doc.get("publicationDate")
            or doc.get("issued")
            or doc.get("created")
            or ""
        )

        year_match = re.match(
            r"(\d{4})",
            pub_date
        )

        year = (
            int(year_match.group(1))
            if year_match
            else 0
        )

        papers.append(
            {
                "title": clean_text(
                    doc.get(
                        "title",
                        ""
                    )
                ),
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "source": "NASA NTRS",
                "doi": clean_text(
                    str(
                        doc.get(
                            "doi",
                            ""
                        )
                    )
                ),
                "url": (
                    f"https://ntrs.nasa.gov/"
                    f"citations/{ntrs_id}"
                    if ntrs_id
                    else ""
                ),
                "secondary_url": "",
            }
        )

    return papers


# -----------------------------
# UI
# -----------------------------
st.title(
    "🪐 Astrobiology Discovery Explorer"
)

st.caption(
    "Experimental semantic discovery across scientific literature "
    "and NASA technical resources using NASA-IMPACT's "
    "INDUS-SDE-ST model."
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

    year_range = st.slider(
        "Publication years",
        min_value=1990,
        max_value=CURRENT_YEAR,
        value=(2015, CURRENT_YEAR),
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
            "75% query similarity + 25% astrobiology-theme similarity."
        )

    st.divider()

    st.caption(
        f"Embedding model: `{MODEL_NAME}`"
    )

    st.caption(
        "Similarity scores are ranking signals, not probabilities."
    )


query = st.text_area(
    "Ask an astrobiology research question",
    placeholder=(
        "Example: What mechanisms could generate methane "
        "in hydrothermal environments on ocean worlds, "
        "and how could we distinguish biological from "
        "abiotic production?"
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
        ""
    )

    if source_ads and not ads_api_key:
        st.info(
            "ADS/SciX is enabled, but no `ADS_API_KEY` "
            "is configured in Streamlit Secrets. "
            "This search will continue with the other "
            "selected sources."
        )

    if not any(
        [
            source_ads and bool(ads_api_key),
            source_arxiv,
            source_ntrs,
        ]
    ):
        st.warning(
            "Enable at least one usable source."
        )
        st.stop()

    papers: List[dict] = []

    errors = []

    with st.status(
        "Gathering candidate science...",
        expanded=True,
    ) as status:

        if source_ads and ads_api_key:

            st.write(
                "Searching ADS/SciX..."
            )

            try:
                papers.extend(
                    fetch_ads(
                        query,
                        ads_api_key,
                        year_range[0],
                        year_range[1],
                        candidates_per_source,
                    )
                )

            except Exception as exc:
                errors.append(
                    f"ADS/SciX: {exc}"
                )

        if source_arxiv:

            st.write(
                "Searching arXiv..."
            )

            try:
                papers.extend(
                    fetch_arxiv(
                        query,
                        year_range[0],
                        year_range[1],
                        candidates_per_source,
                    )
                )

            except Exception as exc:
                errors.append(
                    f"arXiv: {exc}"
                )

        if source_ntrs:

            st.write(
                "Searching NASA NTRS..."
            )

            try:
                papers.extend(
                    fetch_ntrs(
                        query,
                        year_range[0],
                        year_range[1],
                        candidates_per_source,
                    )
                )

            except Exception as exc:
                errors.append(
                    f"NASA NTRS: {exc}"
                )

        papers = deduplicate_papers(
            papers
        )

        st.write(
            f"Collected {len(papers)} "
            f"unique candidate records."
        )

        if papers:

            if use_astro_lens:
                st.write(
                    "Loading INDUS-SDE-ST and ranking candidates "
                    "with the optional Astrobiology Lens..."
                )

            else:
                st.write(
                    "Loading INDUS-SDE-ST and ranking candidates "
                    "by semantic similarity to your question..."
                )

            ranked = rank_with_indus(
                papers,
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
                st.warning(error)


    if not ranked:

        st.warning(
            "No candidate records with abstracts were retrieved. "
            "Try a broader question or year range."
        )

        st.stop()


    top_results = ranked[
        :results_to_show
    ]

    score_values = np.array(
        [
            result["combined_score"]
            for result in top_results
        ],
        dtype=float,
    )


    if use_astro_lens:

        st.subheader(
            "Astrobiology-ranked results"
        )

        st.caption(
            f"INDUS reranked {len(ranked)} candidate records. "
            "The optional Astrobiology Lens is enabled."
        )

    else:

        st.subheader(
            "INDUS semantic results"
        )

        st.caption(
            f"INDUS-SDE-ST reranked {len(ranked)} candidate records "
            "according to semantic similarity to your research question."
        )


    for rank, paper in enumerate(
        top_results,
        start=1,
    ):

        label = strength_label(
            paper["combined_score"],
            score_values,
        )

        source = paper["source"]

        year = (
            paper.get("year")
            or "—"
        )

        tags = " · ".join(
            paper.get(
                "astro_tags",
                []
            )
        )


        st.markdown(
            f"### {rank}. {paper['title']}"
        )


        if use_astro_lens and tags:

            st.caption(
                f"{source} · {year} · "
                f"Astrobiology themes: {tags}"
            )

        else:

            st.caption(
                f"{source} · {year}"
            )


        authors = paper.get(
            "authors",
            []
        )

        if authors:

            short_authors = (
                ", ".join(authors[:5])
                + (
                    " et al."
                    if len(authors) > 5
                    else ""
                )
            )

            st.markdown(
                f"**Authors:** {short_authors}"
            )


        with st.expander(
            "Abstract"
        ):

            st.write(
                paper["abstract"]
            )


        col1, col2, col3 = st.columns(
            [1.2, 1.2, 2.6]
        )


        with col1:

            st.metric(
                "Relevance",
                label,
            )


        with col2:

            st.metric(
                "INDUS score",
                f"{paper['combined_score']:.3f}",
            )


        with col3:

            links = []

            if paper.get("url"):

                links.append(
                    f"[Open {source}]"
                    f"({paper['url']})"
                )

            if paper.get(
                "secondary_url"
            ):

                links.append(
                    f"[Alternate / PDF]"
                    f"({paper['secondary_url']})"
                )

            if paper.get("doi"):

                links.append(
                    f"DOI: `{paper['doi']}`"
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
                f"Similarity to your research question: "
                f"**{paper['query_score']:.3f}**"
            )

            if use_astro_lens:

                st.write(
                    f"Astrobiology-lens similarity: "
                    f"**{paper['astro_score']:.3f}**"
                )

                if tags:

                    st.write(
                        f"Closest astrobiology themes: "
                        f"**{tags}**"
                    )

                st.write(
                    "Final ranking uses **75% query similarity "
                    "+ 25% Astrobiology Lens similarity**."
                )

            else:

                st.write(
                    "The Astrobiology Lens is disabled, so this "
                    "result is ranked entirely from INDUS-SDE-ST "
                    "similarity to your research question."
                )


            st.caption(
                "These values are cosine-similarity ranking signals "
                "in the INDUS embedding space. They are not calibrated "
                "probabilities or expert relevance judgments."
            )


        st.divider()
