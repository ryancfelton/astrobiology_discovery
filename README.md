# Astrobiology Discovery Explorer

A standalone Streamlit prototype that uses NASA-IMPACT's public `nasa-impact/indus-sde-st-v0.2`
model to semantically rank scientific literature and NASA technical resources through an
astrobiology-specific discovery lens.

## What v0.1 does

1. A user asks an astrobiology research question in natural language.
2. The app gathers a broad candidate set from:
   - ADS / SciX (requires an ADS API token)
   - arXiv
   - NASA Technical Reports Server (NTRS)
3. `INDUS-SDE-ST` embeds the question and the candidate titles/abstracts.
4. The same model embeds a small set of explicit astrobiology thematic anchors.
5. Results are ranked using both query similarity and an adjustable **Astrobiology Lens**.
6. Each result shows the closest astrobiology themes and the underlying similarity signals.

This is deliberately a retrieval/discovery prototype: there is no generative LLM and no additional
INDUS fine-tuning in v0.1.

## Repository layout

```text
astrobiology-indus/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first semantic search downloads the INDUS-SDE-ST checkpoint from Hugging Face, so the initial
model load is slower than subsequent searches.

## ADS / SciX token

ADS searches require a Harvard ADS API token. For local use, create `.streamlit/secrets.toml`:

```toml
ADS_API_KEY = "your_ads_api_token_here"
```

Do not commit the real token. arXiv and NTRS searches still work if no ADS token is configured.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository and add these files.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the main file path to `app.py`.
4. In the Streamlit app's **Settings -> Secrets**, add:

```toml
ADS_API_KEY = "your_ads_api_token_here"
```

5. Redeploy after dependency or code changes by pushing to GitHub.

## Current ranking design

If `w` is the selected Astrobiology emphasis:

```text
combined_score = (1 - w) * query_similarity + w * astrobiology_similarity
```

- **Query similarity**: cosine similarity between the user question and each candidate record.
- **Astrobiology similarity**: maximum cosine similarity between the record and the predefined
  astrobiology theme anchors.
- Default `w = 0.25`, so the user's research question remains the dominant signal.

The scores are ranking signals, not calibrated probabilities.

## Suggested next steps

- Add PDS dataset discovery so the tool can connect papers to planetary data.
- Add CMR / NASA open-data resources.
- Cache or pre-index a curated astrobiology corpus for deeper semantic retrieval than live API
  candidate generation can provide.
- Build a small expert-labeled astrobiology retrieval benchmark before considering any fine-tuning.
- Compare the released INDUS-SDE-ST model against ADS/SciX and other retrieval baselines.

## INDUS-SDE-ST

Model: `nasa-impact/indus-sde-st-v0.2`

The model is released by NASA-IMPACT and is described as a sentence transformer for semantic
scientific discovery built on the INDUS-SDE encoder. Consult the model card and NASA-IMPACT
repository for model licensing, training details, evaluation, and the recommended citation.
