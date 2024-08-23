# Gemini 1.5 Financial Taxonomy mapping from XLSX files

Sample project for Financial tables extraction using Gemini API LLM and do the taxonomy mapping with Taxonomy data stored in Vertex AI Search for semantic search and use LLM to map to taxonomy code.

## Local Development

### Pre-requiresites

- Python 3.10
- Python venv and pip-tools
- Git
- Vertex AI Search Datastore have been loaded with Taxonomy table
- Set Environment Variables per your environment
  - PROJECT_ID (your project id)
  - LOCATION (default = us-central1)
  - REGION (default = us-central1-b)
  - DATA_STORE_LOCATION (default = global)
  - MAX_DOCUMENTS (default = 5)
  - ENGINE_DATA_TYPE (default = 1 (structured mode))
  - SEARCH_ENGINE_ID (your search engine id)
  - DATA_STORE_ID (your data store id)
  - SEARCH_APP_ID (your search app id)

### Steps

Create a new python venv and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Authentication Google Cloud Services, so app can use your local credentials to access Google Cloud Services

```bash
gcloud auth application-default login
```

Run Gradio app locally

```bash
python ./app/main.py
```

## Deploy to Cloud Run

From root git directory, you can use gcloud cli to deploy to Cloud Run.
Below is example command that will tell Cloud Run to build from source and deploy to Cloud Run Instance.
You can customize other parameters of Cloud Run as needed.

Reference https://cloud.google.com/run/docs/deploying-source-code

```bash
CLOUD_RUN_INSTANCE_NAME=genai-finstmt
gcloud run deploy $CLOUD_RUN_INSTANCE_NAME \
    --source .
```