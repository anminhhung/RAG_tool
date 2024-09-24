# Base RAG tool

### Installation

To install this application, follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/anminhhung/RAG_tool
cd RAG_tool
```

**2. (Optional) Create and activate a virtual environment:**

-   For Unix/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

-   For Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install the required dependencies:**

```bash
pip install -r requirements.txt
pip install sentence-transformers
pip install -U llama-index llama-index-llms-openai
pip install -U llama-index-vector-stores-chroma
```

### Data Ingestion

**Download data**

-   Download demo data:

```bash
cd data
git clone https://github.com/BachNgoH/AIO_Documents.git
```

**Ingest data**

```bash
python src/ingest/document_ingest.py
```

### Start application

```bash
# backend
uvicorn app:app --reload
# UI
streamlit run streamlit_ui.py

```

### Contextual RAG for papers understanding

#### Download papers

```bash
bash scripts/download_papers.sh
```

#### Run database

```bash
docker compose up -d
```

#### Ingest data

```bash
bash scripts/contextual_rag_ingest.sh
```

#### Run demo

```bash
python demo_contextual_rag.py --q "What are two Types Learnable Text Prompt ?" --compare
```
