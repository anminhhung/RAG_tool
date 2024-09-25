## Base RAG tool

### Table of Contents

1. [**Installation**](#installation)

2. [**PaperAgent**](#data-ingestion)

3. [**Contextual RAG**](#contextual-rag)

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

> Note: Please downgrade to `python3.11` if any conflicts occur.

**3. Install the required dependencies:**

```bash
# Run this only for testing contextual RAG
pip install -r requirements.txt

# Optional
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

### Contextual RAG

#### Additional Installation

After activating your environment, run this:

```bash
bash scripts/contextual_rag_additional_installation.sh
```

#### Test

You should run this to ensure all packages installed successfully !

```bash
pip install pytest

pytest tests/
```

#### Run database

```bash
docker compose up -d
```

#### Config URL for database

In [config/config.yaml](./config/config.yaml), please modify urls of QdrantVectorDB and ElasticSearch:

```yml
...
CONTEXTUAL_RAG:
    ...
    QDRANT_URL: <fill here>

    ELASTIC_SEARCH_URL: <fill here>
```

#### Ingest data

```bash
bash scripts/contextual_rag_ingest.sh both sample/
```

> Note: Please refer to [scripts/contextual_rag_ingest.sh](scripts/contextual_rag_ingest.sh) to change the files dir.

#### Continuos Ingestion

You can add more file paths or even folder paths:

```bash
python src/ingest/add_files.py --type both --files a.pdf b.docx docs/ ...
```

#### Run demo

```bash
python demo_contextual_rag.py --q "Cái gì thất bại đề cử di sản thế giới ?" --compare --debug
```

#### Example Usage:

```python
from src.embedding import RAG
from src.settings import setting

rag = RAG(setting)

q = "Cái gì thất bại đề cử di sản thế giới ?"

print(rag.contextual_rag_search(q))
```
