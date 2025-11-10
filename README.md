# CloudRAG: Intelligent Earnings Call Analysis System

CloudRAG is an automated pipeline using RAG (Retrieval-Augmented Generation) system for analyzing earnings call transcripts from major cloud/SaaS companies. It builds embeddings for fast retrieval and allows interactive queries or API access to extract insights from company earnings calls.

## Project Highlights

- **Scalable Architecture**: SQLite → PostgreSQL ready, FAISS vector store
- **Cost Optimization**: Cached embeddings save $$$ on repeated queries
- **Real-time Updates**: Automated web scraper maintains rolling 4-quarter window
- **Full-Stack**: Python backend + React frontend + REST API
- **Production-Ready**: Proper error handling, logging, cost tracking

## System Architecture

```
Data Collection → Storage → Processing → RAG → API → Frontend
     ↓              ↓          ↓         ↓     ↓       ↓
 Web Scraper → SQLite → Embeddings → FAISS → FastAPI → React
```

## Companies Tracked

- Adobe
- Amazon (AWS)
- Microsoft (Azure)
- Oracle
- Salesforce
- Snowflake
- Workday

## Quick Start

### 1. Installation

```bash
# Clone repo
git clone 
cd cloudrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:
```
OPENAI_API_KEY=your_key_here
```

### 3. Initial Setup

```bash
# Migrate existing transcripts to database (if you have .txt files)
python database.py

# OR scrape fresh transcripts
python investor_scraper.py

# Create embeddings
python rag_pipeline_v2.py
```

### 4. Run the System

**Option A: Interactive CLI**
```bash
python query_interactive.py
```

**Option B: API Server**
```bash
python api.py
# Visit http://localhost:8000/docs for API documentation
```

**Option C: Full Stack (Backend + Frontend)**
```bash
# Terminal 1: Start backend
python api.py

# Terminal 2: Start frontend
cd frontend
npm install
npm start
```

## Project Structure

```
CloudRAG/
├── core-system/
│ ├── rag_pipeline_v3.py     # Core RAG logic
│ ├── database.py            # SQLite ORM and schema
│ ├── investor_scraper.py    # Webscraper logic for transcripts
│ ├── api.py                 # FASTAPI backend
│ ├── demo.py
│
├── documentation/
│ ├── PROJECTSHOWCASE.md     # for recruiters
│ ├── SYSTEM_DIAGRAM.md      # Detailed systems design
│ └── webscraper_info.txt    # webscraper notes
│
├── config/
│ └── .env                   # for your API key, not included
│
├── data/
│ ├── transcripts.db         # SQL database
│ ├── faiss_index.bin        # Vector embeddings
│ ├── chunks.npy             # Connects embedding index for dbs
│ ├── metadata.json
│ └── transcripts/           # Raw transcript .txt files
│
├── venv/
│ └── .gitignore
│
├── README.md
└── requirements.txt         # Python dependencies
└── frontend/                # React app (to be created)
    ├── src/
    ├── public/
    └── package.json
```

##  Usage Examples

### CLI Query
```python
from rag_pipeline_v2 import CloudRAGSystem

rag = CloudRAGSystem()

# Query all companies
result = rag.query("What are the biggest AI challenges mentioned?")

# Query specific company
result = rag.query("What is Azure's revenue growth?", company_filter="microsoft")

print(result['answer'])
print(f"Cost: ${result['cost']:.6f}")
```

### API Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare AI strategies across companies",
    "company_filter": null,
    "top_k": 6
  }'
```

##  Key Features

### 1. Web Scraper (`investor_scraper.py`)
- Handles both HTML and PDF transcripts
- Company-specific IR page parsers
- Maintains rolling 4-quarter window
- Respects rate limits and robots.txt
- Tracks what's been scraped to avoid duplicates

### 2. Database (`database.py`)
- SQLite for local dev (easily upgrades to PostgreSQL)
- Proper foreign keys and indexes
- Stores raw transcripts + metadata
- Maps chunks to FAISS positions
- Migration script from .txt files

### 3. RAG Pipeline (`rag_pipeline_v3.py`)
- OpenAI embeddings (text-embedding-3-small)
- FAISS for fast vector search
- Embedding caching (saves $$$)
- Cost tracking per query
- Company filtering

### 4. API (`api.py`)
- FastAPI with auto-generated docs
- CORS enabled for frontend
- REST endpoints for all operations
- Real-time statistics

## Cost Analysis

| Operation | Cost | Notes |
|-----------|------|-------|
| Initial embedding (11 transcripts) | ~$0.20 | One-time |
| Per query | ~$0.005 | Half a cent |
| Scraper | $0 | Free |
| Storage | $0 | Local SQLite |

**After initial embedding, queries are nearly free!**

## Technical Deep Dive

### Why This Architecture?

1. **SQLite → PostgreSQL Path**: Easy local dev, production-ready
2. **FAISS**: Industry-standard vector search, sub-second queries
3. **Cached Embeddings**: Real-world cost optimization
4. **FastAPI**: Modern Python API framework, auto docs
5. **Company Filtering**: Demonstrates query optimization

### Scalability Considerations

- **Current**: Handles ~1000 chunks, 7 companies
- **Scales to**: 100K+ chunks with FAISS IVF indexes
- **Database**: SQLite → PostgreSQL for production
- **Embeddings**: Can switch to Pinecone/Weaviate for distributed
- **Compute**: Single machine → Kubernetes cluster

## Maintenance

### Update Transcripts
```bash
# Run weekly/monthly (logic)
python investor_scraper.py

# Recreate embeddings (if needed)
python rag_pipeline_v3.py
```

### Clean Old Quarters
```python
from database import TranscriptDatabase
db = TranscriptDatabase()
db.delete_old_quarters(keep_quarters=4)
```

## Future Enhancements

- [ ] Streaming responses for long answers
- [ ] Sentiment analysis per company
- [ ] Time-series visualization of topics
- [ ] Compare company performance metrics
- [ ] Export analysis as PDF reports
- [ ] Multi-language support
- [ ] Fine-tuned embeddings on finance domain

## Troubleshooting

**Q: Embeddings too expensive?**
A: Switch to smaller model or use open-source embeddings (sentence-transformers).

**Q: FAISS index mismatch?**
A: run `rag_pipeline_v3.py` to rebuild, differences will be appended to FAISS

**Q: Database locked?**
A: Close all connections. SQLite doesn't handle concurrent writes well - use PostgreSQL for production.

## License

MIT License

## Author

**Stephanie H** - Psychology → Data Science
- LinkedIn: [(https://www.linkedin.com/in/stephaniehur/)]
- GitHub: [(https://github.com/step6836)]
---

**Built with**: Python, OpenAI, FAISS, FastAPI, SQLite, BeautifulSoup, React
