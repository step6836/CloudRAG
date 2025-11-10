# CloudRAG: Intelligent Earnings Call Analysis System

## Project Overview
A production-grade RAG system for analyzing earnings call transcripts from major cloud SaaS companies. Built for scale with automated data collection, efficient embedding storage, and real-time query capabilities.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  • Web Scraper (investor_scraper.py)                        │
│  • PDF Extractor (pdf_processor.py)                         │
│  • Schedule: Weekly updates, maintains rolling 4 quarters   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  • SQLite Database (transcripts.db)                         │
│    - Raw transcripts table                                  │
│    - Metadata table (date, company, quarter, source URL)    │
│    - Embedding chunks table                                 │
│  • FAISS Vector Store (faiss_index.bin)                     │
│  • Raw Files (transcripts/*.txt, raw_pdfs/*.pdf)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  • Text Chunker (optimal overlap: 200 chars)                │
│  • OpenAI Embeddings (text-embedding-3-small)               │
│  • Embedding Cache (saves $$ on re-runs)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       RAG LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  • FAISS similarity search                                  │
│  • Context assembly with company filtering                  │
│  • GPT-4o-mini generation                                   │
│  • Cost tracking per query                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      API LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  • Flask/FastAPI REST endpoints                             │
│  • /query - Q&A endpoint                                    │
│  • /companies - List available companies                    │
│  • /update - Trigger scraper                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  • React chat interface                                     │
│  • Company filter dropdown                                  │
│  • Real-time streaming responses                            │
│  • Source citation display                                  │
└─────────────────────────────────────────────────────────────┘
```

## Companies Tracked
- Adobe
- Amazon (AWS)
- Microsoft (Azure)
- Oracle
- Salesforce
- Snowflake
- Workday

## Technical Highlights
1. **Cost Optimization**: Embeddings cached to disk → $0 for repeated queries
2. **Scalability**: SQLite → easy migration to PostgreSQL for production
3. **Automation**: Web scraper maintains rolling 4-quarter window
4. **Real-time**: FAISS enables sub-second vector search across 1000+ chunks
5. **Production-Ready**: Proper error handling, logging, and data validation

## Database Schema

### transcripts table
```sql
id              INTEGER PRIMARY KEY
company         TEXT NOT NULL
quarter         TEXT NOT NULL (e.g., 'Q1 2026')
fiscal_year     TEXT NOT NULL (e.g., 'FY2026')
transcript_date DATE
source_url      TEXT
raw_text        TEXT NOT NULL
created_at      TIMESTAMP
updated_at      TIMESTAMP
```

### embedding_chunks table
```sql
id              INTEGER PRIMARY KEY
transcript_id   INTEGER FOREIGN KEY
chunk_index     INTEGER
chunk_text      TEXT NOT NULL
embedding_id    INTEGER (maps to FAISS index position)
created_at      TIMESTAMP
```

### metadata table
```sql
id              INTEGER PRIMARY KEY
last_scrape     TIMESTAMP
total_transcripts INTEGER
total_chunks    INTEGER
embedding_model TEXT
embedding_cost  REAL
```

## Key Metrics
- **Embedding Model**: text-embedding-3-small (1536 dimensions)
- **Chunk Size**: 1000 chars with 200 char overlap
- **Retrieval**: Top-6 chunks per query
- **Cost per Query**: ~$0.005 (half a cent)
- **Initial Embedding Cost**: ~$0.20 for 11 transcripts
- **Storage**: ~50MB for 1000 chunks + embeddings
