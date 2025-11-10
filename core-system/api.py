"""
api.py
------
FastAPI backend for CloudRAG system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from rag_pipeline_v3 import CloudRAGSystem
from database import TranscriptDatabase
from investor_scraper import TranscriptScraper
import uvicorn

app = FastAPI(
    title="CloudRAG API",
    description="Intelligent Earnings Call Analysis System",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
rag_system = CloudRAGSystem()
db = TranscriptDatabase()

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    company_filter: Optional[str] = None
    top_k: int = 6

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    cost: float
    metadata: Dict

class UpdateRequest(BaseModel):
    force_update: bool = False

# Endpoints
@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "online",
        "message": "CloudRAG API is running",
        "version": "1.0.0"
    }

@app.get("/companies")
async def get_companies():
    """Get list of available companies."""
    companies = db.get_all_companies()
    
    # Get transcript count for each
    company_info = []
    for company in companies:
        transcripts = db.get_transcripts_by_company(company)
        company_info.append({
            "name": company,
            "transcript_count": len(transcripts),
            "quarters": [f"{t['quarter']} {t['fiscal_year']}" for t in transcripts]
        })
    
    return {
        "companies": company_info,
        "total": len(companies)
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_system.query(
            question=request.question,
            company_filter=request.company_filter,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = rag_system.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def update_transcripts(request: UpdateRequest):
    """
    Trigger scraper to update transcripts.
    This is a long-running operation - in production, use background tasks.
    """
    try:
        scraper = TranscriptScraper()
        results = scraper.scrape_all_companies(force_update=request.force_update)
        
        # Re-create embeddings if new transcripts added
        successful_scrapes = sum(1 for r in results.values() if r['success'])
        if successful_scrapes > 0:
            rag_system._create_embeddings()
        
        return {
            "status": "success",
            "results": results,
            "message": f"Updated {successful_scrapes} companies"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcripts/{company}")
async def get_company_transcripts(company: str):
    """Get all transcripts for a specific company."""
    try:
        transcripts = db.get_transcripts_by_company(company)
        
        if not transcripts:
            raise HTTPException(status_code=404, detail=f"No transcripts found for {company}")
        
        # Return without full text (too large)
        return {
            "company": company,
            "transcripts": [
                {
                    "id": t['id'],
                    "quarter": t['quarter'],
                    "fiscal_year": t['fiscal_year'],
                    "transcript_date": t['transcript_date'],
                    "word_count": t['word_count'],
                    "source_url": t['source_url'],
                    "created_at": t['created_at']
                }
                for t in transcripts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost")
async def get_cost_summary():
    """Get cost tracking summary."""
    try:
        return rag_system.get_cost_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(" Starting CloudRAG API server...")
    print(" API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)