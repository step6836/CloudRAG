"""
database.py
-----------
SQLite database schema and operations for storing transcripts and embeddings.
Easily upgradeable to PostgreSQL for production.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import numpy as np

class TranscriptDatabase:
    """Manages transcript storage and retrieval."""
    
    def __init__(self, db_path: str = None):
        # Default to data folder if no path provided
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / "data" / "transcripts.db")
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        cursor = self.conn.cursor()
        
        # Transcripts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                quarter TEXT NOT NULL,
                fiscal_year TEXT NOT NULL,
                transcript_date DATE,
                source_url TEXT,
                raw_text TEXT NOT NULL,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, quarter, fiscal_year)
            )
        """)
        
        # Embedding chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_index_position INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE,
                UNIQUE(transcript_id, chunk_index)
            )
        """)
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_company ON transcripts(company)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quarter ON transcripts(quarter)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcript_id ON embedding_chunks(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faiss_position ON embedding_chunks(faiss_index_position)")
        
        self.conn.commit()
        print(" Database initialized")
    
    def insert_transcript(self, company: str, quarter: str, fiscal_year: str, 
                         raw_text: str, source_url: Optional[str] = None,
                         transcript_date: Optional[str] = None) -> int:
        """Insert a new transcript. Returns transcript_id."""
        cursor = self.conn.cursor()
        
        word_count = len(raw_text.split())
        
        cursor.execute("""
            INSERT OR REPLACE INTO transcripts 
            (company, quarter, fiscal_year, transcript_date, source_url, raw_text, word_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (company, quarter, fiscal_year, transcript_date, source_url, raw_text, word_count))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_embedding_chunks(self, transcript_id: int, chunks: List[str], 
                               faiss_positions: List[int]):
        """Insert embedding chunks for a transcript."""
        cursor = self.conn.cursor()
        
        # Delete existing chunks for this transcript
        cursor.execute("DELETE FROM embedding_chunks WHERE transcript_id = ?", (transcript_id,))
        
        # Insert new chunks
        for chunk_index, (chunk_text, faiss_pos) in enumerate(zip(chunks, faiss_positions)):
            cursor.execute("""
                INSERT INTO embedding_chunks 
                (transcript_id, chunk_index, chunk_text, faiss_index_position)
                VALUES (?, ?, ?, ?)
            """, (transcript_id, chunk_index, chunk_text, faiss_pos))
        
        self.conn.commit()
    
    def get_transcript_by_id(self, transcript_id: int) -> Optional[Dict]:
        """Get a transcript by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_transcripts_by_company(self, company: str) -> List[Dict]:
        """Get all transcripts for a company."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM transcripts 
            WHERE company = ? 
            ORDER BY fiscal_year DESC, quarter DESC
        """, (company,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_companies(self) -> List[str]:
        """Get list of all companies in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT company FROM transcripts ORDER BY company")
        return [row['company'] for row in cursor.fetchall()]
    
    def get_chunk_by_faiss_position(self, faiss_position: int) -> Optional[Dict]:
        """Get chunk and associated transcript info by FAISS index position."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                ec.chunk_text,
                ec.chunk_index,
                t.company,
                t.quarter,
                t.fiscal_year,
                t.transcript_date
            FROM embedding_chunks ec
            JOIN transcripts t ON ec.transcript_id = t.id
            WHERE ec.faiss_index_position = ?
        """, (faiss_position,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_chunks_by_transcript_id(self, transcript_id: int) -> List[Dict]:
        """Get all chunks for a transcript."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM embedding_chunks 
            WHERE transcript_id = ? 
            ORDER BY chunk_index
        """, (transcript_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total transcripts
        cursor.execute("SELECT COUNT(*) as count FROM transcripts")
        stats['total_transcripts'] = cursor.fetchone()['count']
        
        # Total chunks
        cursor.execute("SELECT COUNT(*) as count FROM embedding_chunks")
        stats['total_chunks'] = cursor.fetchone()['count']
        
        # Companies
        cursor.execute("SELECT COUNT(DISTINCT company) as count FROM transcripts")
        stats['total_companies'] = cursor.fetchone()['count']
        
        # Total words
        cursor.execute("SELECT SUM(word_count) as total FROM transcripts")
        stats['total_words'] = cursor.fetchone()['total'] or 0
        
        # By company
        cursor.execute("""
            SELECT company, COUNT(*) as transcript_count, SUM(word_count) as total_words
            FROM transcripts
            GROUP BY company
            ORDER BY company
        """)
        stats['by_company'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def set_metadata(self, key: str, value: any):
        """Store metadata (e.g., last_scrape, embedding_cost)."""
        cursor = self.conn.cursor()
        
        # Convert value to JSON string if not string
        if not isinstance(value, str):
            value = json.dumps(value)
        
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        
        self.conn.commit()
    
    def get_metadata(self, key: str) -> Optional[any]:
        """Get metadata value."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        
        if row:
            try:
                return json.loads(row['value'])
            except:
                return row['value']
        return None
    
    def delete_old_quarters(self, keep_quarters: int = 4):
        """
        Delete old quarters to maintain rolling window.
        Keeps only the most recent N quarters per company.
        """
        cursor = self.conn.cursor()
        
        for company in self.get_all_companies():
            # Get transcript IDs to keep
            cursor.execute("""
                SELECT id FROM transcripts
                WHERE company = ?
                ORDER BY fiscal_year DESC, quarter DESC
                LIMIT ?
            """, (company, keep_quarters))
            
            keep_ids = [row['id'] for row in cursor.fetchall()]
            
            if not keep_ids:
                continue
            
            # Delete old transcripts
            placeholders = ','.join('?' * len(keep_ids))
            cursor.execute(f"""
                DELETE FROM transcripts
                WHERE company = ? AND id NOT IN ({placeholders})
            """, (company, *keep_ids))
        
        self.conn.commit()
        print(f" Cleaned up old quarters, kept {keep_quarters} most recent per company")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Migration script: Convert existing files to database

def migrate_from_files(transcripts_dir: str = None, db_path: str = None):
    """Migrate existing .txt files to database."""
    print(" Starting migration from files to database...")
    
    # Default paths relative to project root
    if transcripts_dir is None:
        project_root = Path(__file__).parent.parent
        transcripts_dir = str(project_root / "data" / "transcripts")
    if db_path is None:
        project_root = Path(__file__).parent.parent
        db_path = str(project_root / "data" / "transcripts.db")
    
    db = TranscriptDatabase(db_path)
    transcripts_path = Path(transcripts_dir)
    
    if not transcripts_path.exists():
        print(" Transcripts directory not found")
        return
    
    files = list(transcripts_path.rglob("*.txt"))
    print(f"Found {len(files)} transcript files")
    
    for file_path in files:
        # Parse filename (e.g., "salesforce_q2_fy26.txt")
        filename = file_path.stem.lower()
        parts = filename.split('_')
        
        if len(parts) >= 3:
            company = parts[0].title()
            quarter = parts[1].upper()
            fiscal_year = parts[2].upper()
        else:
            print(f"  Skipping {filename} (can't parse)")
            continue
        
        # Read transcript
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Insert into database
        transcript_id = db.insert_transcript(
            company=company,
            quarter=quarter,
            fiscal_year=fiscal_year,
            raw_text=raw_text,
            source_url=None
        )
        
        print(f" Migrated: {company} {quarter} {fiscal_year} (ID: {transcript_id})")
    
    db.close()
    print(" Migration complete!")


if __name__ == "__main__":
    # Example usage
    db = TranscriptDatabase()
    
    # Show stats
    stats = db.get_stats()
    print("\n Database Statistics:")
    print(f"Total Transcripts: {stats['total_transcripts']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Companies: {stats['total_companies']}")
    print(f"Total Words: {stats['total_words']:,}")
    
    print("\nBy Company:")
    for company_stat in stats['by_company']:
        print(f"  {company_stat['company']}: {company_stat['transcript_count']} transcripts, {company_stat['total_words']:,} words")
    
    db.close()