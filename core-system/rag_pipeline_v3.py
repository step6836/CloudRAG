"""
rag_pipeline_v3.py
------------------
Improved RAG pipeline with INCREMENTAL embedding updates.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from database import TranscriptDatabase

# Load .env from config folder
config_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(config_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CloudRAGSystem:
    
    def __init__(self):
        # Define paths relative to project root
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        
        # Initialize with organized paths
        self.db = TranscriptDatabase(str(data_dir / "transcripts.db"))
        self.faiss_index_path = str(data_dir / "faiss_index.bin")
        self.chunks_path = str(data_dir / "chunks.npy")
        self.metadata_path = str(data_dir / "metadata.json")
        
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-4o-mini"
        
        # Cost tracking
        self.embedding_cost_per_token = 0.02 / 1_000_000
        self.generation_cost_input = 0.15 / 1_000_000
        self.generation_cost_output = 0.60 / 1_000_000
        self.total_cost = 0.0
        
        # Load or create embeddings
        self._load_or_update_embeddings()
    
    def _load_or_update_embeddings(self):
        
        if Path(self.faiss_index_path).exists():
            print("Loading existing FAISS index from disk (FREE!)...")
            self._load_existing_embeddings()
            
            # Check for new transcripts
            db_chunks = self.db.get_stats()['total_chunks']
            faiss_chunks = self.index.ntotal
            
            if db_chunks > faiss_chunks:
                print(f"\n Found {db_chunks - faiss_chunks} new chunks in database!")
                print("   Adding incremental embeddings (only paying for new data)...\n")
                self._add_new_embeddings()
            elif db_chunks < faiss_chunks:
                print(f"\n Database has fewer chunks than FAISS!")
                print(f"   DB: {db_chunks}, FAISS: {faiss_chunks}")
                print("   Consider running cleanup or rebuild.")
            else:
                print(f" Loaded {faiss_chunks} vectors from {self.db.get_stats()['total_companies']} companies")
        else:
            print(" No FAISS index found. Creating embeddings from scratch...")
            self._create_all_embeddings()
    
    def _load_existing_embeddings(self):
        self.index = faiss.read_index(self.faiss_index_path)
        self.chunks = np.load(self.chunks_path, allow_pickle=True).tolist()
        
        with open(self.metadata_path, "r") as f:
            self.chunk_metadata = json.load(f)
    
    def _add_new_embeddings(self):
        """
        Incremental updates!
        """
        
        # Get all transcript IDs that already have embeddings
        embedded_transcript_ids = set()
        for meta in self.chunk_metadata:
            # Metadata has transcript_id from database
            if 'transcript_id' in meta:
                embedded_transcript_ids.add(meta['transcript_id'])
        
        print(f" Already embedded: {len(embedded_transcript_ids)} transcripts")
        
        # Find transcripts in DB that aren't embedded yet
        all_transcripts = []
        for company in self.db.get_all_companies():
            all_transcripts.extend(self.db.get_transcripts_by_company(company))
        
        new_transcripts = [t for t in all_transcripts 
                          if t['id'] not in embedded_transcript_ids]
        
        if not new_transcripts:
            print(" No new transcripts to embed")
            return
        
        print(f" Found {len(new_transcripts)} new transcripts to embed:")
        for t in new_transcripts:
            print(f"   • {t['company']} {t['quarter']} {t['fiscal_year']}")
        
        # Embed only new transcripts
        new_chunks = []
        new_metadata = []
        
        for transcript in new_transcripts:
            print(f"\n Processing: {transcript['company']} {transcript['quarter']} {transcript['fiscal_year']}")
            
            # Chunk the transcript
            chunks = self._chunk_text(transcript['raw_text'])
            new_chunks.extend(chunks)
            
            # Store metadata
            for chunk in chunks:
                new_metadata.append({
                    'transcript_id': transcript['id'],
                    'company': transcript['company'],
                    'quarter': transcript['quarter'],
                    'fiscal_year': transcript['fiscal_year']
                })
            
            print(f" Created {len(chunks)} chunks")
        
        print(f"\n Creating embeddings for {len(new_chunks)} new chunks...")
        
        # Generate embeddings for new chunks only
        new_embeddings = []
        total_tokens = 0
        
        for i, chunk in enumerate(new_chunks):
            response = client.embeddings.create(
                model=self.embedding_model,
                input=chunk
            )
            new_embeddings.append(response.data[0].embedding)
            total_tokens += response.usage.total_tokens
            
            if (i + 1) % 50 == 0:
                print(f"   Embedded {i+1}/{len(new_chunks)} chunks...")
        
        # Calculate cost
        embedding_cost = total_tokens * self.embedding_cost_per_token
        print(f"\n Incremental embedding cost: ${embedding_cost:.4f} ({total_tokens:,} tokens)")
        
        new_embeddings_array = np.array(new_embeddings).astype('float32')
        
        starting_faiss_position = self.index.ntotal
        
        self.index.add(new_embeddings_array)
        print(f" Added {len(new_embeddings)} vectors to FAISS")
        
        self.chunks.extend(new_chunks)

        self.chunk_metadata.extend(new_metadata)
        
        faiss.write_index(self.index, self.faiss_index_path)
        np.save(self.chunks_path, np.array(self.chunks, dtype=object))
        with open(self.metadata_path, "w") as f:
            json.dump(self.chunk_metadata, f)
        
        print(f" Saved updated embeddings")
        
        # Update database with new chunk mappings
        for transcript in new_transcripts:
            transcript_chunks = [chunk for chunk, meta in zip(new_chunks, new_metadata)
                               if meta['transcript_id'] == transcript['id']]
            
            chunk_count = sum(1 for meta in new_metadata 
                            if meta['transcript_id'] == transcript['id'])
            faiss_positions = list(range(starting_faiss_position, 
                                        starting_faiss_position + chunk_count))
            starting_faiss_position += chunk_count
            
            self.db.insert_embedding_chunks(transcript['id'], 
                                          transcript_chunks, 
                                          faiss_positions)
        
        # Update metadata
        existing_cost = self.db.get_metadata("last_embedding_cost") or 0.0
        self.db.set_metadata("last_embedding_cost", existing_cost + embedding_cost)
        
        print(f" Database updated with {len(new_transcripts)} new transcript mappings\n")
    
    def _create_all_embeddings(self):
        """Create embeddings from scratch (first run only)."""
        print("\n Creating embeddings from scratch...")
        
        # Get all transcripts
        companies = self.db.get_all_companies()
        if not companies:
            print(" No transcripts in database!")
            return
        
        all_chunks = []
        chunk_metadata = []
        
        # Process each company
        for company in companies:
            transcripts = self.db.get_transcripts_by_company(company)
            
            for transcript in transcripts:
                print(f" Processing: {company} {transcript['quarter']} {transcript['fiscal_year']}")
                
                raw_text = transcript['raw_text']
                chunks = self._chunk_text(raw_text)
                
                all_chunks.extend(chunks)
                
                for chunk in chunks:
                    chunk_metadata.append({
                        'transcript_id': transcript['id'],
                        'company': company,
                        'quarter': transcript['quarter'],
                        'fiscal_year': transcript['fiscal_year']
                    })
                
                print(f" Created {len(chunks)} chunks")
        
        print(f"\n Total: {len(all_chunks)} chunks from {len(companies)} companies")
        
        print(f"\n Generating embeddings...")
        embeddings = []
        total_tokens = 0
        
        for i, chunk in enumerate(all_chunks):
            response = client.embeddings.create(
                model=self.embedding_model,
                input=chunk
            )
            embeddings.append(response.data[0].embedding)
            total_tokens += response.usage.total_tokens
            
            if (i + 1) % 50 == 0:
                print(f"   Embedded {i+1}/{len(all_chunks)} chunks...")
        
        # Calculate cost again
        embedding_cost = total_tokens * self.embedding_cost_per_token
        print(f"\n Embedding cost: ${embedding_cost:.4f} ({total_tokens:,} tokens)")
        
        # Store cost in database
        self.db.set_metadata("last_embedding_cost", embedding_cost)
        self.db.set_metadata("last_embedding_date", str(np.datetime64('now')))
        self.db.set_metadata("embedding_model", self.embedding_model)
        
        # Build FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        print(f" Built FAISS index: {embeddings_array.shape}")
        
        # Save everything
        faiss.write_index(self.index, self.faiss_index_path)
        np.save(self.chunks_path, np.array(all_chunks, dtype=object))
        with open(self.metadata_path, "w") as f:
            json.dump(chunk_metadata, f)
        
        self.chunks = all_chunks
        self.chunk_metadata = chunk_metadata
        
        print(f" Saved embeddings to disk")
        
        # Update database with chunk mappings
        faiss_position = 0
        for transcript_id in set(meta['transcript_id'] for meta in chunk_metadata):
            # Get chunks for this transcript
            transcript_chunks = [chunk for chunk, meta in zip(all_chunks, chunk_metadata)
                               if meta['transcript_id'] == transcript_id]
            transcript_positions = list(range(faiss_position, 
                                            faiss_position + len(transcript_chunks)))
            
            # Store in database
            self.db.insert_embedding_chunks(transcript_id, 
                                          transcript_chunks, 
                                          transcript_positions)
            faiss_position += len(transcript_chunks)
        
        print(" Database updated with chunk mappings")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk text with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        
        return chunks
    
    def query(self, question: str, company_filter: Optional[str] = None, 
             top_k: int = 6) -> Dict:
        """Query the RAG system."""
        query_cost = 0.0
        
        # Get question embedding
        q_response = client.embeddings.create(
            model=self.embedding_model,
            input=question
        )
        q_embedding = np.array([q_response.data[0].embedding]).astype('float32')
        query_cost += q_response.usage.total_tokens * self.embedding_cost_per_token
        
        # Search FAISS
        search_k = top_k * 10 if company_filter else top_k
        distances, indices = self.index.search(q_embedding, search_k)
        
        # Get chunks and filter
        context_chunks = []
        sources = []
        
        for idx in indices[0]:
            # Get chunk and metadata
            chunk_text = self.chunks[idx]
            meta = self.chunk_metadata[idx]
            
            # Filter by company if specified
            if company_filter and meta['company'].lower() != company_filter.lower():
                continue
            
            context_chunks.append(chunk_text)
            sources.append({
                "company": meta['company'],
                "quarter": meta['quarter'],
                "fiscal_year": meta['fiscal_year']
            })
            
            if len(context_chunks) >= top_k:
                break
        
        # Build context
        context = "\n\n".join(context_chunks)
        
        # Generate answer with GPT
        response = client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": "You are analyzing earnings call transcripts from major cloud/SaaS companies. Answer based only on the provided context. When relevant, mention which company you're referring to. Be specific with numbers, quotes, and strategic insights."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # Calculate generation cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        query_cost += (input_tokens * self.generation_cost_input + 
                       output_tokens * self.generation_cost_output)
        
        self.total_cost += query_cost
        
        # Unique sources
        unique_sources = []
        seen = set()
        for source in sources:
            key = f"{source['company']}_{source['quarter']}_{source['fiscal_year']}"
            if key not in seen:
                unique_sources.append(source)
                seen.add(key)
        
        return {
            "answer": answer,
            "sources": unique_sources[:3],
            "cost": query_cost,
            "metadata": {
                "chunks_used": len(context_chunks),
                "total_context_length": len(context),
                "model": self.generation_model
            }
        }
    
    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary."""
        embedding_cost = self.db.get_metadata("last_embedding_cost") or 0.0
        
        return {
            "total_embedding_cost": embedding_cost,
            "total_query_cost": self.total_cost,
            "total_cost": embedding_cost + self.total_cost
        }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        db_stats = self.db.get_stats()
        cost_summary = self.get_cost_summary()
        
        return {
            **db_stats,
            **cost_summary,
            "faiss_vectors": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model
        }


if __name__ == "__main__":
    print("="*70)
    print("CloudRAG v3: Incremental Embedding Updates")
    print("="*70)
    
    rag = CloudRAGSystem()
    
    stats = rag.get_system_stats()
    print(f"\n System Statistics:")
    print(f"  Companies: {stats['total_companies']}")
    print(f"  Transcripts: {stats['total_transcripts']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  FAISS vectors: {stats['faiss_vectors']}")
    print(f"\n Total embedding cost: ${stats['total_embedding_cost']:.4f}")
    print("="*70)