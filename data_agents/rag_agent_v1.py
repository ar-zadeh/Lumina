"""
RAG Agent for Querying Extracted Structured Data
Uses embedding-based vector search with persistent storage
"""

import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings

client = OpenAI()

# ============================================================================
# Data Models
# ============================================================================

class ColumnSelection(BaseModel):
    """Selected columns for analysis"""
    selected_columns: List[str] = Field(description="List of column names relevant to the query")
    reasoning: str = Field(description="Why these columns were selected")
    query_type: str = Field(description="Type of query: comparison, aggregation, filtering, search, etc.")

class AnalysisResult(BaseModel):
    """Result of RAG analysis"""
    answer: str = Field(description="Natural language answer to the query")
    relevant_records: List[Dict[str, Any]] = Field(description="Records that were used to answer")
    columns_used: List[str] = Field(description="Columns that were analyzed")
    confidence: float = Field(description="Confidence score 0-1")
    additional_insights: Optional[str] = Field(default=None, description="Additional insights found")

@dataclass
class EmbeddingCache:
    """Cache entry for embeddings"""
    data_hash: str
    embeddings: np.ndarray
    metadata: List[Dict[str, Any]]
    timestamp: datetime
    schema_columns: List[str]

# ============================================================================
# Column Selection Agent
# ============================================================================

class ColumnSelectionAgent:
    """Sub-agent that determines which columns are relevant for a query"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = """You are a data analysis expert. Given a user's question and available data columns,
you must identify which columns are most relevant to answer the question.

Consider:
1. Direct relevance: columns that directly contain the information needed
2. Contextual relevance: columns that provide important context
3. Query type: what kind of analysis is needed (comparison, aggregation, filtering, search)

Be precise and only select columns that will actually help answer the question."""

    def select_columns(
        self, 
        user_query: str, 
        available_columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None
    ) -> ColumnSelection:
        """Determine which columns are relevant for the query"""
        
        # Build column information
        column_info = "Available columns:\n"
        for col in available_columns:
            desc = column_descriptions.get(col, "No description") if column_descriptions else "No description"
            column_info += f"  - {col}: {desc}\n"
        
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""User Query: {user_query}

{column_info}

Which columns should be used to answer this query?"""}
            ],
            response_format=ColumnSelection
        )
        
        return response.choices[0].message.parsed

# ============================================================================
# Embedding Manager with Persistence
# ============================================================================

class EmbeddingManager:
    """Manages embeddings with persistent storage to avoid recalculation"""
    
    def __init__(
        self, 
        cache_dir: str = "./embedding_cache",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_model = embedding_model
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self.cache: Dict[str, EmbeddingCache] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, EmbeddingCache]:
        """Load cached embeddings from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _compute_data_hash(self, data: List[Dict[str, Any]]) -> str:
        """Compute hash of data for cache key"""
        # Simple hash based on data content
        data_str = json.dumps(data, sort_keys=True)
        return str(hash(data_str))
    
    def get_embeddings(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        force_refresh: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Get embeddings for data, using cache if available
        
        Args:
            data: List of records to embed
            columns: Columns to include in embeddings
            force_refresh: Force recomputation even if cached
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        data_hash = self._compute_data_hash(data)
        cache_key = f"{data_hash}_{'-'.join(sorted(columns))}"
        
        # Check cache
        if not force_refresh and cache_key in self.cache:
            cached = self.cache[cache_key]
            print(f"✓ Using cached embeddings from {cached.timestamp}")
            return cached.embeddings, cached.metadata
        
        print(f"Computing embeddings for {len(data)} records...")
        
        # Prepare text for embedding
        texts = []
        metadata = []
        
        for idx, record in enumerate(data):
            # Create text representation using selected columns
            text_parts = []
            for col in columns:
                if col in record and record[col] is not None:
                    text_parts.append(f"{col}: {record[col]}")
            
            text = " | ".join(text_parts)
            texts.append(text)
            
            # Store metadata
            metadata.append({
                'index': idx,
                'record': record,
                'text': text
            })
        
        # Get embeddings from OpenAI
        embeddings = self._embed_texts(texts)
        
        # Cache the results
        self.cache[cache_key] = EmbeddingCache(
            data_hash=data_hash,
            embeddings=embeddings,
            metadata=metadata,
            timestamp=datetime.now(),
            schema_columns=columns
        )
        self._save_cache()
        
        print(f"✓ Computed and cached {len(embeddings)} embeddings")
        
        return embeddings, metadata
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API with batching"""
        # OpenAI allows up to 2048 texts per request
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string"""
        response = client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        return np.array(response.data[0].embedding)
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("✓ Cache cleared")

# ============================================================================
# Vector Store with ChromaDB
# ============================================================================

class VectorStore:
    """Persistent vector store using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def create_collection(
        self, 
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create or get a collection"""
        try:
            # Delete if exists
            self.client.delete_collection(collection_name)
        except:
            pass
        
        return self.client.create_collection(
            name=collection_name,
            metadata=metadata or {}
        )
    
    def add_documents(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """Add documents to collection"""
        collection = self.client.get_or_create_collection(collection_name)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(documents)} documents to collection '{collection_name}'")
    
    def search(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Search collection with query embedding"""
        collection = self.client.get_collection(collection_name)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✓ Deleted collection '{collection_name}'")
        except:
            print(f"Collection '{collection_name}' does not exist")

# ============================================================================
# Main RAG Agent
# ============================================================================

class RAGAgent:
    """
    Main RAG Agent for querying extracted structured data
    Combines column selection, embedding search, and LLM analysis
    """
    
    def __init__(
        self,
        embedding_cache_dir: str = "./embedding_cache",
        vector_store_dir: str = "./chroma_db",
        model: str = "gpt-4o"
    ):
        self.model = model
        self.column_selector = ColumnSelectionAgent(model=model)
        self.embedding_manager = EmbeddingManager(cache_dir=embedding_cache_dir)
        self.vector_store = VectorStore(persist_directory=vector_store_dir)
        self.current_collection = None
        self.current_data = None
        self.current_schema = None
    
    def index_data(
        self,
        data: List[Dict[str, Any]],
        schema_description: Optional[Dict[str, str]] = None,
        collection_name: str = "extraction_data"
    ):
        """
        Index extracted data for querying
        
        Args:
            data: List of extracted records
            schema_description: Optional descriptions of each column
            collection_name: Name for this data collection
        """
        print(f"\n{'='*80}")
        print(f"INDEXING DATA")
        print(f"{'='*80}\n")
        
        if not data:
            raise ValueError("No data to index")
        
        # Store data and schema
        self.current_data = data
        self.current_schema = schema_description or {}
        self.current_collection = collection_name
        
        # Get all columns
        all_columns = list(data[0].keys())
        
        print(f"📊 Indexing {len(data)} records with {len(all_columns)} columns")
        print(f"Columns: {', '.join(all_columns)}\n")
        
        # Get embeddings (uses cache if available)
        embeddings, metadata = self.embedding_manager.get_embeddings(
            data=data,
            columns=all_columns
        )
        
        # Store in vector database
        documents = [m['text'] for m in metadata]
        metadatas = [{'record_json': json.dumps(m['record'])} for m in metadata]
        
        self.vector_store.add_documents(
            collection_name=collection_name,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"\n✓ Data indexed successfully in collection '{collection_name}'")
    
    def query(
        self,
        user_query: str,
        n_results: int = 10,
        collection_name: Optional[str] = None
    ) -> AnalysisResult:
        """
        Query the indexed data
        
        Args:
            user_query: Natural language question
            n_results: Number of similar records to retrieve
            collection_name: Collection to query (uses current if None)
            
        Returns:
            AnalysisResult with answer and supporting data
        """
        collection_name = collection_name or self.current_collection
        
        if not collection_name:
            raise ValueError("No data indexed. Call index_data() first.")
        
        print(f"\n{'='*80}")
        print(f"PROCESSING QUERY")
        print(f"{'='*80}\n")
        print(f"❓ Query: {user_query}\n")
        
        # Step 1: Select relevant columns
        print("🔍 Step 1: Selecting relevant columns...")
        all_columns = list(self.current_data[0].keys())
        
        column_selection = self.column_selector.select_columns(
            user_query=user_query,
            available_columns=all_columns,
            column_descriptions=self.current_schema
        )
        
        print(f"   Selected: {', '.join(column_selection.selected_columns)}")
        print(f"   Reasoning: {column_selection.reasoning}")
        print(f"   Query Type: {column_selection.query_type}\n")
        
        # Step 2: Embed query
        print("🔍 Step 2: Embedding query...")
        query_embedding = self.embedding_manager.embed_query(user_query)
        print("   ✓ Query embedded\n")
        
        # Step 3: Vector search
        print(f"🔍 Step 3: Searching for top {n_results} relevant records...")
        search_results = self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        # Parse results
        relevant_records = []
        for metadata, distance in zip(search_results['metadatas'][0], search_results['distances'][0]):
            record = json.loads(metadata['record_json'])
            record['_similarity_score'] = float(1 - distance)  # Convert distance to similarity
            relevant_records.append(record)
        
        print(f"   ✓ Found {len(relevant_records)} relevant records\n")
        
        # Step 4: Generate answer using LLM
        print("🔍 Step 4: Generating answer...")
        analysis_result = self._generate_answer(
            user_query=user_query,
            relevant_records=relevant_records,
            selected_columns=column_selection.selected_columns,
            query_type=column_selection.query_type
        )
        
        print(f"   ✓ Answer generated (confidence: {analysis_result.confidence:.2f})\n")
        
        return analysis_result
    
    def _generate_answer(
        self,
        user_query: str,
        relevant_records: List[Dict[str, Any]],
        selected_columns: List[str],
        query_type: str
    ) -> AnalysisResult:
        """Generate final answer using LLM with retrieved records"""
        
        # Prepare context with relevant records and columns
        context = self._format_context(relevant_records, selected_columns)
        
        system_prompt = f"""You are a data analysis expert. Answer the user's question based on the provided records.

Query Type: {query_type}
Relevant Columns: {', '.join(selected_columns)}

Guidelines:
1. Be precise and cite specific values from the records
2. If doing comparisons or aggregations, show your work
3. Assign a confidence score (0-1) based on data quality and completeness
4. Provide additional insights if you notice interesting patterns
5. If the data doesn't fully answer the question, be honest about limitations

Format your response as a structured analysis."""

        user_prompt = f"""Question: {user_query}

Relevant Records:
{context}

Please analyze these records and provide a comprehensive answer."""

        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AnalysisResult
        )
        
        return response.choices[0].message.parsed
    
    def _format_context(
        self, 
        records: List[Dict[str, Any]], 
        columns: List[str]
    ) -> str:
        """Format records for LLM context"""
        
        context_parts = []
        for i, record in enumerate(records, 1):
            record_parts = [f"Record {i} (Similarity: {record.get('_similarity_score', 0):.3f}):"]
            for col in columns:
                if col in record and record[col] is not None:
                    record_parts.append(f"  {col}: {record[col]}")
            context_parts.append("\n".join(record_parts))
        
        return "\n\n".join(context_parts)
    
    def batch_query(
        self,
        queries: List[str],
        n_results: int = 10
    ) -> List[AnalysisResult]:
        """Process multiple queries"""
        results = []
        for query in queries:
            result = self.query(query, n_results=n_results)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed data"""
        if not self.current_data:
            return {"error": "No data indexed"}
        
        return {
            "total_records": len(self.current_data),
            "columns": list(self.current_data[0].keys()),
            "collection_name": self.current_collection,
            "cache_entries": len(self.embedding_manager.cache)
        }
    
    def clear_all_caches(self):
        """Clear embedding cache and vector store"""
        self.embedding_manager.clear_cache()
        if self.current_collection:
            self.vector_store.delete_collection(self.current_collection)
        print("✓ All caches cleared")

# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use the RAG Agent"""
    
    # Sample extracted data (from your extraction pipeline)
    extracted_data = [
        {
            "material_name": "Titanium Dioxide Nanoparticles",
            "synthesis_method": "Sol-gel process",
            "temperature": 450.0,
            "temperature_unit": "Celsius",
            "pressure": 1.0,
            "pressure_unit": "atm",
            "reaction_time": 2.5,
            "time_unit": "hours",
            "crystalline_phase": "Anatase",
            "particle_size": 15.2,
            "size_unit": "nm",
            "bandgap": 3.2,
            "bandgap_unit": "eV",
            "application": "Photocatalysis"
        },
        {
            "material_name": "Zinc Oxide Nanowires",
            "synthesis_method": "Hydrothermal synthesis",
            "temperature": 180.0,
            "temperature_unit": "Celsius",
            "pressure": 10.0,
            "pressure_unit": "bar",
            "reaction_time": 6.0,
            "time_unit": "hours",
            "crystalline_phase": "Wurtzite",
            "particle_size": 50.0,
            "size_unit": "nm",
            "bandgap": 3.37,
            "bandgap_unit": "eV",
            "application": "Gas sensing"
        },
        {
            "material_name": "Silicon Nanowires",
            "synthesis_method": "Chemical Vapor Deposition",
            "temperature": 850.0,
            "temperature_unit": "Celsius",
            "pressure": 0.1,
            "pressure_unit": "torr",
            "reaction_time": 1.0,
            "time_unit": "hours",
            "crystalline_phase": "Cubic",
            "particle_size": 100.0,
            "size_unit": "nm",
            "bandgap": 1.1,
            "bandgap_unit": "eV",
            "application": "Electronics"
        },
        {
            "material_name": "Gold Nanoparticles",
            "synthesis_method": "Citrate reduction",
            "temperature": 100.0,
            "temperature_unit": "Celsius",
            "pressure": 1.0,
            "pressure_unit": "atm",
            "reaction_time": 0.5,
            "time_unit": "hours",
            "crystalline_phase": "FCC",
            "particle_size": 20.0,
            "size_unit": "nm",
            "bandgap": None,
            "bandgap_unit": None,
            "application": "Biosensing"
        }
    ]
    
    # Schema descriptions (optional but helpful)
    schema_descriptions = {
        "material_name": "Name of the synthesized material",
        "synthesis_method": "Method used for synthesis",
        "temperature": "Synthesis temperature (numeric value)",
        "temperature_unit": "Unit of temperature measurement",
        "pressure": "Synthesis pressure (numeric value)",
        "pressure_unit": "Unit of pressure measurement",
        "reaction_time": "Duration of reaction (numeric value)",
        "time_unit": "Unit of time measurement",
        "crystalline_phase": "Crystal structure of the material",
        "particle_size": "Size of particles (numeric value)",
        "size_unit": "Unit of size measurement",
        "bandgap": "Electronic bandgap energy (numeric value)",
        "bandgap_unit": "Unit of bandgap measurement",
        "application": "Intended application of the material"
    }
    
    # Initialize RAG Agent
    rag_agent = RAGAgent()
    
    # Index the data
    rag_agent.index_data(
        data=extracted_data,
        schema_description=schema_descriptions,
        collection_name="materials_synthesis"
    )
    
    # Query the data
    queries = [
        "What materials were synthesized at temperatures above 400°C?",
        "Which synthesis method produces the smallest particles?",
        "What is the average bandgap of the materials?",
        "Which materials are suitable for electronic applications?"
    ]
    
    print(f"\n{'='*80}")
    print("RUNNING QUERIES")
    print(f"{'='*80}\n")
    
    for query in queries:
        result = rag_agent.query(query, n_results=5)
        
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        print(f"\n📊 Answer:\n{result.answer}")
        print(f"\n🎯 Confidence: {result.confidence:.2%}")
        print(f"\n📋 Used columns: {', '.join(result.columns_used)}")
        print(f"\n📄 Based on {len(result.relevant_records)} records")
        
        if result.additional_insights:
            print(f"\n💡 Additional insights:\n{result.additional_insights}")
        
        print(f"\n{'─'*80}\n")
    
    # Get statistics
    stats = rag_agent.get_statistics()
    print(f"\n📊 Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    example_usage()