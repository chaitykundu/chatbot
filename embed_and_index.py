import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from uuid import uuid4
import torch
import re
from chunking import ImprovedExerciseChunker  # Import your chunker

# ---------------- CONFIG ----------------
INPUT_FILE = Path("Files/exercises_schema_v2_2025-09-22.json")  # JSON file with exercises
OUTPUT_FILE = Path("exercise_embedding.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32  # For embedding generation
PINECONE_INDEX_NAME = "exercise-embedding"  # Pinecone index name
PINECONE_DIMENSION = 1024  # Dimension for intfloat/multilingual-e5-large
PINECONE_REGION = "us-east-1"  # Use us-east-1 for free plan support
MAX_PAYLOAD_SIZE = 4 * 1024 * 1024  # 4 MB limit in bytes (unused here, but kept for reference)

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

def sanitize_id(original_id: str, max_length: int = 200) -> str:
    """
    Sanitize ID to ASCII-only for Pinecone compatibility.
    Removes non-ASCII characters and limits length.
    """
    if not original_id:
        return str(uuid4())
    
    # Keep only ASCII alphanumeric, hyphen, underscore, and period
    sanitized = re.sub(r'[^a-zA-Z0-9\-_.]', '', str(original_id))
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Fallback to UUID if empty after sanitization
    if not sanitized:
        sanitized = str(uuid4())
    
    return sanitized

def convert_content_flags(content_flags: Dict[str, bool]) -> List[str]:
    """
    Convert content_flags dictionary to a list of flag names that are True.
    Example: {'has_svg': False, 'has_table': True} -> ['has_table']
    """
    return [key for key, value in content_flags.items() if value]

def embed_and_index_chunks():
    """
    Chunk exercises, generate embeddings, and index into Pinecone.
    """
    # Initialize chunker
    chunker = ImprovedExerciseChunker(max_tokens_per_chunk=1500, min_tokens_per_chunk=300)
    
    # Load exercise data
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found")
    
    with INPUT_FILE.open('r', encoding='utf-8') as f:
        exercises = json.load(f)
    
    print(f"Loaded {len(exercises)} exercises from {INPUT_FILE}")
    
    # Generate chunks
    print("Chunking exercises...")
    chunks = chunker.chunk_exercises(exercises)
    print(f"Created {len(chunks)} chunks")
    
    # Initialize embedding model
    print(f"\nLoading {MODEL_NAME} model...")
    model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare texts with 'passage: ' prefix
    texts = [f"passage: {chunk['searchable_text']}" for chunk in chunks]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON
    
    # Save embedded chunks
    with OUTPUT_FILE.open('w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} embedded chunks to {OUTPUT_FILE}")
    
    # Initialize Pinecone
    print("\nInitializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # List existing indexes for debugging
    existing_indexes = pc.list_indexes().names()
    print(f"Existing indexes: {list(existing_indexes)}")
    
    # Create or connect to Pinecone index
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME} in {PINECONE_REGION}...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=PINECONE_REGION)
            )
            print("Index created successfully!")
        except Exception as e:
            print(f"Error creating index: {e}")
            print("Tip: Ensure you're on the free Starter plan and using us-east-1. Upgrade for other regions.")
            return  # Exit early if creation fails
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Prepare vectors for upsert
    vectors = []
    for i, chunk in enumerate(chunks):
        # Sanitize exercise_id for ASCII-only compatibility
        sanitized_exercise_id = sanitize_id(chunk['exercise_id'])
        vector_id = f"{sanitized_exercise_id}_{uuid4()}"  # Unique ID with sanitized prefix
        
        # Prepare metadata, ensuring it fits within Pinecone limits
        content_flags = chunk.get('content_flags', {})
        metadata = {
            'chunk_type': chunk['chunk_type'],
            'exercise_id': chunk['exercise_id'],  # Keep original in metadata
            'sanitized_exercise_id': sanitized_exercise_id,
            'section_id': str(chunk.get('section_id', '')),
            'section_number': str(chunk.get('section_number', '')),
            'searchable_text': chunk['searchable_text'][:1000],  # Truncate to avoid metadata size limit
            'token_count': chunk['token_count'],
            'content_flags': convert_content_flags(content_flags)  # FIXED: Convert to list of strings
        }
        # Check metadata size (Pinecone has ~40KB limit)
        metadata_size = len(json.dumps(metadata, ensure_ascii=False).encode('utf-8'))
        if metadata_size > 40000:  # Slightly below 40KB to be safe
            metadata['searchable_text'] = metadata['searchable_text'][:500]  # Further truncate if needed
        
        vectors.append({
            'id': vector_id,
            'values': chunk['embedding'],
            'metadata': metadata
        })
        
        if i < 5:  # Debug: Print first few sanitized IDs and content_flags
            print(f"Debug: Original exercise_id='{chunk['exercise_id']}' -> Sanitized='{sanitized_exercise_id}' -> Vector ID='{vector_id[:20]}...'")
            print(f"Debug: Original content_flags={content_flags} -> Converted={metadata['content_flags']}")
    
    # Upsert vectors in batches
    batch_size = 100  # Pinecone batch size
    print(f"Upserting {len(vectors)} vectors to Pinecone...")
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1} of {(len(vectors) + batch_size - 1) // batch_size}")
        except Exception as e:
            print(f"Error upserting batch {i // batch_size + 1}: {e}")
            # Skip batch and continue to diagnose
            continue
    
    print(f"Successfully indexed {len(vectors)} vectors into Pinecone index '{PINECONE_INDEX_NAME}'")
    
    # Verify index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    # Analyze chunking results
    analysis = chunker.analyze_chunking_results(chunks)
    print(f"\nChunking analysis:")
    print(f"Total chunks: {analysis['total_chunks']}")
    print(f"Average tokens per chunk: {analysis['avg_tokens_per_chunk']:.1f}")
    print(f"Token distribution (min/max): {min(analysis['token_distribution'])}/{max(analysis['token_distribution'])}")
    print("Chunk type distribution:")
    for chunk_type, count in analysis['chunk_types'].items():
        print(f"  {chunk_type}: {count}")
    print("Content flags distribution:")
    for flag, count in analysis['content_flags_stats'].items():
        print(f"  {flag}: {count}")

if __name__ == "__main__":
    embed_and_index_chunks()