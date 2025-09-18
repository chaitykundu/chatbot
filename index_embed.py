import json
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import re

# ---------------- CONFIG ----------------
INPUT_FILE = Path("Files/merged_output.json")  # JSON file with exercises
OUTPUT_FILE = Path("exercise_embeddings.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32  # For embedding generation
PINECONE_INDEX_NAME = "exercise-embeddings"  # Replace with your Pinecone index name
PINECONE_DIMENSION = 1024  # Dimension for intfloat/multilingual-e5-large
MAX_PAYLOAD_SIZE = 4 * 1024 * 1024  # 4 MB limit in bytes

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# Function to sanitize string for ASCII-only
def sanitize_ascii(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]', '_', str(text))

# Function to estimate payload size
def estimate_payload_size(vectors: List[dict]) -> int:
    return sum(len(json.dumps(v).encode('utf-8')) for v in vectors)

# ---------------- FUNCTIONS ----------------
def chunk_nested_exercise(exercise: dict) -> List[dict]:
    """
    Chunk a single exercise (nested format) into semantic units.
    """
    chunks = []

    if not isinstance(exercise, dict):
        print(f"Skipping invalid exercise format leopard: {type(exercise)}")
        return chunks

    meta = exercise.get("exercise_metadata", {})
    content = exercise.get("exercise_content", {})
    main_data = content.get("main_data") if content.get("main_data") is not None else {}
    sections = content.get("sections", [])

    exercise_id = sanitize_ascii(
        f"{meta.get('class','')}_{meta.get('lesson_number','')}_{meta.get('exercise_number','')}"
    )

    print(f"Processing exercise: {exercise_id}")
    print(f"main_data: {main_data}")
    print(f"sections count: {len(sections)}")

    has_hints = any(
        isinstance(sec.get("hint"), dict) and sec.get("hint").get("text") is not None
        for sec in sections
    )
    has_solution = any(
        (isinstance(sec.get("solution"), dict) and sec.get("solution").get("text") is not None) or
        (isinstance(sec.get("full_solution"), dict) and sec.get("full_solution").get("text") is not None)
        for sec in sections
    )
    svg_exists = (
        (isinstance(main_data, dict) and main_data.get("svg") is not None) or
        any(
            (isinstance(sec.get("section_data"), dict) and sec.get("section_data").get("svg") is not None) or
            (isinstance(sec.get("question"), dict) and sec.get("question").get("svg") is not None) or
            (isinstance(sec.get("hint"), dict) and sec.get("hint").get("svg") is not None) or
            (isinstance(sec.get("solution"), dict) and sec.get("solution").get("svg") is not None) or
            (isinstance(sec.get("full_solution"), dict) and sec.get("full_solution").get("svg") is not None)
            for sec in sections
        )
    )

    base_metadata = {
        "content_type": meta.get("content_type", "exercise"),
        "exercise_number": str(meta.get("exercise_number", "")),
        "exercise_type": meta.get("exercise_type", "math"),
        "total_sections": len(sections),
        "difficulty": meta.get("difficulty", "medium"),
        "topic": meta.get("topic", "algebra"),
        "grade": meta.get("class", ""),
        "has_hints": has_hints,
        "solution": has_solution,
        "mathematical_concept": meta.get("mathematical_concept", "linear_equations"),
        "retrieval_priority": meta.get("retrieval_priority", 1),
        "svg_exists": svg_exists
    }

    if isinstance(main_data, dict) and main_data.get("text"):
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_type": "main_text",
            "table_headers": main_data.get("table", {}).get("headers", []) if isinstance(main_data.get("table"), dict) else [],
            "rows_data": json.dumps(main_data.get("table", {}).get("rows_data", [])) if isinstance(main_data.get("table"), dict) else "[]"
        })
        chunks.append({
            "exercise_id": exercise_id,
            "section_id": None,
            "section_number": None,
            "type": "main_text",
            "text": main_data["text"],
            "metadata": chunk_metadata
        })

    for sec in sections:
        section_id = sanitize_ascii(str(sec.get("section_id", "unknown")))
        section_number = sec.get("section_number")

        print(f"  Section {section_number}: {sec.keys()}")

        section_data = sec.get("section_data", {}) if sec.get("section_data") is not None else {}
        if isinstance(section_data, dict) and section_data.get("text"):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "section_data",
                "table_headers": section_data.get("table", {}).get("headers", []) if isinstance(section_data.get("table"), dict) else [],
                "rows_data": json.dumps(section_data.get("table", {}).get("rows_data", [])) if isinstance(section_data.get("table"), dict) else "[]"
            })
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "section_data",
                "text": section_data["text"],
                "metadata": chunk_metadata
            })

        question = sec.get("question", {}) if sec.get("question") is not None else {}
        if isinstance(question, dict) and question.get("text"):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "question",
                "table_headers": question.get("table", {}).get("headers", []) if isinstance(question.get("table"), dict) else [],
                "rows_data": json.dumps(question.get("table", {}).get("rows_data", [])) if isinstance(question.get("table"), dict) else "[]"
            })
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "question",
                "text": question["text"],
                "metadata": chunk_metadata
            })

        hint = sec.get("hint", {}) if sec.get("hint") is not None else {}
        if isinstance(hint, dict) and hint.get("text"):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "hint",
                "table_headers": hint.get("table", {}).get("headers", []) if isinstance(hint.get("table"), dict) else [],
                "rows_data": json.dumps(hint.get("table", {}).get("rows_data", [])) if isinstance(hint.get("table"), dict) else "[]"
            })
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "hint",
                "text": hint["text"],
                "metadata": chunk_metadata
            })

        solution = sec.get("solution", {}) if sec.get("solution") is not None else {}
        if isinstance(solution, dict) and solution.get("text"):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "solution",
                "table_headers": solution.get("table", {}).get("headers", []) if isinstance(solution.get("table"), dict) else [],
                "rows_data": json.dumps(solution.get("table", {}).get("rows_data", [])) if isinstance(solution.get("table"), dict) else "[]"
            })
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "solution",
                "text": solution["text"],
                "metadata": chunk_metadata
            })

        full_solution = sec.get("full_solution", {}) if sec.get("full_solution") is not None else {}
        if isinstance(full_solution, dict) and full_solution.get("text"):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "full_solution",
                "table_headers": full_solution.get("table", {}).get("headers", []) if isinstance(full_solution.get("table"), dict) else [],
                "rows_data": json.dumps(full_solution.get("table", {}).get("rows_data", [])) if isinstance(full_solution.get("table"), dict) else "[]"
            })
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "full_solution",
                "text": full_solution["text"],
                "metadata": chunk_metadata
            })

    return chunks

def main():
    # Validate input file existence
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")
    
    # Load JSON (handle dict or list)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    exercises = []
    if isinstance(data, dict):
        exercises.append(data)
    elif isinstance(data, list):
        exercises = data
    else:
        raise ValueError("JSON must be a dict or a list of exercises")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(PINECONE_INDEX_NAME)

    # Chunk all exercises
    all_chunks = []
    for ex in exercises:
        # If ex is a list, flatten it
        if isinstance(ex, list):
            for sub_ex in ex:
                chunks = chunk_nested_exercise(sub_ex)
                if chunks:
                    all_chunks.extend(chunks)
                else:
                    print(f"No chunks generated for exercise: {sub_ex.get('exercise_metadata', 'Unknown')}")
        elif isinstance(ex, dict):
            chunks = chunk_nested_exercise(ex)
            if chunks:
                all_chunks.extend(chunks)
            else:
                print(f"No chunks generated for exercise: {ex.get('exercise_metadata', 'Unknown')}")

    print(f"Total chunks before embedding: {len(all_chunks)}")
    if not all_chunks:
        print("No valid chunks to process. Exiting.")
        return

    # Initialize embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings in batches
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    # Prepare and upsert vectors in batches
    vectors = []
    for i, chunk in enumerate(all_chunks):
        vector_id = sanitize_ascii(
            f"{chunk['exercise_id']}_{chunk['type']}_{chunk.get('section_id', 'main')}_{i}"
        )
        chunk["embedding"] = embeddings[i].tolist()
        metadata = chunk["metadata"].copy()
        # Truncate text in metadata to reduce size (optional)
        metadata["text"] = chunk["text"][:500]  # Limit to 500 characters
        metadata["parent_id"] = chunk["exercise_id"]  # <--- Add this
        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": metadata
        })

    # Batch upsert to stay within payload size limit
    batch = []
    total_size = 0
    for vector in vectors:
        vector_size = len(json.dumps(vector).encode('utf-8'))
        if total_size + vector_size > MAX_PAYLOAD_SIZE and batch:
            index.upsert(vectors=batch)
            print(f"Upserted batch of {len(batch)} vectors, total size: {total_size} bytes")
            batch = [vector]
            total_size = vector_size
        else:
            batch.append(vector)
            total_size += vector_size
    if batch:
        index.upsert(vectors=batch)
        print(f"Upserted final batch of {len(batch)} vectors, total size: {total_size} bytes")

    total_vectors = len(vectors)
    print(f"Upserted {total_vectors} vectors to Pinecone index {PINECONE_INDEX_NAME}")

    # Save to JSON (optional)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(all_chunks)} chunks. Embeddings saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()