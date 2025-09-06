import json
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INPUT_FILE = Path("Files/merged_lessons.json")  # JSON file with exercises
OUTPUT_FILE = Path("exercise_embeddings.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32

# ---------------- FUNCTIONS ----------------
def chunk_nested_exercise(exercise: dict) -> List[dict]:
    """
    Chunk a single exercise (nested format) into semantic units
    """
    chunks = []

    # Ensure exercise is a dictionary
    if not isinstance(exercise, dict):
        print(f"Skipping invalid exercise format: {type(exercise)}")
        return chunks

    meta = exercise.get("exercise_metadata", {})
    content = exercise.get("exercise_content", {})
    
    # Ensure main_data is a dictionary
    main_data = content.get("main_data") if content.get("main_data") is not None else {}
    sections = content.get("sections", [])

    # Create a unique exercise_id
    exercise_id = f"{meta.get('class','')}_{meta.get('lesson_number','')}_{meta.get('exercise_number','')}"

    # Debug: Print exercise structure
    print(f"Processing exercise: {exercise_id}")
    print(f"main_data: {main_data}")
    print(f"sections count: {len(sections)}")

    # --- Main text chunk ---
    if isinstance(main_data, dict) and main_data.get("text"):
        chunks.append({
            "exercise_id": exercise_id,
            "section_id": None,
            "section_number": None,
            "type": "main_text",
            "text": main_data["text"],
            "metadata": meta
        })

    # --- Section chunks ---
    for sec in sections:
        section_id = sec.get("section_id")
        section_number = sec.get("section_number")

        # Debug: Print section details
        print(f"  Section {section_number}: {sec.keys()}")

        # Section data text
        section_data = sec.get("section_data", {}) if sec.get("section_data") is not None else {}
        if isinstance(section_data, dict) and section_data.get("text"):
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "section_data",
                "text": section_data["text"],
                "metadata": meta
            })

        # Question
        question = sec.get("question", {}) if sec.get("question") is not None else {}
        if isinstance(question, dict) and question.get("text"):
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "question",
                "text": question["text"],
                "metadata": meta
            })

        # Hint
        hint = sec.get("hint", {}) if sec.get("hint") is not None else {}
        if isinstance(hint, dict) and hint.get("text"):
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "hint",
                "text": hint["text"],
                "metadata": meta
            })

        # Solution
        solution = sec.get("solution", {}) if sec.get("solution") is not None else {}
        if isinstance(solution, dict) and solution.get("text"):
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "solution",
                "text": solution["text"],
                "metadata": meta
            })

        # Full Solution
        full_solution = sec.get("full_solution", {}) if sec.get("full_solution") is not None else {}
        if isinstance(full_solution, dict) and full_solution.get("text"):
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "full_solution",
                "text": full_solution["text"],
                "metadata": meta
            })

    return chunks

# ---------------- MAIN PIPELINE ----------------
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

    # Chunk all exercises
    all_chunks = []
    for ex in exercises:
        chunks = chunk_nested_exercise(ex)
        if chunks:
            all_chunks.extend(chunks)
        else:
            print(f"No chunks generated for exercise: {ex.get('exercise_metadata', 'Unknown')}")

    print(f"Total chunks before embedding: {len(all_chunks)}")

    # Initialize embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings in batches
    texts = [chunk["text"] for chunk in all_chunks]
    if not texts:
        print("No valid chunks to process. Exiting.")
        return

    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    # Attach embeddings to chunks
    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embeddings[i].tolist()

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(all_chunks)} chunks. Embeddings saved to {OUTPUT_FILE}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()