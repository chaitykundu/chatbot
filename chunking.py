from typing import List
import json
from pathlib import Path

def chunk_nested_exercise(exercise: dict) -> List[dict]:
    """
    Chunk a single exercise (nested format) into semantic units.
    
    Args:
        exercise (dict): A dictionary containing exercise_metadata and exercise_content.
    
    Returns:
        List[dict]: List of chunk dictionaries with text and metadata.
    """
    chunks = []

    meta = exercise.get("exercise_metadata", {})
    content = exercise.get("exercise_content", {})
    # Ensure main_data is a dict, default to empty dict if None or not a dict
    main_data = content.get("main_data", {}) if isinstance(content.get("main_data"), dict) else {}

    sections = content.get("sections", [])

    exercise_id = f"{meta.get('class', '')}_{meta.get('lesson_number', '')}_{meta.get('exercise_number', '')}"

    # 1. Main text
    if isinstance(main_data, dict) and main_data.get("text"):
        chunks.append({
            "exercise_id": exercise_id,
            "section_id": None,
            "section_number": None,
            "type": "main_text",
            "text": main_data["text"]
        })

    # 2. Sections
    for sec in sections:
        section_id = sec.get("section_id")
        section_number = sec.get("section_number")

        # question
        q_text = sec.get("question", {}).get("text")
        if q_text:
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "question",
                "text": q_text
            })

        # hint
        hint_text = sec.get("hint", {}).get("text")
        if hint_text:
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "hint",
                "text": hint_text
            })

        # solution
        sol_text = sec.get("solution", {}).get("text")
        if sol_text:
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "solution",
                "text": sol_text
            })

        # full_solution
        full_sol_text = sec.get("full_solution", {}).get("text")
        if full_sol_text:
            chunks.append({
                "exercise_id": exercise_id,
                "section_id": section_id,
                "section_number": section_number,
                "type": "full_solution",
                "text": full_sol_text
            })

    return chunks

# Example usage
INPUT_FILE = Path("Files/merged_lessons.json")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load JSON directly
        if not isinstance(data, list):
            data = [data]  # Wrap in list if it's a single exercise (dict)
        exercises = data  # Use the loaded data as the list of exercises
except FileNotFoundError:
    print(f"Error: File {INPUT_FILE} not found.")
    exercises = []
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in {INPUT_FILE}.")
    exercises = []

all_chunks = []
for ex in exercises:
    all_chunks.extend(chunk_nested_exercise(ex))

print(f"Total chunks: {len(all_chunks)}")
for c in all_chunks:
    print(c['type'],c['text'][:50])  # Enhanced preview