from typing import List
import json
from pathlib import Path

def chunk_nested_exercise(exercise: dict) -> List[dict]:
    """
    Chunk a single exercise (nested format) into semantic units.
    """
    chunks = []

    meta = exercise.get("exercise_metadata", {})
    content = exercise.get("exercise_content", {})
    main_data = content.get("main_data", {}) if isinstance(content.get("main_data"), dict) else {}
    sections = content.get("sections", [])

    exercise_id = f"{meta.get('class', '')}_{meta.get('lesson_number', '')}_{meta.get('exercise_number', '')}"

    # Main text
    if isinstance(main_data, dict) and main_data.get("text"):
        chunks.append({
            "exercise_id": exercise_id,
            "section_id": None,
            "section_number": None,
            "type": "main_text",
            "text": main_data["text"]
        })

    # Sections
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


def flatten_exercises(exercises: list) -> List[dict]:
    """
    Recursively flatten nested lists in exercises.
    Ensures that every item is a dictionary before processing.
    """
    flat_list = []
    for ex in exercises:
        if isinstance(ex, list):
            flat_list.extend(flatten_exercises(ex))
        elif isinstance(ex, dict):
            flat_list.append(ex)
        else:
            print(f"Skipping invalid exercise type: {type(ex)}")
    return flat_list


# ---------------- MAIN ----------------
INPUT_FILE = Path("Files/merged_output.json")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            data = [data]
except FileNotFoundError:
    print(f"Error: File {INPUT_FILE} not found.")
    data = []
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in {INPUT_FILE}.")
    data = []

# Flatten nested lists
exercises = flatten_exercises(data)

all_chunks = []
for ex in exercises:
    all_chunks.extend(chunk_nested_exercise(ex))

print(f"Total chunks: {len(all_chunks)}")
for c in all_chunks[:10]:  # Preview first 10 chunks
    print(c['type'], c['text'][:50])
