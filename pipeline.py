import json
from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import re
import asyncio
import platform
from datetime import datetime

# ---------------- CONFIG ----------------
INPUT_FILE = Path("Files/exercises_schema_v2_2025-09-22.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32
PINECONE_INDEX_NAME = "exercise-embedding"
PINECONE_DIMENSION = 1024

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# Initialize Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

# Memory slots
memory = {
    "has_exam": None,
    "exam_date": None,
    "days_until": None,
    "topics": [],
    "last_lesson": None,
    "preference": None,
    "exercise_counter": 0,
    "preferred_topic": "algebraic representation of a straight line"
}

# State machine functions
async def state_1_opening():
    print("Hey hey, how are you?")
    await asyncio.sleep(30)  # Wait for reply
    print("Just waiting for your answer...")

async def state_2_diagnostic():
    print("Do you have a test coming up?")
    # Simulate user input for demo
    memory["has_exam"] = True
    memory["exam_date"] = "2025-10-04"
    memory["days_until"] = (datetime.fromisoformat(memory["exam_date"]) - datetime.now()).days
    if memory["days_until"] <= 10:
        print("We’ll pick up the pace and increase practice—no stress. Do you have the exam topics?")
    else:
        print("We’ve got time—let’s focus on current topics.")
    print("What did you cover in the last class?")
    memory["last_lesson"] = "linear equations"
    print("What would you like to work on today?")
    memory["preference"] = "linear functions"

async def state_3_learning():
    memory["exercise_counter"] += 1
    if memory["exercise_counter"] <= 2:
        print("Today we’ll focus on representing a line—algebraically, graphically, and with a value table. We’ll start with a slightly easier question and try to progress. Try on your own for a moment; I’m here if you need.")
        # Simulate help ladder
        await asyncio.sleep(5)
        print("Try reading the question again carefully…")
        await asyncio.sleep(5)
        print("Guiding Question 1.")
        await asyncio.sleep(5)
        print("Hint.")
    if memory["exercise_counter"] == 2:
        print("Great! Let’s move to the next exercise.")

async def state_4_summary():
    print("Great session… we covered: line representations (algebraic/graphic), plotting from points…")
    print("Today we reviewed linear functions…")
    print("Great! I’ll send similar exercises… if you get stuck—you’re a genius and I’m here  :)")

# Main function with state machine
async def main():
    await state_1_opening()
    await state_2_diagnostic()
    await state_3_learning()
    await state_4_summary()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())