# chatbot.py (Enhanced AI-Based Math Tutor with Progressive Guidance and Improved SVG Handling)
import os
import re
import json
import random
import time
import threading
from pathlib import Path
from enum import Enum, auto
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer
import logging
import uuid

# Load environment variables
load_dotenv(dotenv_path=Path(".env"))

# Set up logging
#xflogging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = Path("Files/merged_output.json")  # Updated to match bot.py
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# Pinecone Config
INDEX_NAME = "exercise-embeddings"  # Align with index_embed PINECONE_INDEX_NAME
EMBED_DIM = 1024  # Align with index_embed PINECONE_DIMENSION
TOP_K_RETRIEVAL = 20

# Embedding Model Config
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # Align with index_embed MODEL_NAME

# Enhanced Inactivity Settings
INACTIVITY_TIMEOUT = 60  # Increased from 30 seconds
TYPING_DETECTION_THRESHOLD = 10  # Seconds to wait for complete input

# Progressive Guidance Settings
MIN_ATTEMPTS_BEFORE_HINT = 1
MIN_ATTEMPTS_BEFORE_SOLUTION = 2
MAX_GUIDANCE_LEVELS = 3  # 0=encouragement, 1=guiding_question, 2=hint, 3=solution

# -----------------------------
# GenAI Chat Client (using LangChain)
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt template for the RAG chatbot (bilingual support)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Math AI tutor. 
    
    Language Rules:
    - If the question is in Hebrew → respond in Hebrew
    - If the question is in English → respond in English
    - Always match the user's language preference
    - For Hebrew responses, wrap conversational text in Unicode RTL markers (\u202B for start, \u202C for end)
    - Ensure all mathematical expressions and scientific notation are wrapped in LTR markers (\u202A for start, \u202C for end)
    - Example Hebrew response: \u202Bשלום, בוא נתחיל עם התרגיל:\u202C \u202Ay = mx + b\u202C
    
    Teaching Guidelines:
    - Never give direct answers immediately
    - Use a gradual assistance approach: encouragement → guiding questions → hints → solution
    - Ask guiding questions to help students think through problems
    - Build understanding step by step
    - Use the provided context to give accurate information
    - If context lacks crucial information, state what's missing
    - When providing hints, use EXACT TEXT from context when available
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Context: {context}\n\nQuestion: {input}")
])
rag_chain = rag_prompt | llm

# Small talk prompt (bilingual)
small_talk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor starting a conversation.
    
    Language Rules:
    - Detect the user's language and respond in the same language
    - If Hebrew is detected, respond in Hebrew
    - If English is detected, respond in English
    - Default to English if language is unclear
    
    Personality:
    - Warm, encouraging, and approachable
    - Enjoys chatting about hobbies, school, and daily life
    - Enthusiastic about helping with math
    - Keep responses short and conversational (1-2 sentences max) and conversational
    - Understand the user's intent even with spelling mistakes or unclear input
    - Examples: "Hey! How are you doing today?", "Hi there! What's up?", "How's everything going?"
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
small_talk_chain = small_talk_prompt | llm

# Personal follow-up prompt (bilingual)
personal_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are continuing a casual conversation with a student.
    
    Language Rules:
    - Match the user's language (Hebrew or English)
    - Keep the same language throughout the conversation
    
    Guidelines:
    - Acknowledge their response warmly
    - Keep it brief and natural (1-2 sentences)
    - Show genuine interest in personal topics like work, sports, daily life
    - Gradually transition toward academic readiness
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
personal_followup_chain = personal_followup_prompt | llm

# Diagnostic prompt (bilingual)
diagnostic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a math tutor conducting a diagnostic assessment to understand the student's needs.
    
    Language Rules:
    - Match the user's language (Hebrew or English)
    - For Hebrew: Use proper RTL formatting for general text, keep math expressions LTR
    
    Diagnostic Guidelines:
    - Ask one diagnostic question at a time based on the sequence: test, last class, focus
    - Acknowledge the previous response briefly if relevant
    - Examples: "Sounds interesting! Do you have a test coming up?" or "Got it. What did you cover in your last class?"
    - Keep it conversational and encouraging
    - Responses should be short (1-2 sentences max)
    - Transition naturally to the next question or to academic topics after all questions
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Previous diagnostic response: {previous_response}\nCurrent question to ask: {current_question}\nGenerate response:"),
])
diagnostic_chain = diagnostic_prompt | llm

academic_transition_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are transitioning from personal chat to academic topics.
        
    Language Rules:
    - Match the user's language (Hebrew or English)
    - For Hebrew: Use proper RTL formatting for general text, keep math expressions LTR

    Academic Transition Guidelines:
    - Ask about recent learning or upcoming academic events, but AVOID directly asking for grade/class—focus on subjects or interests.
    - Examples: "What did you learn recently?", "When is your next exam?", "How's school going?", "What subjects are you studying?"
    - Bridge from personal to academic naturally
    - Keep it friendly but start showing academic interest
    - Keep responses short (1 sentence)
    - Make the transition feel natural
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
academic_transition_chain = academic_transition_prompt | llm

# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)",
        #"ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
        #"ask_topic": "Great! Grade {grade}. Which topic would you like to practice? (e.g., {topics})",
        "ready_for_question": "Awesome! Let's start with this exercise:",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "Not quite right. Let me help you think through this...",
        "guiding_question": "🤔 Let me ask you this: ",
        "encouragement": "You're making progress — give it try first!",
        "try_again": "Can you try again? Think about your approach.",
        "need_more_attempts": "Give it another try first - I believe you can work through this! {guiding_prompt}",
        "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
        "no_more_hints": "No more hints available. Would you like to see the solution?",
        "no_relevant_exercises": "I couldn't find any relevant exercises for your query.",
        "ask_for_solution": "Would you like me to show you the solution?",
        "irrelevant_msg": "I can only help with math exercises and related questions.",
        "inactivity_check": "Are you still there? I'm here to help whenever you're ready!",
        "session_timeout": "It looks like you stepped away. Feel free to continue whenever you're ready!",
        "ask_for_doubts": "Great work! You've completed several exercises on {topic}. Do you have any questions or doubts about this topic?",
        "no_doubts_response": "Perfect! It looks like you understand {topic} well. Great job today!",
        "doubt_clearing_intro": "Good question! Let me address your question about {topic}:",
        "ask_more_doubts": "Do you have any other questions about {topic}?",
        "small_talk_hobbies": "What hobbies do you have?",
        "diagnostic_test": "Okay! Do you have a test coming up?",
        "diagnostic_last_class": "What did you cover in your last class?",
        "diagnostic_focus": "What would you like to work on today?",
        "doubt_answer_complete": "I hope that helps clarify things about {topic} for you!",
        "lesson_closing": "Great, that was an awesome lesson! I’ll send you similar exercises for practice and see you in the next session. If you have questions, feel free to message me. And if you get stuck – just remember, you’re a genius. Bye!",
        "invalid_topic": "Invalid topic. Please choose one of these:",
        "invalid_class": "Invalid class. Please choose one of these:"  # Add this line

    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)",
        #"ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז, ח)",
        #"ask_topic": "מצוין! כיתה {grade}. באיזה נושא תרצה להתרגל? (לדוגמה: {topics})",
        "ready_for_question": "מעולה! בואו נתחיל עם התרגיל הזה:",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "לא בדיוק נכון. בוא אעזור לך לחשוב על זה...",
        "guiding_question": "🤔 תן לי לשאול אותך את זה: ",
        "encouragement": "אתה מתקדם - תנסה קודם!",
        "try_again": "תוכל לנסות שוב? חשוב על הגישה שלך.",
        "need_more_attempts": "תן לזה עוד ניסיון - אני מאמין שאתה יכול לעבוד על זה!",
        "no_exercises": "לא נמצאו תרגילים עבור כיתה {grade} ונושא {topic}.",
        "no_more_hints": "אין עוד רמזים זמינים. האם תרצה לראות את הפתרון?",
        "no_relevant_exercises": "לא הצלחתי למצוא תרגילים רלוונטיים לשאלתך.",
        "ask_for_solution": "האם תרצה שאראה לך את הפתרון?",
        "irrelevant_msg": "אני יכול לעזור רק עם תרגילי מתמטיקה ושאלות קשורות.",
        "inactivity_check": "אתה עדיין כאן? אני כאן לעזור בכל עת שתהיה מוכן!",
        "session_timeout": "נראה שיצאת לרגע. הרגש בנוח להמשיך בכל עת שתהיה מוכן!",
        "ask_for_doubts": "עבודה מעולה! השלמת מספר תרגילים על {topic}. יש לך שאלות או ספקות על הנושא הזה?",
        "no_doubts_response": "מושלם! נראה שאתה מבין את {topic} היטב. עבודה נהדרת היום!",
        "doubt_clearing_intro": "אני כאן לעזור! תן לי לענות על השאלה שלך על {topic}:",
        "ask_more_doubts": "יש לך שאלות נוספות על {topic}?",
        "small_talk_hobbies": "אילו תחביבים יש לך?",
        "diagnostic_test": "יש לך מבחן בקרוב?",
        "diagnostic_last_class": "מה סקרתם בשיעור האחרון שלך?",
        "diagnostic_focus": "על מה תרצה לעבוד היום?",
        "doubt_answer_complete": "אני מקווה שזה עוזר להבהיר דברים על {topic} עבורך!",
        "lesson_closing": "נהדר, זה היה שיעור מדהים! אשלח לך תרגילים דומים לתרגול וניפגש בשיעור הבא. אם יש לך שאלות, אל תהסס לפנות אליי. ואם תיתקע – זכור, אתה גאון. להתראות!",
        "invalid_topic": "נושא לא חוקי. אנא בחר אחד מהבאים:",
        "invalid_class": "כיתה לא חוקית. אנא בחר אחת מהבאות:"  # Add this line

    }
}

# -----------------------------
# FSM STATES
# -----------------------------
class State(Enum):
    START = auto()
    SMALL_TALK = auto()
    PERSONAL_FOLLOWUP = auto()
    DIAGNOSTIC = auto()
    ACADEMIC_TRANSITION = auto()
    PICK_CLASS = auto()  # New state for picking class/grade
    PICK_TOPIC = auto()  # New state for picking topic
    QUESTION_ANSWER = auto()
    GUIDING_QUESTION = auto()
    PROVIDING_HINT = auto()
    ASK_FOR_DOUBTS = auto()
    DOUBT_CLEARING = auto()

# -----------------------------
# Helper Functions
# -----------------------------
def detect_language(text: str) -> str:
    """Detect if text is Hebrew or English."""
    if any('\u0590' <= char <= '\u05FF' for char in text):
        return "he"
    return "en"

# -----------------------------
# Helper Functions-Cleantext
# -----------------------------
def clean_math_text(text: str) -> str:
    """Remove LaTeX-style $ signs and other delimiters from math expressions."""
    if not text:
        return text
    # Remove single and double dollar signs, wrapping math in LTR Unicode characters
    text = re.sub(r'\$(.*?)\$', '\u202A\\1\u202C', text, flags=re.DOTALL)  # Wrap math in LTR
    text = re.sub(r'\$\$(.*?)\$\$', '\u202A\\1\u202C', text, flags=re.DOTALL)
    
    # Handle fractions with or without curly braces
    def replace_fraction(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return f'\u202A({numerator}/{denominator})\u202C'  # Wrap fractions in LTR
    text = re.sub(r'\\frac\s*\{?([^} ]+)\}?\s*\{?([^} ]+)\}?', replace_fraction, text)
    
    # Remove other LaTeX commands with arguments
    text = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', '\u202A\\2\u202C', text)  # Wrap content in LTR
    # Remove standalone LaTeX commands
    text = re.sub(r'\\([a-zA-Z]+)', r'', text)
    # Normalize whitespace and remove stray $
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('$', '')
    return text.strip()


def translate_text_to_english(text: str) -> str:
    """Translate text (likely Hebrew) to English using GenAI."""
    if not text or not text.strip():
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise translator. Translate the following text to English.
            If it's already in English, return it unchanged with title case (e.g., 'linear functions' -> 'Linear Functions').
            Preserve markdown formatting (e.g., **bold**, *italic*).
            For math expressions, keep them intact (e.g., y = mx + b).
            Provide ONLY the translated text, no extra explanations."""),
            ("user", "{input}")
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        translated = clean_math_text(response.content.strip())
        logger.debug(f"Translation input: '{text}' -> Output: '{translated}'")
        return translated.title()  # Normalize to title case (e.g., "Linear Functions")
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return text.title()  # Return original text in title case as fallback

def is_likely_hebrew(text: str) -> bool:
    """Simple heuristic to check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def load_exercises():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

        # Ensure we always get a flat list of dict exercises
    exercises = []
    if isinstance(data, dict):
        exercises = [data]
    elif isinstance(data, list):
        for ex in data:
            if isinstance(ex, dict):
                exercises.append(ex)
            elif isinstance(ex, list):  # flatten nested lists
                exercises.extend([e for e in ex if isinstance(e, dict)])
    return exercises

all_exercises = load_exercises()

def get_classes():
     # Normalize classes to numbers (e.g., "ז" to "7")
    grade_map = {"ז": "7", "ח": "8", "ט": "9", "י": "10", "יא": "11", "יב": "12"}
    classes = sorted(set(
        grade_map.get(ex["exercise_metadata"]["class"], ex["exercise_metadata"]["class"])
        for ex in load_exercises()
    ))
    return classes

def get_topics(chosen_class):
    # Normalize chosen_class to number
    grade_map = {"ז": "7", "ח": "8", "ט": "9", "י": "10", "יא": "11", "יב": "12"}
    chosen_class = grade_map.get(chosen_class, chosen_class)
    return sorted(set(
        ex["exercise_metadata"]["topic"].title()
        for ex in load_exercises()
        if grade_map.get(ex["exercise_metadata"]["class"], ex["exercise_metadata"]["class"]) == chosen_class
    ))

def get_exercises(chosen_class, chosen_topic):
    # Normalize chosen_class to number
    grade_map = {"ז": "7", "ח": "8", "ט": "9", "י": "10", "יא": "11", "יב": "12"}
    chosen_class = grade_map.get(chosen_class, chosen_class)
    return [
        ex for ex in load_exercises()
        if grade_map.get(ex["exercise_metadata"]["class"], ex["exercise_metadata"]["class"]) == chosen_class
        and ex["exercise_metadata"]["topic"] == chosen_topic
    ]

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedding_model = None

def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(INDEX_NAME)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a given text using SentenceTransformer."""
    if embedding_model is None:
        logger.error("Embedding model not loaded.")
        return []
    try:
        return embedding_model.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def retrieve_relevant_chunks(query: str, pc_index: Any, grade: Optional[str] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from Pinecone based on a query."""
    query = clean_math_text(query)
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []

    filter_dict = {}
    if grade:
        filter_dict["grade"] = {"$eq": grade}
    if topic:
        filter_dict["topic"] = {"$eq": topic}

    try:
        response = pc_index.query(
            vector=query_embedding,
            top_k=TOP_K_RETRIEVAL,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return [match.metadata for match in response.matches]
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {str(e)}", exc_info=True)
        return []

def describe_svg_content(svg_content: str) -> str:
    """Describe SVG content using GenAI."""
    try:
        svg_description_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Provide a CONCISE English description of the main mathematical elements in the SVG (e.g., axes, points, lines, shapes). Do not include the raw SVG code. Provide only a brief description."),
            ("user", "Describe the following SVG content:\n```svg\n{svg_input}\n```"),
        ])
        svg_description_chain = svg_description_prompt | llm
        response = svg_description_chain.invoke({"svg_input": svg_content})
        return clean_math_text(response.content)
    except Exception as e:
        logger.error(f"Error describing SVG content: {str(e)}")
        return "An error occurred while describing the image."
    

def save_svg_to_file(svg_content: str, filename: str) -> Optional[str]:
    """Save SVG content to a file in the svg_outputs folder and return its path."""
    output_dir = Path("svg_outputs")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"SVG saved to {file_path}. Open it in a browser to view.")
        return str(file_path)
    except Exception as e:
        print(f"Error saving SVG: {e}")
        return None

import re

def to_markdown_steps(text: str) -> str:
    """
    Convert text containing 'Step N:' into a markdown numbered list.
    Tolerates leading '**', '>', '-' etc (e.g., '**Step 1:** ...').
    """
    if not text:
        return ""

    # Normalize common bold markers around 'Step N:'
    # e.g., '**Step 1:**' -> 'Step 1:'
    normalized = re.sub(r'\*\*\s*Step\s+(\d+)\s*:\s*\*\*', r'Step \1:', text, flags=re.IGNORECASE)
    normalized = re.sub(r'>\s*Step\s+(\d+)\s*:', r'Step \1:', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^\s*-\s*(Step\s+\d+\s*:)', r'\1', normalized, flags=re.IGNORECASE | re.MULTILINE)

    # Robust pattern: allow spaces and ignore case
    pattern = re.compile(
        r'(?:^|\n)\s*Step\s*(\d+)\s*:\s*(.*?)(?=(?:\n\s*Step\s*\d+\s*:)|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    items = pattern.findall(normalized)

    if not items:
        # No explicit steps—just return the cleaned text (still markdown-friendly)
        return re.sub(r'[ \t]+\n', '\n', normalized).strip()

    lines = []
    for n, content in sorted(((int(n), c) for n, c in items), key=lambda x: x[0]):
        cleaned = re.sub(r"\s+", " ", content).strip()
        lines.append(f"{n}. {cleaned}")

    # Add a blank line before the list so markdown renderers recognize it
    return "\n\n" + "\n".join(lines)



# -----------------------------
# Enhanced Inactivity Timer with Typing Detection
# -----------------------------
class EnhancedInactivityTimer:
    def __init__(self, callback, timeout=INACTIVITY_TIMEOUT):
        self.callback = callback
        self.timeout = timeout
        self.timer = None
        self.last_activity_time = time.time()
        self.typing_detected = False
        
    def start(self):
        self.stop()
        self.timer = threading.Timer(self.timeout, self._check_inactivity)
        self.timer.daemon = True
        self.timer.start()
        
    def stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
            
    def reset(self):
        self.last_activity_time = time.time()
        self.typing_detected = False
        self.start()
    
    def mark_typing(self):
        """Mark that user is typing - prevents premature timeout."""
        self.typing_detected = True
        self.last_activity_time = time.time()
    
    def _check_inactivity(self):
        """Check if user is truly inactive."""
        current_time = time.time()
        time_since_activity = current_time - self.last_activity_time
        
        if self.typing_detected and time_since_activity < TYPING_DETECTION_THRESHOLD:
            self.start()
        elif time_since_activity >= self.timeout:
            self.callback()
        else:
            remaining_time = self.timeout - time_since_activity
            self.timer = threading.Timer(remaining_time, self._check_inactivity)
            self.timer.daemon = True
            self.timer.start()

# -----------------------------
# Enhanced Attempt Tracking
# -----------------------------
class AttemptTracker:
    def __init__(self):
        self.total_attempts = 0
        self.incorrect_attempts = 0
        self.guidance_level = 0
        self.has_requested_hint = False
        self.has_requested_solution = False
        
    def reset(self):
        """Reset for new question."""
        self.total_attempts = 0
        self.incorrect_attempts = 0
        self.guidance_level = 0
        self.has_requested_hint = False
        self.has_requested_solution = False
    
    def record_attempt(self, is_correct: bool):
        """Record an attempt and return if guidance should be offered."""
        self.total_attempts += 1
        if not is_correct:
            self.incorrect_attempts += 1
        
        return not is_correct and self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_HINT
    
    def can_provide_hint(self) -> bool:
        """Check if hint can be provided based on attempts."""
        return (self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_HINT or 
                self.has_requested_hint)
    
    def can_provide_solution(self) -> bool:
        """Check if solution can be provided based on attempts."""
        return (self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_SOLUTION or
                self.has_requested_solution or
                (self.has_requested_hint and self.incorrect_attempts >= 1))
    
    def should_encourage_more_attempts(self, is_hint_request: bool = False, is_solution_request: bool = False) -> bool:
        """Determine if we should encourage more attempts instead of giving help."""
        if is_solution_request and not self.can_provide_solution():
            return True
        if is_hint_request and not self.can_provide_hint():
            return True
        return False
    
    def generate_exercise_with_llm(topic: str, grade: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Generate a new exercise (with SVG) directly using Gemini, without LangChain prompts."""
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")

            # Build plain-text prompt
            prompt = f"""
            You are a Math tutor creating a NEW exercise for grade {grade}, topic: {topic}.
            Language: {"Hebrew" if language == "he" else "English"}

            Output JSON only, no explanations, in this structure:
            {{
              "question": "<short math problem text>",
              "solution": "<step by step solution>",
              "svg": "<valid single <svg> element string>"
            }}

            SVG requirements:
            - viewBox="0 0 400 300"
            - include axes x (0–10) and y (0–10)
            - grid lines
            - plot a blue line with two distinct points labeled
            - ensure well-formed XML <svg>…</svg>
            """

            response = model.generate_content(prompt)
            raw = response.text.strip("` \n").replace("json", "")
            data = json.loads(raw)

            # Wrap into your exercise schema
            ex_num = str(uuid.uuid4())[:8]
            exercise = {
                "exercise_metadata": {
                    "exercise_number": ex_num,
                    "exercise_type": "generated",
                    "class": grade,
                    "grade": grade,
                    "topic": topic,
                    "total_sections": 1
                },
                "exercise_content": {
                    "main_data": {
                        "text": clean_math_text(data.get("question", "")),
                        "svg": data.get("svg")
                    },
                    "sections": [
                        {
                            "section_number": 1,
                            "question": {"text": clean_math_text(data.get("question", ""))},
                            "solution": {"text": clean_math_text(data.get("solution", ""))}
                        }
                    ]
                }
            }
            return exercise

        except Exception as e:
            logger.error(f"❌ Error generating exercise with Gemini: {e}", exc_info=True)
            return None

# -----------------------------
# Enhanced Dialogue FSM
# -----------------------------
class DialogueFSM:
    def __init__(self, exercises_data, pinecone_index,llm):
        self.state = State.START
        self.grade = None
        self.hebrew_grade = None
        self.llm = llm  # Store llm as an instance attribute
        self.exercises_data = exercises_data
        self.pinecone_index = pinecone_index
        self.topic = None
        self.current_exercise = None
        self.current_hint_index = 0
        self.current_question_index = 0
        self.chat_history = []
        self.current_svg_description = None
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 20
        self.small_talk_turns = 0
        self.user_language = "en"
        self.topic_exercises_count = 0   # Track number of completed exercises per topic
        self.MAX_EXERCISES = 2  # Strictly 2 exercises before doubt checking
        
        self.small_talk_chain = small_talk_chain
        self.personal_followup_chain = personal_followup_chain
        self.diagnostic_chain = diagnostic_chain  
        self.academic_transition_chain = academic_transition_chain

        # Enhanced attempt tracking
        self.attempt_tracker = AttemptTracker()
        
        # Doubt clearing functionality
        self.doubt_questions_count = 0
        self.MAX_DOUBT_QUESTIONS = 2
        # Enhanced inactivity timer
        self.inactivity_timer = EnhancedInactivityTimer(self._handle_inactivity)
        self._start_inactivity_timer()
        
        # SVG handling attributes
        self.current_svg_file_path = None  # Track the current SVG file
        self.svg_generated_for_question = False  # Track if SVG was generated for current question
        self.completed_exercises = 0  # New counter for completed exercises
        
        # Student answers recording
        self.student_answers = []

    def _start_inactivity_timer(self):
        """Start or reset the inactivity timer."""
        self.inactivity_timer.reset()

    def _handle_inactivity(self):
        """Handle inactivity timeout - only triggers when truly inactive."""
        lang_dict = I18N[self.user_language]
        
        if self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            self._send_inactivity_message(lang_dict["inactivity_check"])
        else:
            self._send_inactivity_message(lang_dict["session_timeout"])
    
    def _send_inactivity_message(self, message):
          print(f"\nA_GUY (auto): {message}")
          self.chat_history.append(AIMessage(content=message))

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        """Translate numeric grade to Hebrew equivalent."""
        grade_map = {"7": "ז", "8": "ח"}
        return grade_map.get(grade_num, grade_num)

    def _get_localized_text(self, key: str, **kwargs) -> str:
        """Get localized text based on current user language."""
        lang_dict = I18N[self.user_language]
        text = lang_dict.get(key, I18N["en"][key])
        formatted_text = text.format(**kwargs) if kwargs else text
        if self.user_language == "he":
            # Wrap conversational text in RTL markers, exclude math expressions
            formatted_text = f"\u202B{formatted_text}\u202C"
        return formatted_text
    
    def _extract_grade_from_input(self, text: str) -> Optional[str]:
        """Extract grade/class number (7/8 or Hebrew equivalent) from user input using regex."""
        import re

        # Normalize input to handle "grade" or "class"
        text = text.lower().replace("class", "grade").strip()
        
        # English grades: 7 or 8 (case-insensitive, whole word)
        english_match = re.search(r'\b(7|8)\b', text, re.IGNORECASE)
        if english_match:
            return english_match.group(1)
        
        # Hebrew grades: ז (7), ח (8) - using the existing map
        hebrew_map = {"ז": "7", "ח": "8"}  # Matches _translate_grade_to_hebrew
        hebrew_match = re.search(r'(ז|ח)', text)
        if hebrew_match:
            hebrew_char = hebrew_match.group(1)
            return hebrew_map.get(hebrew_char, None)
        
        # Handle phrases like "grade 7", "class 7", "כיתה ז"
        phrase_match = re.search(r'(?:grade|class|כיתה)\s*(\d+|ז|ח)', text, re.IGNORECASE)
        if phrase_match:
            value = phrase_match.group(1)
            return hebrew_map.get(value, value) if value in hebrew_map else value if value in ["7", "8"] else None
        
        return None
    
    # Add the new function here or with other helper methods
    def _get_grade_key(self, metadata: dict) -> Optional[str]:
        """
        Retrieves the grade from exercise metadata, checking for 'grade' and 'class' keys.
        """
        # Prioritize the 'grade' key if it exists, otherwise fall back to 'class'
        grade_key = metadata.get("grade")
        if grade_key is None:
            grade_key = metadata.get("class")
        return grade_key
    
    def _extract_metadata(self, exercise: Any) -> Optional[Dict]:
        # Assuming this is the truncated part, I'll assume it's defined or add placeholder
        # For completeness, let's assume it's to extract metadata
        return exercise.get("exercise_metadata", None)
    
    def _save_svg(self, svg_content: str) -> Optional[str]:
        """Save SVG content to a file and return its path, skipping if identical content already exists."""
        if not svg_content:
            return None
        filename = f"exercise_{self.current_exercise['exercise_metadata']['exercise_number']}_q{self.current_question_index}.svg"
        file_path = SVG_OUTPUT_DIR / filename  # Assuming SVG_OUTPUT_DIR is a Path object
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                existing_content = f.read().strip()
            if existing_content == svg_content.strip():
                logger.debug(f"SVG already saved and identical, skipping: {file_path}")
                return str(file_path)  # Return the path without re-saving or printing
        # If file doesn't exist or content differs, save it
        return save_svg_to_file(svg_content, filename)

    def _get_current_question(self) -> str:
        if not self.current_exercise or "exercise_content" not in self.current_exercise:
            return self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)

        try:
            meta = self.current_exercise.get("exercise_metadata", {})
            content = self.current_exercise.get("exercise_content", {})
            sections = content.get("sections", [])

            if self.current_question_index >= len(sections):
                logger.warning(f"Question index {self.current_question_index} exceeds sections length {len(sections)}")
                return ""

            section = sections[self.current_question_index]
            
            # DEBUG: Print available fields in the section to see what's actually there
            if self.current_question_index == 0:  # Only debug for first question
                logger.debug(f"Available section fields: {list(section.keys())}")
                if "question" in section:
                    logger.debug(f"Question fields: {list(section['question'].keys())}")
            
            main_data = content.get("main_data", {})
            main_text = clean_math_text(main_data.get("text", ""))
            
            # ✅ FIXED: Handle exercise metadata with translation
            exercise_number = meta.get('exercise_number', 'N/A')
            exercise_type_raw = meta.get('exercise_type', 'Unknown')
            
            # Translate exercise type immediately if needed
            exercise_type = exercise_type_raw
            if self.user_language == "en" and is_likely_hebrew(exercise_type_raw):
                exercise_type = translate_text_to_english(exercise_type_raw)
                logger.debug(f"Translated exercise type: '{exercise_type_raw}' -> '{exercise_type}'")
            
            # Translate main text if needed
            if main_text and self.user_language == "en" and is_likely_hebrew(main_text):
                main_text = translate_text_to_english(main_text)
            
            # Check for different possible base/guide question field names
            base_question = None
            guide_question = None
            
            # Try different possible field names for base/guide questions
            possible_base_names = ["base_question", "baseQuestion", "base", "guide_question", "guideQuestion", "guide"]
            possible_guide_names = ["guide_question", "guideQuestion", "guide", "intro_question", "introQuestion"]
            
            for field_name in possible_base_names:
                if field_name in section:
                    base_question = section[field_name].get("text") if isinstance(section[field_name], dict) else section[field_name]
                    logger.debug(f"Found base question in field: {field_name}")
                    break
                    
            for field_name in possible_guide_names:
                if field_name in section and not base_question:  # Only use guide if no base found
                    guide_question = section[field_name].get("text") if isinstance(section[field_name], dict) else section[field_name]
                    logger.debug(f"Found guide question in field: {field_name}")
                    break
            
            regular_question = section.get("question", {}).get("text")
            
            # FIXED: Always prioritize base_question for the FIRST question (index 0)
            # regardless of availability, and ensure we start with base questions
            if self.current_question_index == 0:
                # For the first question, ALWAYS try base question first
                if base_question:
                    q_text = clean_math_text(base_question)
                    question_type = "Base Question"
                    logger.debug("Using base question for first exercise")
                elif guide_question:
                    q_text = clean_math_text(guide_question)
                    question_type = "Guide Question"
                    logger.debug("Using guide question for first exercise (no base found)")
                else:
                    q_text = clean_math_text(regular_question or "")
                    question_type = "Question"
                    logger.debug("Using regular question for first exercise (no base/guide found)")
            else:
                # For subsequent questions, use regular question
                q_text = clean_math_text(regular_question or "")
                question_type = "Question"

            # Translate question text if needed
            if q_text and self.user_language == "en" and is_likely_hebrew(q_text):
                q_text = translate_text_to_english(q_text)

            # Translate section number if needed
            section_number = section.get('section_number', 'N/A')
            if self.user_language == "en" and is_likely_hebrew(str(section_number)):
                hebrew_to_number = {"א": "1", "ב": "2", "ג": "3", "ד": "4", "ה": "5"}
                section_number = hebrew_to_number.get(str(section_number), section_number)

            # ✅ FIXED: Format question output with already translated components
            formatted_question = f"📘 Exercise {exercise_number} ({exercise_type})\n"
            if main_text:
                formatted_question += f"Main Text: {main_text}\n"
            formatted_question += f"➤ Section {section_number} - {question_type}: {q_text}"
        

            # Handle question table (for base/guide questions, check their specific table field)
            question_table = None
            if self.current_question_index == 0 and base_question:
                question_table = section.get("base_question", {}).get("table", {}) if isinstance(section.get("base_question"), dict) else {}
            elif self.current_question_index == 0 and guide_question:
                question_table = section.get("guide_question", {}).get("table", {}) if isinstance(section.get("guide_question"), dict) else {}
            else:
                question_table = section.get("question", {}).get("table", {})
                
            if question_table and isinstance(question_table, dict) and question_table.get("headers"):
                table_text = self._stringify_table(question_table)
                if self.user_language == "en" and is_likely_hebrew(table_text):
                    table_text = translate_text_to_english(table_text)
                formatted_question += "\n" + table_text

            # Handle question options (check base/guide question options)
            question_options = []
            if self.current_question_index == 0 and base_question:
                question_options = section.get("base_question_options", [])
            elif self.current_question_index == 0 and guide_question:
                question_options = section.get("guide_question_options", [])
            else:
                question_options = section.get("question_options", [])
                
            if question_options:
                options_text = self._stringify_options(question_options)
                if self.user_language == "en" and is_likely_hebrew(options_text):
                    options_text = translate_text_to_english(options_text)
                formatted_question += "\n" + options_text

            # Handle main table
            main_table = main_data.get("table", {})
            if main_table and isinstance(main_table, dict) and main_table.get("headers"):
                main_table_text = self._stringify_table(main_table)
                if self.user_language == "en" and is_likely_hebrew(main_table_text):
                    main_table_text = translate_text_to_english(main_table_text)
                formatted_question += "\nMain Table:\n" + main_table_text

            # Handle SVGs (prioritize section SVG, then main SVG)
            svg_content = None
            # Debug: Check what SVG fields are available
            logger.debug(f"Checking SVG content for question index: {self.current_question_index}")
            logger.debug(f"svg_generated_for_question flag: {self.svg_generated_for_question}")

            if self.current_question_index == 0 and base_question:
                svg_content = section.get("base_question", {}).get("svg") if isinstance(section.get("base_question"), dict) else None
                logger.debug(f"Base question SVG content found: {bool(svg_content)}")
            elif self.current_question_index == 0 and guide_question:
                svg_content = section.get("guide_question", {}).get("svg") if isinstance(section.get("guide_question"), dict) else None
                logger.debug(f"Guide question SVG content found: {bool(svg_content)}")
            else:
                svg_content = section.get("question", {}).get("svg")
                logger.debug(f"Regular question SVG content found: {bool(svg_content)}")
                
            if not svg_content:
                svg_content = main_data.get("svg")
                logger.debug(f"Main data SVG content found: {bool(svg_content)}")
                
            # Always try to save SVG if content exists, regardless of flag
            if svg_content:
                logger.debug(f"SVG content length: {len(svg_content) if svg_content else 0}")
                self.current_svg_file_path = self._save_svg(svg_content)
                if self.current_svg_file_path:
                    formatted_question += f"\n[SVG: {self.current_svg_file_path}]"
                    logger.debug(f"SVG saved to: {self.current_svg_file_path}")
                else:
                    logger.warning("Failed to save SVG content")
                self.svg_generated_for_question = True
            else:
                logger.debug("No SVG content found for this question")
                
            # ✅ FIXED: Apply RTL for Hebrew responses (no additional translation needed)
            if self.user_language == "he":
                formatted_question = f"\u202B{formatted_question}\u202C"
            # ✅ REMOVED: No need for final translation since we translate components individually

            return formatted_question.strip()

        except Exception as e:
            logger.error(f"Error formatting question: {str(e)}")
            return self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)

    def _get_current_solution(self) -> str:
        content = self.current_exercise["exercise_content"]
        sec = content["sections"][self.current_question_index]
        sol_text = ""
        
        # Handle solution text
        solution = sec.get("solution")
        if solution and isinstance(solution, dict):
            sol = solution.get("text")
            if sol:
                sol_text += clean_math_text(sol) + "\n\n"
        
        # Handle full solution text
        full_solution = sec.get("full_solution")
        if full_solution and isinstance(full_solution, dict):
            full_sol = full_solution.get("text")
            if full_sol:
                sol_text += clean_math_text(full_sol) + "\n\n"

        # Handle solution table
        if solution and isinstance(solution, dict):
            sol_table = solution.get("table")
            if sol_table and isinstance(sol_table, dict) and sol_table.get("headers") and sol_table.get("rows_data"):
                sol_text += "Solution Table:\n" + json.dumps(sol_table, ensure_ascii=False) + "\n\n"
        
        # Handle full solution table
        if full_solution and isinstance(full_solution, dict):
            full_sol_table = full_solution.get("table")
            if full_sol_table and isinstance(full_sol_table, dict) and full_sol_table.get("headers") and full_sol_table.get("rows_data"):
                sol_text += "Full Solution Table:\n" + json.dumps(full_sol_table, ensure_ascii=False) + "\n\n"

        if not sol_text:
            return "No solution available."

        # Return in user's preferred language
        if self.user_language == "en":
            return translate_text_to_english(sol_text)
        return sol_text
    
    def _generate_lesson_summary(self) -> str:
        closing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly math tutor closing a short session.
            Write a 2–3 sentence positive summary of the lesson.
            End with a light humorous encouragement."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Summarize the lesson we just completed.")
        ])
        summary_chain = closing_prompt | llm
        try:
            response = summary_chain.invoke({"chat_history": self.chat_history[-10:]})
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating lesson summary: {e}")
            return "Great job today! You tackled some tough problems with confidence."

    def _pick_new_exercise(self, grade: str, topic: Optional[str] = None):
        exercises = get_exercises(grade, topic) if topic else [
            ex for ex in all_exercises if ex["exercise_metadata"]["class"] == grade
        ]
        if not exercises:
            # No dataset exercises → generate new one
            self.current_exercise = self.attempt_tracker.generate_exercise_with_llm(topic or "geometry", grade)
            if not self.current_exercise:
                return None
            # Save SVG for generated exercise
            self.current_svg_file_path = self._save_svg(self.current_exercise.get("svg"))
            self.svg_generated_for_question = True
        else:
            # Only pick exercises not used before
            available = [ex for ex in exercises if ex["exercise_metadata"]["exercise_number"]
                        not in self.recently_asked_exercise_ids]
            
            if not available:
                # No available exercises → signal exhaustion
                return None

            # Prioritize Basic exercises
            basic_exercises = [ex for ex in available if ex["exercise_metadata"].get("exercise_type", "").lower() == "basic"]
            if basic_exercises:
                self.current_exercise = random.choice(basic_exercises)
                logger.debug(f"Selected Basic exercise: {self.current_exercise['exercise_metadata']['exercise_number']}")
            else:
                # Fall back to any available exercise
                self.current_exercise = random.choice(available)
                logger.debug(f"No Basic exercises available, selected: {self.current_exercise['exercise_metadata']['exercise_number']}")

            # If geometric exercise requires SVG but missing, generate new one
            question_text = self.current_exercise["exercise_content"]["sections"][0].get("question", {}).get("text", "")
            main_svg = self.current_exercise.get("exercise_content", {}).get("main_data", {}).get("svg")
            if ("parallel" in question_text.lower() or "axes" in question_text.lower() or "segment" in question_text.lower()) and not main_svg:
                self.current_exercise = self.attempt_tracker.generate_exercise_with_llm(topic or "geometry", grade)
                if self.current_exercise:
                    self.current_svg_file_path = self._save_svg(self.current_exercise.get("svg"))
                    self.svg_generated_for_question = True

        # Track used exercises
        if self.current_exercise:
            self.recently_asked_exercise_ids.append(
                self.current_exercise["exercise_metadata"]["exercise_number"]
            )
            if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
                self.recently_asked_exercise_ids.pop(0)

            # Reset state
            self.current_question_index = 0
            self.current_hint_index = 0
            self.attempt_tracker.reset()
            self.student_answers = []

        return self.current_exercise  

    def _move_to_next_exercise_or_question(self) -> str:
        """Moves to the next question in the current exercise or to a new exercise."""
        if not self.current_exercise or "exercise_content" not in self.current_exercise:
            self.state = State.PICK_TOPIC
            self.current_exercise = None
            self.current_question_index = 0
            self.svg_generated_for_question = False
            self.completed_exercises = 0  # Reset counter if no exercise
            return self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)
        
        sections = self.current_exercise.get("exercise_content", {}).get("sections", [])
        self.current_question_index += 1
        
        if self.current_question_index < len(sections):
            # Move to the next question in the current exercise
            self.state = State.QUESTION_ANSWER
            self.attempt_tracker.reset()  # Reset attempts for the new question
            self.svg_generated_for_question = False  # Allow new SVG for the next question
            return f"\n{self._get_current_question()}"
        else:
            # Exercise completed, increment counter
            self.completed_exercises += 1
            self.current_question_index = 0  # Reset for new exercise
            self.svg_generated_for_question = False
            
            if self.completed_exercises < 2:
                new_ex = self._pick_new_exercise(self.hebrew_grade, self.topic)
                if not new_ex:
                    self.state = State.PICK_TOPIC
                    return "✅ You’ve finished all available exercises for this topic! Want to try another topic?"
                self.state = State.QUESTION_ANSWER
                return f"\nGreat job! Let's try another exercise:\n{self._get_current_question()}"

            else:
                # At least 2 exercises completed, move to ASK_FOR_DOUBTS
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"
            
    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate response to clear student's doubts using RAG."""
        try:
            # Translate the question if needed
            translated_question = translate_text_to_english(user_question) if self.user_language == "en" else user_question
            
            # Retrieve relevant context for the doubt
            retrieved_context = retrieve_relevant_chunks(
                translated_question,
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            context_str = clean_math_text(context_str)  # Clean context
            
            # Create a doubt-clearing prompt
            doubt_clearing_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a helpful Math AI tutor addressing a student's doubt or question.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Provide clear, detailed explanations
                - Use the context to give accurate information
                - Be patient and encouraging
                - Break down complex concepts into simple steps
                - If context doesn't contain relevant information, acknowledge this
                - Start with encouragement
                - Give step-by-step explanation
                - Use examples from context when available
                - End with confirmation question
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Student's Question: {question}\n\nRelevant Context: {context}")
            ])
            
            doubt_clearing_chain = doubt_clearing_prompt | llm
            response = doubt_clearing_chain.invoke({
                "chat_history": self.chat_history[-5:],
                "question": translated_question,
                "context": context_str
            })
            
            topic_name = self.topic or "this topic"
            intro = self._get_localized_text("doubt_clearing_intro", topic=topic_name)
            return f"{intro}\n\n{clean_math_text(response.content.strip())}"
            
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return "I'd be happy to help with your question, but I'm having trouble processing it right now. Could you try asking it in a different way?"

    def _handle_hint_request(self, user_input: str) -> str:
        """Always provide progressive guidance when hint is requested."""
        self.attempt_tracker.has_requested_hint = True
        self.state = State.PROVIDING_HINT

        current_question = self._get_current_question()
        retrieved_context = retrieve_relevant_chunks(
            f"Question: {current_question} User's Answer: {user_input}",
            self.pinecone_index,
            grade=self.hebrew_grade,
            topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
        )
        context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
        context_str = clean_math_text(context_str)
        if getattr(self, "current_svg_description", None):
            context_str += f"\n\nImage Description: {self.current_svg_description}"

        guidance = self._provide_progressive_guidance(
            user_input=user_input,  # Changed from user_answer to user_input
            question=current_question,
            context=context_str,
            is_forced=True  # ensure >= level 1
        )
        return guidance
    
    def _handle_solution_request(self, user_input: str) -> str:
        """Handle explicit solution requests by providing guiding questions first."""
        lang_dict = I18N[self.user_language]

        # Ensure we are working with the CURRENT question only
        current_question = self._get_current_question()
        current_solution = self._get_current_solution()
        
        # If they haven't gone through the guidance sequence, provide guiding questions first
        #if self.attempt_tracker.guidance_level < 2:  # Less than hint level
            # To this:
        if self.attempt_tracker.guidance_level < 3:  # Haven't completed guiding questions + hint
            current_question = self._get_current_question()
            retrieved_context = retrieve_relevant_chunks(
                f"Question: {current_question} User's Answer: {user_input}",
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            context_str = clean_math_text(context_str)  # Clean context
            #if self.current_svg_description:
                #context_str += f"\n\nImage Description: {self.current_svg_description}"
            
            # Provide guiding question instead of direct solution
            encouragement = lang_dict["encouragement"]
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            guiding_prefix = lang_dict["guiding_question"]
            self.attempt_tracker.guidance_level = 1
            self.state = State.GUIDING_QUESTION
            return f"{encouragement}{guiding_prefix}{guiding_q}"
        
        # If they've been through guidance, provide solution
        else:
            self.attempt_tracker.has_requested_solution = True
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            
            # Generate detailed explanation using AI
            try:
                explanation_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are a Math AI tutor providing a detailed solution explanation.
                    
                    Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                    
                    Guidelines:
                    - Always explain in steps: Step 1, Step 2, Step 3...
                    - First: state the key formula or rule used.
                    - Second: substitute values from the problem.
                    - Third: simplify step by step.
                    - Fourth: conclude with the final answer.
                    - End with a short "check your answer" verification if possible.
                    - Be clear, concise, and educational.
                    - Never show raw $ signs or LaTeX markup.
                    """),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "Question: {question}\nSolution: {solution}\n\nProvide a step-by-step explanation:")
                ])
                
                explanation_chain = explanation_prompt | llm
                current_question = self._get_current_question()
                
                response = explanation_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "question": current_question,
                    "solution": solution
                })
                
                explanation = clean_math_text(response.content.strip())
                
                # ✅ Turn the explanation into markdown steps
                steps_md = to_markdown_steps(explanation)
                result = f"{solution_prefix}\n{steps_md}"
                
                # Generate NEW SVG for solution explanation
                svg_reference = ""
                if self.current_exercise and self.current_exercise.get("svg"):
                    svg_reference = self._generate_and_save_svg(for_solution_explanation=True)
                
                result = f"{solution_prefix}{solution}\n\n{explanation}"
    
                if svg_reference:
                    result += f"\n{svg_reference}"
                result +="\n\n"
                result += self._move_to_next_exercise_or_question()
                return result
                
            except Exception as e:
                logger.error(f"Error generating solution explanation: {e}")
                result = f"{solution_prefix}{solution}"
                result += self._move_to_next_exercise_or_question()
                return result
            
    def _generate_progressive_hint(self, hint_index: int) -> str:
        """Generate a progressive hint for the current question."""
        try:
            # Get the current question and solution
            current_question = self._get_current_question()
            current_solution = self._get_current_solution()

            # Retrieve relevant context from Pinecone
            retrieved_context = retrieve_relevant_chunks(
                f"Question: {current_question}",
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            context_str = clean_math_text(context_str)

            # Add SVG description if available
            if getattr(self, "current_svg_description", None):
                context_str += f"\n\nImage Description: {self.current_svg_description}"

            # Create a prompt to generate a hint
            hint_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor providing a concise hint for a math problem.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Provide a single, concise hint (1-2 sentences) to guide the student toward the solution
                - Do NOT reveal the full solution
                - Focus on a key concept, formula, or step needed to solve the problem
                - Use the context to ensure relevance
                - Example: "Consider the slope of each line to match it with the function."
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Question: {question}\nContext: {context}\n\nGenerate a concise hint:")
            ])

            hint_chain = hint_prompt | self.llm
            response = hint_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": current_question,
                "context": context_str
            })

            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating hint: {str(e)}")
            return "Try breaking the problem into smaller steps."

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str) -> Dict[str, Any]:
        """Enhanced answer evaluation with progressive guidance system."""
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Math AI tutor evaluating a student's answer.
            
            Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
            
            Evaluation Guidelines:
            1. Determine if the answer is CORRECT or INCORRECT
            2. If INCORRECT, identify the specific mistake or misconception
            3. Provide encouragement regardless of correctness
            4. DO NOT reveal the correct answer
            5. Be supportive and educational
            
            Response Format:
            CORRECT: [brief encouraging comment]
            OR
            SORRY: [brief explanation of what went wrong without giving the answer]
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {question}\nStudent Answer: {answer}\nContext: {context}\n\nEvaluate the answer:"),
        ])
        
        evaluation_chain = evaluation_prompt | llm
        try:
            eval_response = evaluation_chain.invoke({
                "chat_history": self.chat_history[-4:],
                "question": question,
                "answer": user_input,
                "context": context
            })
            
            evaluation_result = clean_math_text(eval_response.content.strip())  # Clean evaluation response
            is_correct = evaluation_result.lower().startswith("correct:")
            #is_partial = evaluation_result.lower().startswith("partial:") ##updated
            
            return {
                "is_correct": is_correct,
                #"is_partial": is_partial,  #updated
                "feedback": evaluation_result,
                "needs_guidance": not is_correct
            }
        except Exception as e:
            logger.error(f"Error in answer evaluation: {e}")
            return {
                "is_correct": False,
                "feedback": "I couldn't evaluate your answer right now.",
                "needs_guidance": True
            }

    def _provide_progressive_guidance(self, user_input: str, question: str, context: str, is_forced: bool = False) -> str:
        """Provide progressive guidance based on attempts and current guidance level."""
        lang_dict = I18N[self.user_language]
        
        if self.attempt_tracker.guidance_level == 0:  # Encouragement
            
            self.attempt_tracker.guidance_level = 1
            self.state = State.GUIDING_QUESTION
            
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
    
            return f"{guiding_prefix}{guiding_q}"
            
        elif self.attempt_tracker.guidance_level == 1:  # Second Guiding Question
            self.attempt_tracker.guidance_level = 2
            self.state = State.GUIDING_QUESTION
            # Generate a different guiding question, possibly using more context or rephrasing
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
            return f"{guiding_prefix}{guiding_q}"

        elif self.attempt_tracker.guidance_level == 2:  # Hint
            if not is_forced and not self.attempt_tracker.can_provide_hint():
                return f"{lang_dict['encouragement']}{lang_dict['try_again']}"
            self.attempt_tracker.guidance_level = 3
            self.state = State.PROVIDING_HINT
            hint = self._generate_progressive_hint(0)
            if hint:
                return f"{hint}"
            else:
                self.attempt_tracker.guidance_level = 4
                return self._get_current_solution()
                
        else:  # guidance_level >= 3, provide solution
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            return f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise_or_question()}"

    def _reset_attempt_tracking(self):
        self.attempt_tracker.reset()

    def _generate_grade_acknowledgment(self, user_input: str, grade: str) -> str:
        ack_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a short acknowledgment for the grade."),
            ("user", "Grade: {grade}")
        ])
        chain = ack_prompt | llm
        response = chain.invoke({"grade": grade})
        return response.content.strip()

    def _get_diagnostic_question(self) -> str:
        questions = ["diagnostic_test", "diagnostic_last_class", "diagnostic_focus"]
        return self._get_localized_text(questions[self.diagnostic_question_index])

    def _generate_ai_personal_followup(self, user_input: str) -> str:
        response = self.personal_followup_chain.invoke({
            "chat_history": self.chat_history[-3:],
            "input": user_input
        })
        return clean_math_text(response.content.strip())

    def _generate_academic_transition(self, user_input: str) -> str:
        response = self.academic_transition_chain.invoke({
            "chat_history": self.chat_history[-3:],
            "input": user_input
        })
        return clean_math_text(response.content.strip())
    
    def _generate_guiding_question(self, user_answer: str, question: str, context: str) -> str:
        """Generate a guiding question to help student think through the problem."""
        try:
            guiding_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor. Generate a guiding question to help the student think through the problem step by step.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Ask a question that guides them toward the solution
                - Don't give away the answer directly
                - Focus on the mathematical concept or method
                - Be encouraging and supportive
                - Keep it concise (1-2 sentences)
                
                Example guiding questions:
                - "What operation should we use first?"
                - "Can you identify what type of equation this is?"
                - "What do you think the first step should be?"
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Problem: {question}\nStudent's Answer: {answer}\nContext: {context}\n\nGenerate a helpful guiding question:")
            ])
            
            guiding_chain = guiding_prompt | llm
            response = guiding_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": question,
                "answer": user_answer,
                "context": context
            })
            
            return clean_math_text(response.content.strip())  # Clean guiding question
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            guiding_text = self._get_localized_text("guiding_question")
            return f"{guiding_text}What do you think the first step should be?"

    def transition(self, user_input: str) -> Dict[str, Any]:
        """Enhanced FSM transition with progressive guidance and improved attempt tracking."""
        # Mark user activity (prevents premature inactivity timeout)
        if user_input.strip():
            self.inactivity_timer.reset()
            
        text_lower = (user_input or "").strip().lower()

        # Detect user language from input
        if user_input and user_input.strip():
            detected_lang = detect_language(user_input)
            if detected_lang in ["he", "en"]:
                self.user_language = detected_lang
            else:
                self.user_language = "en"
                
        # Add user input to chat history
        if user_input:
            self.chat_history.append(HumanMessage(content=clean_math_text(user_input)))

        response_dict = {"text": "", "svg_file_path": None}  # Initialize response dictionary

        # --- State Transitions ---
        if self.state == State.START:
            self.state = State.SMALL_TALK
            self.small_talk_question_index = 0
            self.small_talk_responses = []
            self.small_talk_turns = 1
            simple_greetings = ["Hey! How are you?", "Hi there!", "What's up?", "How's it going?"]
            response_dict["text"] = random.choice(simple_greetings)
            self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.SMALL_TALK:
            if self.small_talk_question_index == 0:
                try:
                    response = self.small_talk_chain.invoke({
                        "chat_history": self.chat_history[-3:],
                        "input": user_input or ""
                    })
                    response_text = clean_math_text(response.content.strip())
                    hobbies_q = self._get_localized_text("small_talk_hobbies")
                    response_dict["text"] = f"{response_text} {hobbies_q}"
                    self.small_talk_question_index += 1
                    self.chat_history.append(AIMessage(content=response_dict["text"]))
                except Exception as e:
                    logger.error(f"Error generating contextual small talk: {e}")
                    fallback_response = "I'm doing great, thanks for asking!" if self.user_language == "en" else "אני בסדר, תודה ששאלת!"
                    hobbies_q = self._get_localized_text("small_talk_hobbies")
                    response_dict["text"] = f"{fallback_response} {hobbies_q}"
                    self.small_talk_question_index += 1
                    self.chat_history.append(AIMessage(content=response_dict["text"]))

            else:
                self.small_talk_responses.append(user_input)
                self.state = State.PERSONAL_FOLLOWUP
                response_dict["text"] = self._generate_ai_personal_followup(user_input)
                self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.DIAGNOSTIC
            self.diagnostic_question_index = 0
            self.diagnostic_responses = []
            response_dict["text"] = self._get_diagnostic_question()
            self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.DIAGNOSTIC:
            self.diagnostic_responses.append(user_input)
            self.diagnostic_question_index += 1
            if self.diagnostic_question_index < 3:
                next_question = self._get_diagnostic_question()
                try:
                    contextual_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""You are a friendly math tutor.
                        Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                        Guidelines:
                        - Acknowledge the user's previous response in ONE short sentence (5-10 words).
                        - Be warm, conversational, and encouraging.
                        - Examples: 'Wow, great to hear!', 'That's awesome!', 'Cool, love that!'
                        - Return ONLY the acknowledgment sentence."""),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("user", "{input}")
                    ])
                    contextual_chain = contextual_prompt | llm
                    ack_response = contextual_chain.invoke({
                        "chat_history": self.chat_history[-3:],
                        "input": user_input
                    })
                    acknowledgment = clean_math_text(ack_response.content.strip()) or "That's awesome!"
                except Exception as e:
                    logger.error(f"Error generating contextual acknowledgment: {e}")
                    acknowledgment = "That's awesome!"
                response_dict["text"] = f"{acknowledgment} So, {next_question}"
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            else:
                self.state = State.ACADEMIC_TRANSITION
                response_dict["text"] = self._generate_academic_transition(user_input)
                self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.ACADEMIC_TRANSITION:
            self.state = State.PICK_CLASS
            classes = get_classes()
            response_dict["text"] = self._generate_academic_transition(user_input) + f"\n\nAvailable classes: {classes}\nPick a class: "
            self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.PICK_CLASS:
            chosen_class = user_input.strip()
            classes_hebrew = get_classes()
            if self.user_language == "en":
                classes_display = [translate_text_to_english(c) for c in classes_hebrew]
            else:
                classes_display = classes_hebrew
            if chosen_class not in classes_display:
                response_dict["text"] = f"{self._get_localized_text('invalid_class')} {classes_display}"
            else:
                if self.user_language == "en":
                    idx = classes_display.index(chosen_class)
                    self.grade = classes_display[idx]
                    self.hebrew_grade = classes_hebrew[idx]
                else:
                    self.grade = chosen_class if not is_likely_hebrew(chosen_class) else self._translate_grade_to_hebrew(chosen_class)
                    self.hebrew_grade = self._translate_grade_to_hebrew(self.grade) if not is_likely_hebrew(chosen_class) else chosen_class
                self.state = State.PICK_TOPIC
                topics_hebrew = get_topics(self.hebrew_grade)
                topics_display = [translate_text_to_english(t) for t in topics_hebrew] if self.user_language == "en" else topics_hebrew[:]
                response_dict["text"] = f"{'Available topics' if self.user_language=='en' else 'נושאים זמינים'}: {topics_display}\n{'Pick a topic:' if self.user_language=='en' else 'בחר נושא:'}"
            self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.PICK_TOPIC:
            chosen_topic = user_input.strip()
            topics_hebrew = get_topics(self.hebrew_grade)
            if user_input and detect_language(user_input) == "en":
                self.user_language = "en"
            topics_display = [translate_text_to_english(t) for t in topics_hebrew] if self.user_language == "en" else topics_hebrew
            topics_display_lower = [t.lower() for t in topics_display]
            if chosen_topic.lower() not in topics_display_lower:
                response_dict["text"] = f"{self._get_localized_text('invalid_topic')} {topics_display[:]}"
            else:
                if self.user_language == "en":
                    idx = topics_display.index(chosen_topic)
                    self.topic = topics_hebrew[idx]
                else:
                    self.topic = chosen_topic
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self._pick_new_exercise(self.hebrew_grade, self.topic)
                if not self.current_exercise:
                    response_dict["text"] = self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)
                else:
                    self.state = State.QUESTION_ANSWER
                    ready_text = self._get_localized_text("ready_for_question")
                    response_dict["text"] = f"{ready_text}\n{self._get_current_question()}"
                    response_dict["svg_file_path"] = self.current_svg_file_path  # Pass SVG file path
            self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            irrelevant_keywords = ["recipe", "cake", "story", "joke", "weather", "song", "news", "football", "music", "movie", "politics", "food", "travel", "holiday"]
            if any(word in text_lower for word in irrelevant_keywords):
                response_dict["text"] = self._get_localized_text("irrelevant_msg") + "\n\nLet's focus on the current exercise."
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            elif (text_lower == "hint" or any(keyword in text_lower for keyword in ["hint", "help", "clue", "tip", "stuck", "don't know", "not sure", "confused", "רמז", "עזרה"]) or
                ("give" in text_lower and any(keyword in text_lower for keyword in ["hint", "help", "clue"])) or
                ("can you" in text_lower and any(keyword in text_lower for keyword in ["hint", "help"]))):
                response_dict["text"] = self._handle_hint_request(user_input)
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            elif (text_lower in {"solution", "pass"} or
                any(keyword in text_lower for keyword in ["solution", "answer", "pass", "skip", "give up", "show me the solution", "פתרון", "תשובה"]) or
                ("give me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("show me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"]))):
                response_dict["text"] = self._handle_solution_request(user_input)
                response_dict["svg_file_path"] = self.current_svg_file_path  # Pass SVG file path if updated
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            else:
                current_question = self._get_current_question()
                current_solution = self._get_current_solution()
                retrieved_context = retrieve_relevant_chunks(
                    f"Question: {current_question} User's Answer: {user_input}",
                    self.pinecone_index,
                    grade=self.hebrew_grade,
                    topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
                )
                context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
                if self.current_svg_description:
                    context_str += f"\n\nImage Description: {self.current_svg_description}"
                evaluation_result = self._evaluate_answer_with_guidance(user_input, current_question, current_solution, context_str)
                should_offer_guidance = self.attempt_tracker.record_attempt(evaluation_result["is_correct"])
                if evaluation_result["is_correct"]:
                    self.student_answers.append({
                        "section": self.current_exercise["exercise_content"]["sections"][self.current_question_index]['section_number'],
                        "question": current_question,
                        "answer": user_input
                    })
                    response_dict["text"] = "✅ Correct!" + self._move_to_next_exercise_or_question()
                    response_dict["svg_file_path"] = self.current_svg_file_path  # Pass SVG file path for next question
                    self.state = State.QUESTION_ANSWER
                    self.chat_history.append(AIMessage(content=response_dict["text"]))
                else:
                    feedback_lines = evaluation_result.get("feedback", self._get_localized_text("wrong_answer")).split('\n')
                    main_feedback = feedback_lines[0] if feedback_lines else self._get_localized_text("wrong_answer")
                    if should_offer_guidance:
                        guidance = self._provide_progressive_guidance(user_input, current_question, context_str)
                        response_dict["text"] = f"{main_feedback}\n\n{guidance}"
                    else:
                        encouragement = self._get_localized_text("encouragement")
                        try_again = self._get_localized_text("try_again")
                        response_dict["text"] = f"{main_feedback}\n\n{encouragement}{try_again}"
                    self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.ASK_FOR_DOUBTS:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no doubts", "no questions", "לא", "אין", "בסדר"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            topic_name = self.topic or "this topic"
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                summary = self._generate_lesson_summary()
                closing_message = self._get_localized_text("lesson_closing")
                self.state = State.PICK_TOPIC
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self.completed_exercises = 0  # Reset counter for new topic
                self.current_exercise = None
                response_dict["text"] = f"{summary}\n\n{closing_message}\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                if "?" in user_input:
                    doubt_response = self._generate_doubt_clearing_response(user_input)
                else:
                    doubt_response = f"I'm ready to help! What would you like me to explain or clarify about {topic_name}?"
                response_dict["text"] = doubt_response + f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            else:
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                response_dict["text"] = self._generate_doubt_clearing_response(user_input) + f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=response_dict["text"]))

        elif self.state == State.DOUBT_CLEARING:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no more", "that's all", "thanks", "לא", "אין", "תודה", "זה הכל"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            topic_name = self.topic or "this topic"
            if any(indicator in text_lower for indicator in no_doubt_indicators) or self.doubt_questions_count >= self.MAX_DOUBT_QUESTIONS:
                summary = self._generate_lesson_summary()
                closing_message = self._get_localized_text("lesson_closing")
                self.state = State.PICK_TOPIC
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self.completed_exercises = 0  # Reset counter for new topic
                self.current_exercise = None
                response_dict["text"] = f"{summary}\n\n{closing_message}\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.doubt_questions_count += 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                if self.doubt_questions_count < self.MAX_DOUBT_QUESTIONS:
                    doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                response_dict["text"] = doubt_response
                self.chat_history.append(AIMessage(content=response_dict["text"]))
            else:
                response_dict["text"] = f"Could you clarify your question about {topic_name} or say 'no' if you're ready to move on?"
                self.chat_history.append(AIMessage(content=response_dict["text"]))

        else:
            response_dict["text"] = "I'm not sure how to proceed. Type 'exit' to quit."
            self.chat_history.append(AIMessage(content=response_dict["text"]))

        return response_dict

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not INPUT_FILE.exists():
        logger.error("❌ Missing JSON file.")
        return

    try:
        exercises = load_exercises()
        pinecone_index = get_pinecone_index()
    except Exception as e:
        logger.error(f"❌ Error loading data or connecting to Pinecone: {e}")
        return

    fsm = DialogueFSM(exercises, pinecone_index,llm)  # Pass llm to FSM

    # Initial transition to start the conversation
    initial_response = fsm.transition("")

    print(f"A_GUY: {initial_response['text']}")
   # if initial_response["svg_file_path"]:
        #print(f"[SVG available at: {initial_response['svg_file_path']}]")


    #print(f"A_GUY: {initial_response}")

    while True:
        try:
            # Enhanced input handling with typing detection
            fsm.inactivity_timer.mark_typing()
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            fsm.inactivity_timer.stop()
            break
        if user_input.lower() in {"exit", "quit", "done"}:
            print("👋 Bye!")
            fsm.inactivity_timer.stop()
            break

        response = fsm.transition(user_input)
        response_text = response.get("text", "")
        if fsm.user_language == "he":
            response_text = f"\u202B{response_text}\u202C"

        print(f"A_GUY: {response['text']}")
        if response["svg_file_path"]:
            print(f"[SVG available at: {response['svg_file_path']}]")

if __name__ == "__main__":
    main()