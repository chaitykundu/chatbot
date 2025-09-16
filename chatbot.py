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

logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("Files/merged_output.json")  # Align with index_embed INPUT_FILE
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
    - For Hebrew responses, use Right-to-Left (RTL) formatting for conversational text.
    - Ensure all mathematical expressions and scientific notation remain Left-to-Right (LTR), even within Hebrew sentences.
    
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

# New prompt for generating question and SVG
question_svg_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Math tutor creating a new question and SVG visualization.
    
    Language: Generate the question in {language} ({'Hebrew' if language == 'he' else 'English'})
    
    Task:
    - Generate a NEW math question for the given topic and grade.
    - Include a corresponding SVG visualization (single <svg> element).
    - For slope questions: Use two distinct points (e.g., (1,1) and (4,7)), calculate slope (e.g., 2).
    - Structure output as JSON with fields: question (str), solution (str), svg (str).
    - SVG Guidelines:
      - Use viewBox="0 0 400 300".
      - Include axes: x (0-10, bottom), y (0-10, left, inverted).
      - Draw a bold blue line for the slope, add light grid, label points and axes.
      - Ensure valid, self-contained SVG XML (no interactions).
    - Question Guidelines:
      - Simple, solvable, grade-appropriate (e.g., grade 7/8).
      - Include specific points (e.g., "Find the slope of the line through (1,1) and (4,7)").
      - Solution should show steps (e.g., "Slope = (y2-y1)/(x2-x1) = (7-1)/(4-1) = 6/3 = 2").
    - Return ONLY the JSON object, no explanations or markdown."""),
    ("user", "Topic: {topic}\nGrade: {grade}")
])
question_svg_generation_chain = question_svg_generation_prompt | llm


# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)",
        "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
        "ask_topic": "Great! Grade {grade}. Which topic would you like to practice? (e.g., {topics})",
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
        "lesson_closing": "Great, that was an awesome lesson! I’ll send you similar exercises for practice and see you in the next session. If you have questions, feel free to message me. And if you get stuck – just remember, you’re a genius. Bye!"
    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)",
        "ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז, ח)",
        "ask_topic": "מצוין! כיתה {grade}. באיזה נושא תרצה להתרגל? (לדוגמה: {topics})",
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
        "lesson_closing": "נהדר, זה היה שיעור מדהים! אשלח לך תרגילים דומים לתרגול וניפגש בשיעור הבא. אם יש לך שאלות, אל תהסס לפנות אליי. ואם תיתקע – זכור, אתה גאון. להתראות!"
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
    ASK_GRADE = auto()
    EXERCISE_SELECTION = auto()
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
    # Remove inline LaTeX ($...$)
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)
    # Remove display LaTeX ($$...$$)
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    def replace_fraction(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return f'({numerator}/{denominator})'
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_fraction, text)
    # Remove backslash commands (e.g., \frac, \sqrt) but keep content
    text = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', r'\2', text)
   
    # Remove standalone backslashes
    text = re.sub(r'\\([a-zA-Z]+)', r'', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any remaining $ or $$ that might be malformed
    text = text.replace('$', '')
    return text.strip()


def translate_text_to_english(text: str) -> str:
    """Translate text (likely Hebrew) to English using GenAI."""
    if not text or not text.strip():
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise translator. Translate the following text to English. If it's already in English, return it as is. Provide ONLY the translation."),
            ("user", "{input}"),
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        translated = response.content.strip()
        translated = clean_math_text(translated)
        if is_likely_hebrew(text) and not is_likely_hebrew(translated):
             return translated
        elif not is_likely_hebrew(text):
             return text
        else:
             logger.warning(f"Potential translation issue. Input: {text}, Output: {translated}")
             return translated
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"[Translation Error: {text}]"

def is_likely_hebrew(text: str) -> bool:
    """Simple heuristic to check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

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
    
    def generate_exercise_with_llm(topic: str, grade: str) -> Optional[Dict[str, Any]]:
        """Generates a new exercise and SVG using a generative LLM."""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = question_svg_generation_prompt.format(topic=topic, grade=grade)
            response = model.generate_content(prompt)
            
            # Extract and parse the JSON from the response
            generated_json = response.text.strip('` \n').strip('json\n')
            exercise = json.loads(generated_json)
            return exercise
        except Exception as e:
            logger.error(f"❌ Error generating exercise with LLM: {e}")
            return None

# -----------------------------
# Enhanced Dialogue FSM
# -----------------------------
class DialogueFSM:
    def __init__(self, exercises_data, pinecone_index):
        self.state = State.START
        self.grade = None
        self.hebrew_grade = None
        self.exercises_data = exercises_data
        self.pinecone_index = pinecone_index
        self.topic = None
        self.current_exercise = None
        self.current_hint_index = 0
        self.current_question_index = 0
        self.chat_history = []
        self.current_svg_description = None
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 5
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
        logger.info(f"[INACTIVITY TIMEOUT] {message}")

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        """Translate numeric grade to Hebrew equivalent."""
        grade_map = {"7": "ז", "8": "ח"}
        return grade_map.get(grade_num, grade_num)

    def _get_localized_text(self, key: str, **kwargs) -> str:
        """Get localized text based on current user language."""
        lang_dict = I18N[self.user_language]
        text = lang_dict.get(key, I18N["en"][key])
        return text.format(**kwargs) if kwargs else text
    
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
    
    def _extract_metadata(self, exercise: Any) -> Optional[Dict[str, Any]]:
        """Safely extract metadata from an exercise, handling list or dict structures."""
        try:
            if isinstance(exercise, dict):
                # Direct dictionary with exercise_metadata
                return exercise.get("exercise_metadata", {})
            elif isinstance(exercise, list):
                # List structure, e.g., [..., {"grade": "ז", "topic": "algebra"}]
                for item in exercise:
                    if isinstance(item, dict) and ("grade" in item or "class" in item):
                        return item
                    elif isinstance(item, dict) and "exercise_metadata" in item:
                        return item["exercise_metadata"]
            logger.warning(f"Unexpected exercise structure: {exercise}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
        

    def _generate_grade_acknowledgment(self, user_input: str, grade: str) -> str:
        """Generate AI-based acknowledgment for grade selection."""
        try:
            acknowledgment_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly Math AI tutor acknowledging a student's grade selection.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Acknowledge the user's input warmly and briefly (1 sentence, 5-10 words)
                - Mention the grade they selected
                - Be encouraging and conversational
                - Examples: 'Great, grade 8 sounds awesome!', 'Nice choice, grade 7!'
                - Return ONLY the acknowledgment sentence
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "User input: {input}\nSelected grade: {grade}")
            ])
            acknowledgment_chain = acknowledgment_prompt | llm
            response = acknowledgment_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input,
                "grade": grade
            })
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating grade acknowledgment: {e}")
            return f"Got it, grade {grade} is perfect!" if self.user_language == "en" else f"מצוין, כיתה {self._translate_grade_to_hebrew(grade)} נהדרת!"

    def _generate_ai_small_talk(self, user_input: str = "") -> str:
        """Generate AI-based small talk response."""
        try:
            response = self.small_talk_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return clean_math_text(response.content.strip())  # Clean small talk response
        except Exception as e:
            logger.error(f"Error generating AI small talk: {e}")
            return "Hey! How's it going today?"

    def _generate_ai_personal_followup(self, user_input: str = "") -> str:
        """Generate AI-based personal follow-up response."""
        try:
            response = self.personal_followup_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating AI personal followup: {e}")
            return "That sounds fun! Tell me more."

    def _get_diagnostic_question(self) -> str:
        """Get the next diagnostic question based on index."""
        diagnostic_questions = [
            self._get_localized_text("diagnostic_test"),
            self._get_localized_text("diagnostic_last_class"),
            self._get_localized_text("diagnostic_focus")
        ]
        return diagnostic_questions[self.diagnostic_question_index]

    def _generate_academic_transition(self, user_input: str = "") -> str:
        """Generate AI-based academic transition response."""
        try:
            response = self.academic_transition_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating academic transition: {e}")
            return "Now, let's talk about math!"

    def _pick_new_exercise_rag(self, query: str, grade: Optional[str] = None, topic: Optional[str] = None) -> None:
        """Retrieve and reconstruct exercise from Pinecone chunks, aligning with index_embed chunking logic."""
        retrieved_chunks = retrieve_relevant_chunks(query, self.pinecone_index, grade, topic)
        if not retrieved_chunks:
            return

        # Group chunks by exercise_id
        exercise_groups = {}
        for chunk in retrieved_chunks:
            exercise_id = chunk.get('exercise_id')  # Align with index_embed exercise_id format
            if exercise_id not in exercise_groups:
                exercise_groups[exercise_id] = []
            exercise_groups[exercise_id].append(chunk)

        available_exercises = list(exercise_groups.keys())
        available_exercises = [ex for ex in available_exercises if ex not in self.recently_asked_exercise_ids]

        if not available_exercises:
            return

        selected_id = random.choice(available_exercises)
        self.recently_asked_exercise_ids.append(selected_id)
        if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
            self.recently_asked_exercise_ids.pop(0)

        selected_chunks = exercise_groups[selected_id]

        # Align with index_embed chunk order (main_text, section_data, question, hint, solution, full_solution)
        chunk_order = ['main_text', 'section_data', 'question', 'hint', 'solution', 'full_solution']
        selected_chunks.sort(key=lambda c: chunk_order.index(c.get('chunk_type', 'unknown')) if c.get('chunk_type') in chunk_order else 99)

        # Reconstruct nested exercise structure to match index_embed input format
        reconstructed = {
            "exercise_metadata": {},  # Will populate from base metadata
            "exercise_content": {
                "main_data": {
                    "text": "",
                    "table": {"headers": [], "rows_data": []},
                    "svg": None  # Handle SVG if exists (from metadata 'svg_exists')
                },
                "sections": []  # List of section dicts
            }
        }

        # Extract base metadata from first chunk (align with index_embed base_metadata)
        if selected_chunks:
            base_meta = selected_chunks[0]
            reconstructed["exercise_metadata"] = {
                "content_type": base_meta.get("content_type", "exercise"),
                "exercise_number": base_meta.get("exercise_number", ""),
                "exercise_type": base_meta.get("exercise_type", "math"),
                "total_sections": base_meta.get("total_sections", 0),
                "difficulty": base_meta.get("difficulty", "medium"),
                "topic": base_meta.get("topic", "algebra"),
                "class": base_meta.get("grade", ""),  # Align 'grade' with 'class' in index_embed
                "has_hints": base_meta.get("has_hints", False),
                "solution": base_meta.get("solution", False),
                "mathematical_concept": base_meta.get("mathematical_concept", "linear_equations"),
                "retrieval_priority": base_meta.get("retrieval_priority", 1),
                "svg_exists": base_meta.get("svg_exists", False)
            }

        # Group by section_id for nested sections (align with index_embed sections handling)
        section_groups = {}
        for chunk in selected_chunks:
            section_id = chunk.get('section_id')
            chunk_type = chunk.get('chunk_type')
            if section_id is None and chunk_type == 'main_text':
                # Handle main_data
                reconstructed["exercise_content"]["main_data"]["text"] = chunk.get('text', '')
                reconstructed["exercise_content"]["main_data"]["table"]["headers"] = chunk.get('table_headers', [])
                reconstructed["exercise_content"]["main_data"]["table"]["rows_data"] = json.loads(chunk.get('rows_data', '[]'))
                # SVG handling can be added if 'svg' key exists in original, but use metadata 'svg_exists' flag
            else:
                if section_id not in section_groups:
                    section_groups[section_id] = {
                        "section_id": section_id,
                        "section_number": chunk.get('section_number'),
                        "section_data": {"text": "", "table": {"headers": [], "rows_data": []}, "svg": None},
                        "question": {"text": "", "table": {"headers": [], "rows_data": []}, "svg": None},
                        "hint": {"text": "", "table": {"headers": [], "rows_data": []}, "svg": None},
                        "solution": {"text": "", "table": {"headers": [], "rows_data": []}, "svg": None},
                        "full_solution": {"text": "", "table": {"headers": [], "rows_data": []}, "svg": None}
                    }
                section = section_groups[section_id]
                if chunk_type == 'section_data':
                    section["section_data"]["text"] = chunk.get('text', '')
                    section["section_data"]["table"]["headers"] = chunk.get('table_headers', [])
                    section["section_data"]["table"]["rows_data"] = json.loads(chunk.get('rows_data', '[]'))
                elif chunk_type == 'question':
                    section["question"]["text"] = chunk.get('text', '')
                    section["question"]["table"]["headers"] = chunk.get('table_headers', [])
                    section["question"]["table"]["rows_data"] = json.loads(chunk.get('rows_data', '[]'))
                elif chunk_type == 'hint':
                    section["hint"]["text"] = chunk.get('text', '')
                    section["hint"]["table"]["headers"] = chunk.get('table_headers', [])
                    section["hint"]["table"]["rows_data"] = json.loads(chunk.get('rows_data', '[]'))
                elif chunk_type == 'solution':
                    section["solution"]["text"] = chunk.get('text', '')
                    section["solution"]["table"]["headers"] = chunk.get('table_headers', [])
                    section["solution"]["table"]["rows_data"] = json.loads(chunk.get('rows_data', '[]'))
                elif chunk_type == 'full_solution':
                    section["full_solution"]["text"] = chunk.get('text', '')
                    section["full_solution"]["table"]["headers"] = chunk.get('table_headers', [])
                    section["full_solution"]["table"]["rows_data"] = json.loads(chunk.get('rows_data', '[]'))

        # Add sections to reconstructed exercise
        reconstructed["exercise_content"]["sections"] = list(section_groups.values())

        self.current_exercise = reconstructed
        self.current_question_index = 0
        self.current_hint_index = 0
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

    def _get_current_question(self) -> str:
        """Compose current question text from reconstructed nested structure (align with index_embed)."""
        if not self.current_exercise:
            return "No question available."

        content = self.current_exercise["exercise_content"]
        main_data = content.get("main_data", {})
        sections = content.get("sections", [])

        q_text = ""

        # Add main_data text if exists
        if main_data.get("text"):
            q_text += clean_math_text(main_data["text"]) + "\n\n"

        # Add table from main_data if exists
        main_table = main_data.get("table", {})
        if main_table.get("headers") and main_table.get("rows_data"):
            q_text += "Table:\n" + json.dumps(main_table, ensure_ascii=False) + "\n\n"  # Or format as table string

        # Add sections (focus on current question index, assuming sections align with questions)
        if self.current_question_index < len(sections):
            sec = sections[self.current_question_index]
            if sec.get("section_data", {}).get("text"):
                q_text += clean_math_text(sec["section_data"]["text"]) + "\n\n"
            if sec.get("question", {}).get("text"):
                q_text += clean_math_text(sec["question"]["text"]) + "\n\n"

            # Add tables from section_data or question
            sec_table = sec.get("section_data", {}).get("table", {})
            if sec_table.get("headers") and sec_table.get("rows_data"):
                q_text += "Section Table:\n" + json.dumps(sec_table, ensure_ascii=False) + "\n\n"
            q_table = sec.get("question", {}).get("table", {})
            if q_table.get("headers") and q_table.get("rows_data"):
                q_text += "Question Table:\n" + json.dumps(q_table, ensure_ascii=False) + "\n\n"

        # SVG handling (generate only once per question)
        if self.current_exercise["exercise_metadata"].get("svg_exists") and not self.svg_generated_for_question:
            svg_reference = self._generate_and_save_svg(for_solution_explanation=False)
            if svg_reference and isinstance(svg_reference, str):
                q_text += f"\n{svg_reference}"
            self.svg_generated_for_question = True
        elif self.current_svg_file_path and self.svg_generated_for_question:
            q_text += f"\n\n[Image File: {self.current_svg_file_path.as_posix()}]"

        # Return in user's preferred language
        if self.user_language == "en":
            return translate_text_to_english(q_text)
        return q_text

    def _get_current_solution(self) -> str:
        if not self.current_exercise:
            return "No solution available."

        sections = self.current_exercise["exercise_content"].get("sections", [])
        if self.current_question_index >= len(sections):
            return "No solution available."

        sec = sections[self.current_question_index]
        sol_text = ""
        if sec.get("solution", {}).get("text"):
            sol_text += clean_math_text(sec["solution"]["text"]) + "\n\n"
        if sec.get("full_solution", {}).get("text"):
            sol_text += clean_math_text(sec["full_solution"]["text"]) + "\n\n"

        # Add tables from solution/full_solution
        sol_table = sec.get("solution", {}).get("table", {})
        if sol_table.get("headers") and sol_table.get("rows_data"):
            sol_text += "Solution Table:\n" + json.dumps(sol_table, ensure_ascii=False) + "\n\n"
        full_sol_table = sec.get("full_solution", {}).get("table", {})
        if full_sol_table.get("headers") and full_sol_table.get("rows_data"):
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

    def _move_to_next_exercise_or_question(self) -> str:
        """Enhanced version that strictly provides 2 exercises before doubt checking."""
        
        # Check if there are more questions in the current exercise (align with total_sections)
        total_sections = self.current_exercise["exercise_metadata"].get("total_sections", 0)
        if self.current_question_index < total_sections - 1:
            
            # Move to next question within the same exercise
            self.current_question_index += 1
            self.current_hint_index = 0
            self._reset_attempt_tracking()  # This will reset SVG tracking too
            return f"\n\nNext question:\n{self._get_current_question()}"

        else:
            # Current exercise is finished - increment counter
            self.topic_exercises_count += 1

            # Strictly check for exactly 2 exercises
            if self.topic_exercises_count >= self.MAX_EXERCISES:
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"

            # Construct query for a new exercise
            query = f"Next exercise for grade {self.hebrew_grade}"
            topic_for_query = None
            if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"]:
                query += f" on topic {self.topic}"
                topic_for_query = self.topic

            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)

            if not self.current_exercise:
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"

            return f"\n\nNext exercise:\n{self._get_current_question()}"

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

    def transition(self, user_input: str) -> str:
        """Enhanced FSM transition with progressive guidance and improved attempt tracking."""
        
        # Mark user activity (prevents premature inactivity timeout)
        if user_input.strip():
            self.inactivity_timer.reset()
            
        text_lower = (user_input or "").strip().lower()

        # Detect user language from input
        if user_input:
            detected_lang = detect_language(user_input)
            if detected_lang != self.user_language and detected_lang in ["he", "en"]:
                self.user_language = detected_lang
            
        # Add user input to chat history
        if user_input:
            self.chat_history.append(HumanMessage(content=clean_math_text(user_input)))

        # --- State Transitions ---
        if self.state == State.START:
            self.state = State.SMALL_TALK
            self.small_talk_question_index = 0
            self.small_talk_responses = []
            self.small_talk_turns = 1
            simple_greetings = ["Hey! How are you?", "Hi there!", "What's up?", "How's it going?"]
            ai_response = random.choice(simple_greetings)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.SMALL_TALK:
            if self.small_talk_question_index == 0:
                # Generate contextual response to greeting + hobbies question
                try:
                    response = self.small_talk_chain.invoke({
                        "chat_history": self.chat_history[-3:],
                        "input": user_input or ""
                    })
                    response_text = clean_math_text(response.content.strip())
                    # Ensure hobbies question is included
                    hobbies_q = self._get_localized_text("small_talk_hobbies")
                    response_text = f"{response_text} {hobbies_q}"
                    self.small_talk_question_index += 1
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                except Exception as e:
                    logger.error(f"Error generating contextual small talk: {e}")
                    # Fallback: static response + hobbies
                    fallback_response = "I'm doing great, thanks for asking!" if self.user_language == "en" else "אני בסדר, תודה ששאלת!"
                    hobbies_q = self._get_localized_text("small_talk_hobbies")
                    response_text = f"{fallback_response} {hobbies_q}"
                    self.small_talk_question_index += 1
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
            else:
                # User responded to hobbies, move to personal followup
                self.small_talk_responses.append(user_input)
                self.state = State.PERSONAL_FOLLOWUP
                ai_response = self._generate_ai_personal_followup(user_input)
                self.chat_history.append(AIMessage(content=ai_response))
                return ai_response

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.DIAGNOSTIC
            self.diagnostic_question_index = 0
            self.diagnostic_responses = []  # Reset responses
            response_text = self._get_diagnostic_question()
            self.chat_history.append(AIMessage(content=response_text))
            return response_text
        
        elif self.state == State.DIAGNOSTIC:
            # Store the user's response
            self.diagnostic_responses.append(user_input)
            self.diagnostic_question_index += 1

            # Check if there are more diagnostic questions
            if self.diagnostic_question_index < 3:  # 3 questions total
                # Get next diagnostic question
                next_question = self._get_diagnostic_question()
                # Generate short contextual acknowledgment
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
                    acknowledgment = clean_math_text(ack_response.content.strip())
                    if not acknowledgment:  # Ensure non-empty
                        acknowledgment = "That's awesome!"
                except Exception as e:
                    logger.error(f"Error generating contextual acknowledgment: {e}")
                    acknowledgment = "That's awesome!"  # Fallback

                # Combine acknowledgment with diagnostic question
                response_text = f"{acknowledgment} So, {next_question}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            else:
                # All questions done - move to academic transition
                self.state = State.ACADEMIC_TRANSITION
                ai_response = self._generate_academic_transition(user_input)
                self.chat_history.append(AIMessage(content=ai_response))
                return ai_response
            
        elif self.state == State.ACADEMIC_TRANSITION:
            # Use existing academic transition
            ai_response = self._generate_academic_transition(user_input)
            self.state = State.ASK_GRADE
            grade_ask = self._get_localized_text("ask_grade")
            response_text = f"{ai_response}\n\n{grade_ask}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.ASK_GRADE:
            # Extract grade number from user input using regex and AI
            extracted_grade = self._extract_grade_from_input(user_input.strip())
            
            if not extracted_grade:
                # Ask again if no valid grade found - localized with examples
                examples = "7 or 8" if self.user_language == "en" else "ז or ח"
                retry_message = f"{self._get_localized_text('ask_grade')} ({self._get_localized_text('examples', examples=examples)})"
                self.chat_history.append(AIMessage(content=retry_message))
                return retry_message
            
            # Success: Set grades
            self.grade = extracted_grade
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            self.state = State.EXERCISE_SELECTION
            
            # Generate contextual acknowledgment using AI
            grade_acknowledgment = self._generate_grade_acknowledgment(user_input, self.grade)
            
            # Get available topics from exercises_data
            available_topics_hebrew = list(set(
                ex["exercise_metadata"].get("topic", "Unknown")
                for ex in self.exercises_data
                if isinstance(ex, dict) and "exercise_metadata" in ex and self._get_grade_key(ex["exercise_metadata"]) == self.hebrew_grade
            ))
            
            if available_topics_hebrew:
                if self.user_language == "en":
                    english_topics = [translate_text_to_english(topic) for topic in available_topics_hebrew[:3]]
                    topics_str = ", ".join(english_topics)
                else:
                    topics_str = ", ".join(available_topics_hebrew[:3])
            else:
                topics_str = "Any topic"
                
            topic_question = self._get_localized_text("ask_topic", grade=self.grade, topics=topics_str)
            response_text = f"{grade_acknowledgment} {topic_question}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.EXERCISE_SELECTION:
            self.topic = clean_math_text(user_input.strip()) # Clean topic input
            # Reset counters when new topic is selected
            self.topic_exercises_count = 0  # Reset for new topic
            self.doubt_questions_count = 0
            
            query = f"Find an exercise for grade {self.hebrew_grade} on topic {self.topic}"
            topic_for_picking = self.topic if self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_picking)

            if not self.current_exercise:
                logger.info(f"No exercises found for grade {self.hebrew_grade} and topic {self.topic}. Trying without topic filter.")
                query = f"Find an exercise for grade {self.hebrew_grade}"
                self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade)

            if not self.current_exercise:
                self.state = State.EXERCISE_SELECTION
                no_exercises = self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)
                no_relevant = self._get_localized_text("no_relevant_exercises")
                response_text = f"{no_exercises}\n{no_relevant}\n\nPlease choose another topic:"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            self.state = State.QUESTION_ANSWER
            ready_text = self._get_localized_text("ready_for_question")
            response_text = f"{ready_text}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            # Handle irrelevant questions
            irrelevant_keywords = [
                "recipe", "cake", "story", "joke", "weather", "song", "news", "football",
                "music", "movie", "politics", "food", "travel", "holiday"
            ]
            if any(word in text_lower for word in irrelevant_keywords):
                irrelevant_msg = self._get_localized_text("irrelevant_msg")
                response_text = irrelevant_msg + "\n\nLet's focus on the current exercise."
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle explicit hint requests with attempt checking
            hint_keywords = ["hint", "help", "clue", "tip", "stuck", "don't know", "not sure", "confused", "רמז", "עזרה"]
            if (text_lower == "hint" or 
                any(keyword in text_lower for keyword in hint_keywords) or
                ("give" in text_lower and any(keyword in text_lower for keyword in ["hint", "help", "clue"])) or
                ("can you" in text_lower and any(keyword in text_lower for keyword in ["hint", "help"]))):
                
                response_text = self._handle_hint_request(user_input)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle solution requests with attempt checking
            solution_keywords = ["solution", "answer", "pass", "skip", "give up", "show me the solution", "פתרון", "תשובה"]
            if (text_lower in {"solution", "pass"} or
                any(keyword in text_lower for keyword in solution_keywords) or
                ("give me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("show me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"]))):
                
                response_text = self._handle_solution_request(user_input)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle answer evaluation with enhanced attempt tracking
            else:
                current_question = self._get_current_question()
                current_solution = self._get_current_solution()

                # Get context for evaluation
                retrieved_context = retrieve_relevant_chunks(
                    f"Question: {current_question} User's Answer: {user_input}",
                    self.pinecone_index,
                    grade=self.hebrew_grade,
                    topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
                )
                context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
                if self.current_svg_description:
                    context_str += f"\n\nImage Description: {self.current_svg_description}"

                # Evaluate answer
                evaluation_result = self._evaluate_answer_with_guidance(user_input, current_question, current_solution, context_str)
                
                # Record the attempt
                should_offer_guidance = self.attempt_tracker.record_attempt(evaluation_result["is_correct"])
                
                if evaluation_result["is_correct"]:
                    response_text = "✅ Correct!"
                    response_text += self._move_to_next_exercise_or_question()
                    self.state = State.QUESTION_ANSWER
                    self._reset_attempt_tracking()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                else:
                    # Incorrect answer - handle based on attempt count
                    feedback_lines = evaluation_result.get("feedback", self._get_localized_text("wrong_answer")).split('\n')
                    main_feedback = feedback_lines[0] if feedback_lines else self._get_localized_text("wrong_answer")
                    
                    if should_offer_guidance:
                        # Provide progressive guidance
                        guidance = self._provide_progressive_guidance(user_input, current_question, context_str)
                        response_text = f"{main_feedback}\n\n{guidance}"
                    else:
                        # Encourage another attempt
                        encouragement = self._get_localized_text("encouragement")
                        try_again = self._get_localized_text("try_again")
                        response_text = f"{main_feedback}\n\n{encouragement}{try_again}"
                    
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

        elif self.state == State.ASK_FOR_DOUBTS:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no doubts", "no questions", "לא", "אין", "בסדר"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                # Generate lesson summary and reset for new topic
                summary = self._generate_lesson_summary()
                closing_message = self._get_localized_text("lesson_closing")
                self.state = State.EXERCISE_SELECTION
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self.current_exercise = None
                response_text = f"{summary}\n\n{closing_message}\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                
                if "?" in user_input:
                    doubt_response = self._generate_doubt_clearing_response(user_input)
                else:
                    doubt_response = f"I'm ready to help! What would you like me to explain or clarify about {topic_name}?"
                
                doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            
            else:
                # Treat as a doubt/question
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        elif self.state == State.DOUBT_CLEARING:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no more", "that's all", "thanks", "לא", "אין", "תודה", "זה הכל"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators) or self.doubt_questions_count >= self.MAX_DOUBT_QUESTIONS:
                # Generate lesson summary and reset for new topic
                summary = self._generate_lesson_summary()
                closing_message = self._get_localized_text("lesson_closing")
                self.state = State.EXERCISE_SELECTION
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self.current_exercise = None
                response_text = f"{summary}\n\n{closing_message}\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
                
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.doubt_questions_count += 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                if self.doubt_questions_count < self.MAX_DOUBT_QUESTIONS:
                    doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            else:
                # Unclear response
                doubt_response = f"Could you clarify your question about {topic_name} or say 'no' if you're ready to move on?"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        # Default fallback
        response_text = "I'm not sure how to proceed. Type 'exit' to quit."
        self.chat_history.append(AIMessage(content=response_text))
        return response_text

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not PARSED_INPUT_FILE.exists():
        logger.error("❌ Missing JSON file.")
        return

    try:
        exercises = load_json(PARSED_INPUT_FILE)
        pinecone_index = get_pinecone_index()
    except Exception as e:
        logger.error(f"❌ Error loading data or connecting to Pinecone: {e}")
        return

    fsm = DialogueFSM(exercises, pinecone_index)

    # Initial transition to start the conversation
    initial_response = fsm.transition("")
    print(f"A_GUY: {initial_response}")

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

        print(f"A_GUY: {response}")

if __name__ == "__main__":
    main()