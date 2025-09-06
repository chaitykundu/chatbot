import json
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

# Third-party imports
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationPhase(Enum):
    GREETING = "greeting"
    PERSONALIZATION = "personalization"
    ACADEMICS = "academics"
    GUIDANCE = "guidance"

class GuidanceLevel(Enum):
    QUESTION = "guiding_question"
    HINT = "hint"
    SOLUTION = "full_solution"

@dataclass
class UserContext:
    """Store user conversation context and preferences"""
    user_id: str
    current_phase: ConversationPhase = ConversationPhase.GREETING
    academic_level: str = ""
    interests: List[str] = None
    recent_topics: List[str] = None
    language_preference: str = "en"  # en for English, he for Hebrew
    conversation_history: List[Dict] = None
    current_guidance_level: GuidanceLevel = GuidanceLevel.QUESTION
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.recent_topics is None:
            self.recent_topics = []
        if self.conversation_history is None:
            self.conversation_history = []

@dataclass
class RetrievalResult:
    """Store retrieved exercise information"""
    exercise_id: str
    content: str
    chunk_type: str
    metadata: Dict
    relevance_score: float
    section_id: Optional[str] = None

class LanguageDetector:
    """Detect and handle Hebrew/English content"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple Hebrew detection based on character ranges"""
        hebrew_chars = re.findall(r'[\u0590-\u05FF]', text)
        total_chars = len(re.findall(r'[a-zA-Z\u0590-\u05FF]', text))
        
        if total_chars == 0:
            return "en"
        
        hebrew_ratio = len(hebrew_chars) / total_chars
        return "he" if hebrew_ratio > 0.3 else "en"
    
    @staticmethod
    def is_rtl_language(text: str) -> bool:
        """Check if text requires RTL direction"""
        return LanguageDetector.detect_language(text) == "he"

class RetrievalSystem:
    """Handle vector search and content retrieval from Pinecone"""
    
    def __init__(self, index_name: str = "exercise-embeddings", model_name: str = "intfloat/multilingual-e5-large"):
        self.index_name = index_name
        self.model = SentenceTransformer(model_name)
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
    def retrieve_relevant_content(self, query: str, top_k: int = 5, 
                                filter_metadata: Dict = None) -> List[RetrievalResult]:
        """Retrieve relevant exercise content based on query"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0].tolist()
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            results = []
            for match in search_results['matches']:
                result = RetrievalResult(
                    exercise_id=match['metadata'].get('exercise_number', 'unknown'),
                    content=match['metadata'].get('text', ''),
                    chunk_type=match['metadata'].get('chunk_type', 'unknown'),
                    metadata=match['metadata'],
                    relevance_score=match['score'],
                    section_id=match['metadata'].get('section_id')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in content retrieval: {e}")
            return []

class VerificationPipeline:
    """Verify and validate LLM responses for accuracy and safety"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        
    def verify_mathematical_accuracy(self, question: str, answer: str, 
                                   retrieved_content: List[RetrievalResult]) -> Tuple[bool, str, float]:
        """Verify mathematical accuracy of the response"""
        
        verification_prompt = f"""
        As a mathematics expert, verify if the following answer is correct:
        
        Question: {question}
        Provided Answer: {answer}
        
        Reference Material:
        {self._format_reference_content(retrieved_content)}
        
        Please:
        1. Check mathematical accuracy step by step
        2. Verify against the reference material
        3. Rate confidence (0.0-1.0)
        4. Provide brief explanation if incorrect
        
        Response format:
        ACCURATE: [YES/NO]
        CONFIDENCE: [0.0-1.0]
        EXPLANATION: [Brief explanation]
        """
        
        try:
            response = self.gemini_model.generate_content(verification_prompt)
            return self._parse_verification_response(response.text)
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, "Verification failed", 0.0
    
    def verify_content_safety(self, content: str) -> Tuple[bool, str]:
        """Check if content is safe and appropriate for educational use"""
        
        safety_prompt = f"""
        Check if this educational content is safe and appropriate:
        
        Content: {content}
        
        Check for:
        1. Age-appropriate language
        2. No harmful or inappropriate content
        3. Educationally sound approach
        
        Response: SAFE or UNSAFE with brief reason
        """
        
        try:
            response = self.gemini_model.generate_content(safety_prompt)
            return "SAFE" in response.text.upper(), response.text
        except Exception as e:
            logger.error(f"Safety verification error: {e}")
            return True, "Safety check unavailable"
    
    def _format_reference_content(self, retrieved_content: List[RetrievalResult]) -> str:
        """Format retrieved content for verification"""
        formatted = ""
        for i, result in enumerate(retrieved_content[:3]):  # Use top 3 results
            formatted += f"\nReference {i+1} ({result.chunk_type}):\n{result.content}\n"
        return formatted
    
    def _parse_verification_response(self, response: str) -> Tuple[bool, str, float]:
        """Parse verification response from LLM"""
        try:
            lines = response.strip().split('\n')
            accurate = False
            confidence = 0.0
            explanation = "No explanation provided"
            
            for line in lines:
                if line.startswith('ACCURATE:'):
                    accurate = 'YES' in line.upper()
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(re.search(r'[\d.]+', line).group())
                elif line.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip()
            
            return accurate, explanation, confidence
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return False, "Parsing failed", 0.0

class ConversationFlowManager:
    """Manage conversation flow and user engagement"""
    
    def __init__(self):
        self.phase_prompts = {
            ConversationPhase.GREETING: {
                "en": "Hi! I'm here to help you with your studies. How are you doing today?",
                "he": "שלום! אני כאן כדי לעזור לך בלימודים. איך אתה מרגיש היום?"
            },
            ConversationPhase.PERSONALIZATION: {
                "en": "Tell me about yourself - what grade are you in? What subjects interest you?",
                "he": "ספר לי על עצמך - באיזה כיתה אתה לומד? אילו מקצועות מעניינים אותך?"
            },
            ConversationPhase.ACADEMICS: {
                "en": "Great! Let's dive into your studies. What would you like to learn about today?",
                "he": "מעולה! בואו נתחיל ללמוד. על מה תרצה ללמוד היום?"
            }
        }
    
    def get_greeting_message(self, user_context: UserContext) -> str:
        """Get appropriate greeting based on conversation phase"""
        phase = user_context.current_phase
        lang = user_context.language_preference
        
        return self.phase_prompts.get(phase, {}).get(lang, 
            "Hello! How can I help you with your studies today?")
    
    def determine_next_phase(self, user_input: str, user_context: UserContext) -> ConversationPhase:
        """Determine next conversation phase based on user input"""
        current_phase = user_context.current_phase
        
        if current_phase == ConversationPhase.GREETING:
            return ConversationPhase.PERSONALIZATION
        elif current_phase == ConversationPhase.PERSONALIZATION:
            # Check if user provided personal info
            if any(keyword in user_input.lower() for keyword in ['grade', 'class', 'subject', 'math', 'science']):
                return ConversationPhase.ACADEMICS
            return ConversationPhase.PERSONALIZATION
        elif current_phase == ConversationPhase.ACADEMICS:
            return ConversationPhase.GUIDANCE
        else:
            return ConversationPhase.ACADEMICS

class EducationalChatSystem:
    """Main chat system orchestrating all components"""
    
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize components
        self.retrieval_system = RetrievalSystem()
        self.verification_pipeline = VerificationPipeline(self.gemini_model)
        self.conversation_manager = ConversationFlowManager()
        self.language_detector = LanguageDetector()
        
        # Store user contexts
        self.user_contexts: Dict[str, UserContext] = {}
        
        # Guidance sequence
        self.guidance_sequence = [
            GuidanceLevel.QUESTION,
            GuidanceLevel.HINT, 
            GuidanceLevel.SOLUTION
        ]
    
    def get_or_create_user_context(self, user_id: str) -> UserContext:
        """Get existing or create new user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(user_id=user_id)
        return self.user_contexts[user_id]
    
    def process_user_input(self, user_id: str, user_input: str) -> str:
        """Main method to process user input and generate response"""
        try:
            # Get user context
            user_context = self.get_or_create_user_context(user_id)
            
            # Detect language
            detected_lang = self.language_detector.detect_language(user_input)
            user_context.language_preference = detected_lang
            
            # Handle conversation flow
            if user_context.current_phase in [ConversationPhase.GREETING, ConversationPhase.PERSONALIZATION]:
                return self._handle_initial_conversation(user_input, user_context)
            else:
                return self._handle_academic_conversation(user_input, user_context)
                
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error. Please try again."
    
    def _handle_initial_conversation(self, user_input: str, user_context: UserContext) -> str:
        """Handle greeting and personalization phases"""
        
        # Update conversation history
        user_context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "phase": user_context.current_phase.value
        })
        
        # Extract personal information
        if user_context.current_phase == ConversationPhase.PERSONALIZATION:
            self._extract_user_info(user_input, user_context)
        
        # Determine next phase
        user_context.current_phase = self.conversation_manager.determine_next_phase(
            user_input, user_context)
        
        # Generate appropriate response
        response = self.conversation_manager.get_greeting_message(user_context)
        
        return response
    
    def _handle_academic_conversation(self, user_input: str, user_context: UserContext) -> str:
        """Handle academic questions with RAG and verification"""
        
        # Check if user is asking for help or solution
        help_keywords = ['help', 'stuck', "don't know", 'hint', 'solution', 'עזרה', 'רמז', 'פתרון']
        needs_help = any(keyword in user_input.lower() for keyword in help_keywords)
        
        if needs_help:
            return self._provide_guided_assistance(user_input, user_context)
        else:
            return self._handle_academic_query(user_input, user_context)
    
    def _provide_guided_assistance(self, user_input: str, user_context: UserContext) -> str:
        """Provide graduated assistance following the guidance sequence"""
        
        # Retrieve relevant content
        retrieved_content = self.retrieval_system.retrieve_relevant_content(
            user_input, top_k=3)
        
        if not retrieved_content:
            return self._generate_fallback_response(user_input, user_context)
        
        # Determine guidance level
        guidance_level = user_context.current_guidance_level
        
        # Generate response based on guidance level
        if guidance_level == GuidanceLevel.QUESTION:
            response = self._generate_guiding_question(user_input, retrieved_content, user_context)
            user_context.current_guidance_level = GuidanceLevel.HINT
            
        elif guidance_level == GuidanceLevel.HINT:
            response = self._generate_hint(user_input, retrieved_content, user_context)
            user_context.current_guidance_level = GuidanceLevel.SOLUTION
            
        else:  # SOLUTION
            response = self._generate_full_solution(user_input, retrieved_content, user_context)
            user_context.current_guidance_level = GuidanceLevel.QUESTION  # Reset for next question
        
        # Verify response
        is_safe, safety_msg = self.verification_pipeline.verify_content_safety(response)
        if not is_safe:
            return "I need to provide a safer response. Let me help you differently."
        
        return response
    
    def _generate_guiding_question(self, user_input: str, retrieved_content: List[RetrievalResult], 
                                 user_context: UserContext) -> str:
        """Generate a guiding question to help student think"""
        
        context_info = self._format_retrieved_content(retrieved_content)
        lang = user_context.language_preference
        
        prompt = f"""
        You are an educational tutor. A student needs help with this problem:
        {user_input}
        
        Context from educational materials:
        {context_info}
        
        Instead of giving the answer directly, ask a guiding question that helps them think through the problem.
        The question should lead them toward the solution without revealing it.
        
        Language: {"Hebrew" if lang == "he" else "English"}
        
        Respond with ONLY the guiding question.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            return "Let me help you think about this step by step. What information do you have to work with?"
    
    def _generate_hint(self, user_input: str, retrieved_content: List[RetrievalResult], 
                      user_context: UserContext) -> str:
        """Generate a hint for the student"""
        
        context_info = self._format_retrieved_content(retrieved_content)
        lang = user_context.language_preference
        
        # Look for hint content in retrieved data
        hint_content = None
        for result in retrieved_content:
            if result.chunk_type == "hint":
                hint_content = result.content
                break
        
        prompt = f"""
        You are an educational tutor. A student still needs help with this problem:
        {user_input}
        
        Context from educational materials:
        {context_info}
        
        {"Use this hint from the materials: " + hint_content if hint_content else ""}
        
        Provide a helpful hint that gives them more direction without solving the problem completely.
        The hint should be more specific than a guiding question but still require them to work.
        
        Language: {"Hebrew" if lang == "he" else "English"}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return f"💡 Hint: {response.text.strip()}"
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "💡 Hint: Try breaking the problem into smaller steps and identify what you're looking for."
    
    def _generate_full_solution(self, user_input: str, retrieved_content: List[RetrievalResult], 
                              user_context: UserContext) -> str:
        """Generate the full solution with explanation"""
        
        context_info = self._format_retrieved_content(retrieved_content)
        lang = user_context.language_preference
        
        # Look for solution content in retrieved data
        solution_content = None
        for result in retrieved_content:
            if result.chunk_type in ["solution", "full_solution"]:
                solution_content = result.content
                break
        
        prompt = f"""
        You are an educational tutor. A student needs the full solution to this problem:
        {user_input}
        
        Context from educational materials:
        {context_info}
        
        {"Reference solution: " + solution_content if solution_content else ""}
        
        Provide a complete, step-by-step solution with clear explanations for each step.
        Make sure the student understands not just the answer, but the reasoning behind it.
        
        Language: {"Hebrew" if lang == "he" else "English"}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Verify mathematical accuracy
            is_accurate, explanation, confidence = self.verification_pipeline.verify_mathematical_accuracy(
                user_input, response.text, retrieved_content)
            
            if not is_accurate and confidence > 0.7:
                return f"Let me reconsider this problem. {explanation}"
            
            return f"📚 Complete Solution:\n{response.text.strip()}"
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return "I'm having trouble generating the solution right now. Please try rephrasing your question."
    
    def _handle_academic_query(self, user_input: str, user_context: UserContext) -> str:
        """Handle general academic questions"""
        
        retrieved_content = self.retrieval_system.retrieve_relevant_content(
            user_input, top_k=5)
        
        if not retrieved_content:
            return self._generate_fallback_response(user_input, user_context)
        
        context_info = self._format_retrieved_content(retrieved_content)
        lang = user_context.language_preference
        
        prompt = f"""
        You are an educational assistant. Answer this student's question:
        {user_input}
        
        Use this context from educational materials:
        {context_info}
        
        Provide a clear, educational response appropriate for their level.
        Language: {"Hebrew" if lang == "he" else "English"}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Verify safety
            is_safe, _ = self.verification_pipeline.verify_content_safety(response.text)
            if not is_safe:
                return "Let me provide a more appropriate educational response."
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error handling academic query: {e}")
            return "I'm sorry, I couldn't process your question right now. Please try again."
    
    def _extract_user_info(self, user_input: str, user_context: UserContext):
        """Extract user information from personalization input"""
        
        # Simple extraction - can be enhanced with NLP
        input_lower = user_input.lower()
        
        # Extract grade/class
        grade_patterns = [r'grade (\d+)', r'class (\d+)', r'כיתה ([א-ת]|\d+)']
        for pattern in grade_patterns:
            match = re.search(pattern, input_lower)
            if match:
                user_context.academic_level = match.group(1)
                break
        
        # Extract subjects
        subjects = ['math', 'mathematics', 'science', 'physics', 'chemistry', 'biology', 
                   'מתמטיקה', 'מדעים', 'פיזיקה', 'כימיה', 'ביולוגיה']
        for subject in subjects:
            if subject in input_lower:
                user_context.interests.append(subject)
    
    def _format_retrieved_content(self, retrieved_content: List[RetrievalResult]) -> str:
        """Format retrieved content for prompts"""
        if not retrieved_content:
            return "No relevant content found."
        
        formatted = ""
        for i, result in enumerate(retrieved_content[:3]):  # Use top 3 results
            formatted += f"\nContent {i+1} (Type: {result.chunk_type}, Relevance: {result.relevance_score:.2f}):\n"
            formatted += f"{result.content}\n"
        
        return formatted
    
    def _generate_fallback_response(self, user_input: str, user_context: UserContext) -> str:
        """Generate fallback response when no content is retrieved"""
        lang = user_context.language_preference
        
        if lang == "he":
            return "מצטער, לא מצאתי מידע רלוונטי במסד הנתונים שלי. האם תוכל לנסח את השאלה בצורה אחרת?"
        else:
            return "I couldn't find relevant information in my database. Could you please rephrase your question or provide more details?"
    
    def check_inactivity(self, user_id: str, timeout_seconds: int = 30) -> Optional[str]:
        """Check if user has been inactive and provide appropriate message"""
        user_context = self.user_contexts.get(user_id)
        if not user_context or not user_context.conversation_history:
            return None
        
        last_interaction = user_context.conversation_history[-1]["timestamp"]
        last_time = datetime.fromisoformat(last_interaction)
        
        if (datetime.now() - last_time).total_seconds() > timeout_seconds:
            lang = user_context.language_preference
            if lang == "he":
                return "אני עדיין כאן אם אתה צריך עזרה! יש לך שאלות נוספות?"
            else:
                return "I'm still here if you need help! Do you have any other questions?"
        
        return None

# Example usage and testing
if __name__ == "__main__":
    # Initialize the chat system
    chat_system = EducationalChatSystem()
    
    # Simulate a conversation
    user_id = "test_user_1"
    
    print("=== Educational Chat System Demo ===\n")
    
    # Initial greeting
    response = chat_system.process_user_input(user_id, "Hello")
    print(f"Bot: {response}\n")
    
    # Personalization
    response = chat_system.process_user_input(user_id, "I'm in grade 8 and I like math")
    print(f"Bot: {response}\n")
    
    # Academic question
    response = chat_system.process_user_input(user_id, "I need help with algebra")
    print(f"Bot: {response}\n")
    
    # Request for help
    response = chat_system.process_user_input(user_id, "I'm stuck on solving 2x + 5 = 13")
    print(f"Bot: {response}\n")
    
    # Continue asking for help (should get hint)
    response = chat_system.process_user_input(user_id, "I still don't understand")
    print(f"Bot: {response}\n")
    
    # Ask for solution
    response = chat_system.process_user_input(user_id, "Can you show me the solution?")
    print(f"Bot: {response}\n")