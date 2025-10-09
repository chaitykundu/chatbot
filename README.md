# 📘 IvriTutor Demo — RAG Math Tutor

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot designed to assist students with 8th-grade math exercises.  
It provides **interactive tutoring** with bilingual (Hebrew + English) support, progressive guidance, and vector-based retrieval.  

---

## 🚀 Features  

- 🤖 **Interactive Math Tutor Chatbot** with FSM-based flow  
- 🧾 **Retrieval from JSON exercise bank**  
- 💡 **Progressive Assistance Strategy**  
  - Encouragement → Guiding Questions → Hints → Full Solution  
- 🔎 **Pinecone for efficient vector search**  
- 🌐 **Powered by Google Gemini/Qwen** for natural language understanding  
- 💬 **Small talk & personalization flow** (warm greeting → personal chat → academics)  
- 🌍 **Bilingual Support** (Hebrew RTL + English LTR)  
- 📊 **SVG Handling**: renders and describes math diagrams automatically  
- 💤 **Inactivity Timer** with typing detection  
- ❓ **Doubt Clearing Session** after exercises  
- 📝 **Automatic Lesson Summary** at the end of each session  

---

## 📋 Prerequisites  

- **Python**: Version 3.8+  
- **API Keys**:  
  - Pinecone → [Get API Key](https://www.pinecone.io/)  
  - Google Gemini (Qwen) → Google AI Studio / Google Cloud Console  
- (Optional) Virtual Environment for dependency isolation  

---

## ⚙️ Setup  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/chatbot.git
   cd chatbot
   ```

2. **Create a Virtual Environment (Recommended)**  
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**  
   Create a `.env` file in the project root:  
   ```env
   PINECONE_API_KEY=your_actual_pinecone_api_key_here
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

---

## 📚 Preparing the Knowledge Base  

- Place parsed exercise JSON files in:  
  ```
  chatbot/parsed_outputs/
  ```

- Generate embeddings (first run or after data changes):  
  ```bash
  python chatbot/parsed_outputs/embedding.py
  ```

- Index embeddings into Pinecone:  
  ```bash
  python chatbot/parsed_outputs/index_embedding.py
  ```

⚠️ Make sure the index name in `index_embedding.py` matches `INDEX_NAME` in `chatbot.py` (default: `exercise-embedding1`).  

---

## 💬 Running the Chatbot  

Activate your environment and run:  

```bash
python chatbot.py
```

### Interaction Flow  

1. The chatbot greets you (friendly small talk)  
2. Asks about personal life → transitions to academics  
3. Student selects grade and topic  
4. Tutor presents math exercises  

You can:  
- Provide an answer → chatbot evaluates  
- Ask for a hint → chatbot gives progressive guidance  
- Request the solution → chatbot only provides it after guiding steps  

---

## 🎓 Project Instructions  

### 1. Chat Flow & Engagement  
- Start with greetings: *"Hey!", "How are you doing?"*  
- Ask about personal life (sports, job search, etc.)  
- Transition smoothly to academics (*“What did you learn recently?”*)  

### 2. Progressive Guidance  
- Student asks for help → chatbot follows sequence:  
  **Guiding question → Another guiding question → Hint → Full solution (step-by-step)**  

### 3. Bilingual Support  
- Detects language automatically  
- Hebrew → **Right-to-Left (RTL)**  
- English → **Left-to-Right (LTR)**  
- Math formulas always **LTR**  

### 4. Handling Full Solution  
- Full solution shown **only after guidance sequence** is completed.  

### 5. Doubt Clearing Session  
- After at least 2 exercises, the chatbot asks:  
  *“Do you have any questions or doubts about this topic?”*  
- If student asks a question, chatbot retrieves relevant context from Pinecone and provides a **step-by-step explanation**.  
- If no doubts are raised, the chatbot acknowledges and closes the session.  

### 6. Lesson Summary  
- At the end of each session, the chatbot provides a **positive summary** of what the student achieved.  
- The summary includes encouragement and sometimes light humor for motivation.  

---

## ✅ Example Session  

```
👩 Student: Hi!  
🤖 Tutor: Hey! How’s it going?  
👩 Student: All good, just had basketball practice.  
🤖 Tutor: Nice! Do you have a test coming up?  
...  
🤖 Tutor: Here’s a question on geometry 📘  
👩 Student: I don’t know the answer.  
🤖 Tutor: 🤔 Let me ask you this…  
...  
💡 Hint: Think about the slope of the line.  
✅ Solution: Step 1 → Step 2 → Final Answer  

❓ Tutor: Great work! Do you have any questions about this topic?  
👩 Student: Yes, how do you calculate slope from coordinates?  
🤖 Tutor: Good question! Here’s the step-by-step explanation…  

📝 Summary: Awesome job today! You practiced geometry, solved tough problems, and even asked great questions. Remember: math is just puzzles in disguise — and you’re a puzzle master! 🎉
```


