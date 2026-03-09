# AI Voice Assistant

An intelligent voice-controlled AI assistant built using Python, Streamlit, Whisper, and Ollama (Llama3).  
The assistant can understand spoken queries, generate AI responses locally, perform system tasks, search the web, and answer questions from uploaded PDF documents.

---

## Features

- Voice interaction using Whisper (speech-to-text)
- AI responses powered by Llama3 through Ollama
- Voice replies using gTTS
- Web search capability
- Weather information retrieval
- Time and date queries
- Desktop automation (open applications)
- System monitoring (CPU and RAM usage)
- PDF document question answering using LangChain and ChromaDB (RAG)
- Chat history 

---

## Tech Stack

Python  
Streamlit  
Whisper  
Ollama (Llama3)  
LangChain  
ChromaDB  
Sentence Transformers  
DuckDuckGo Search  

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ayushisingh51/AI-Voice-Assistant.git
cd AI-Voice-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download Ollama from:

https://ollama.com

Then pull the model:

```bash
ollama pull llama3
```

Run the model:

```bash
ollama run llama3
```

### 4. Run the application

```bash
streamlit run app.py
```

The assistant will open in your browser.

---

## Example Commands

Some example queries you can try:

Open Chrome  
What is the weather in Delhi?  
What time is it?  
Show CPU usage  
Latest AI news  

You can also upload a PDF and ask questions about it.

---


## Author

Ayushi Singh  
GitHub: https://github.com/ayushisingh51
