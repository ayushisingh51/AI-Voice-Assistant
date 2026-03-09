import streamlit as st
import audio_recorder_streamlit as ars
import whisper
import ollama
from gtts import gTTS
import os
import json
import time
import psutil
import requests
from duckduckgo_search import DDGS
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



# ---------------- PDF PROCESSING ----------------

from chromadb.config import Settings

def process_pdf(pdf_file):

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="pdf_db",
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )

    return vectorstore


def ask_pdf(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the following document context to answer the question.

{context}

Question: {question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# ---------------- WEATHER ----------------

def get_weather(city):

    url = f"https://wttr.in/{city}?format=j1"

    response = requests.get(url).json()

    current = response["current_condition"][0]

    temp = current["temp_C"]
    desc = current["weatherDesc"][0]["value"]

    return f"The temperature in {city} is {temp}°C with {desc}."


# ---------------- TIME & DATE ----------------

def get_time():
    now = datetime.now().strftime("%H:%M")
    return f"The current time is {now}"


def get_date():
    today = datetime.now().strftime("%A, %d %B %Y")
    return f"Today is {today}"


# ---------------- SYSTEM INFO ----------------

def get_system_info():

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    return f"CPU usage is {cpu}% and RAM usage is {ram}%."


# ---------------- APPLICATION CONTROL ----------------

apps = {
    "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "vscode": "C:\\Users\\amrit\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
    "notepad": "C:\\Windows\\system32\\notepad.exe",
    "calculator": "C:\\Windows\\system32\\calc.exe",
    "explorer": "C:\\Windows\\explorer.exe"
}


def open_application(command):

    for app in apps:
        if app in command:
            os.startfile(apps[app])
            return f"Opening {app}."

    return "Sorry, I couldn't find that application."


# ---------------- WEB SEARCH ----------------

def search_web(query):

    with DDGS() as ddgs:

        results = ddgs.text(query, max_results=3)

        text_results = []

        for r in results:
            text_results.append(f"{r['title']}: {r['body']}")

    return "\n".join(text_results)


# ---------------- CHAT SAVE ----------------

def save_chat():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.messages, f)


# ---------------- WHISPER MODEL ----------------

@st.cache_resource
def load_whisper():
    return whisper.load_model("small")


model = load_whisper()


# ---------------- SPEECH TO TEXT ----------------

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]


# ---------------- AI RESPONSE ----------------

def fetch_ai_response(input_text):

    text = input_text.lower()

    if "vectorstore" in st.session_state:
        return ask_pdf(st.session_state.vectorstore, input_text)

    if "weather" in text:
        words = text.split()
        if "in" in words:
            city = words[words.index("in") + 1]
            return get_weather(city)

    if "open" in text:
        return open_application(text)

    if "time" in text:
        return get_time()

    if "date" in text:
        return get_date()

    if "cpu" in text or "system usage" in text:
        return get_system_info()

    search_keywords = ["news", "latest", "who is", "what is happening"]

    if any(word in text for word in search_keywords):

        web_results = search_web(input_text)

        prompt = f"""
Use the following web search results to answer the question.

{web_results}

Question: {input_text}
"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    response = ollama.chat(
        model="llama3",
        messages=st.session_state.messages + [{"role": "user", "content": input_text}]
    )

    return response["message"]["content"]


# ---------------- TEXT TO SPEECH ----------------

def text_to_audio(text):

    audio_file = "response.mp3"

    tts = gTTS(text=text, lang="en")
    tts.save(audio_file)

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes


# ---------------- STREAMLIT APP ----------------

def main():

    st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️")

    st.title("🎙️ AI Voice Assistant")
    st.info(
"""
🎙️ **AI Voice Assistant**

Built using **Whisper + Ollama (Llama3) + Streamlit**

This assistant supports:
- Voice interaction
- AI responses using a local LLM
- Web search
- Weather updates
- System monitoring
- Desktop automation
- Document question answering
"""
)
    with st.expander("💡 Example Commands You Can Try"):

        st.markdown("""
Try asking the assistant:

**Desktop Commands**
- Open Chrome
- Open VS Code
- Open Notepad

**Information**
- What is the weather in Delhi?
- What time is it?
- What is today's date?

**System Monitoring**
- Show CPU usage
- Show system usage

**Web Search**
- Latest AI news
- Who is Elon Musk?

**Document AI**
- Upload a PDF and ask questions about it
""")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_pdf = st.file_uploader("📄 Upload a PDF to ask questions about it", type="pdf")

    if uploaded_pdf and "vectorstore" not in st.session_state:

        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        st.success("PDF uploaded successfully!")

        vectorstore = process_pdf("uploaded.pdf")

        st.session_state.vectorstore = vectorstore

        st.info("You can now ask questions about the document.")

    if st.button("🗑 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.write("Speak with the assistant using your microphone.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.divider()

    st.subheader("🎤 Record your question")

    recorded_audio = ars.audio_recorder()

    if recorded_audio:

        audio_file = "audio.wav"

        with open(audio_file, "wb") as f:
            f.write(recorded_audio)

        st.info("Transcribing audio...")

        transcribed_text = transcribe_audio(audio_file)

        st.session_state.messages.append(
            {"role": "user", "content": transcribed_text}
        )

        with st.chat_message("user"):
            st.write(transcribed_text)

        st.info("Generating AI response...")

        ai_response = fetch_ai_response(transcribed_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response}
        )

        save_chat()

        with st.chat_message("assistant"):

            placeholder = st.empty()

            typed_text = ""

            for char in ai_response:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.01)

        response_audio = text_to_audio(ai_response)

        st.audio(response_audio, format="audio/mp3")


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    main()