import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from elevenlabs.client import ElevenLabs
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
import whisper
import sounddevice as sd
from scipy.io.wavfile import write

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

print("ELEVENLABS_API_KEY loaded:", ELEVENLABS_API_KEY is not None)


# Paths
VECTOR_STORE_DIR = "vectorstore"
DOC_FILE = "E:/customer supprot chatbog/company_docs.txt"
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "company_vectorstore")

# Init Groq
groq_client = Groq(api_key=GROQ_API_KEY)


def load_vectorstore():
    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(VECTOR_STORE_PATH) and not st.session_state.knowledge_updated:
        try:
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
            )
            st.session_state.vectorstore = vectorstore
            return vectorstore
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")

    try:
        with open(DOC_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        for file_info in st.session_state.uploaded_files:
            text += f"\n\n{file_info['content']}"

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(text)

        vectorstore = FAISS.from_texts(docs, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)

        st.session_state.vectorstore = vectorstore
        st.session_state.knowledge_updated = False
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None


def whisper_transcribe(duration=3, samplerate=16000):
    import os
    os.environ["PATH"] += os.pathsep + r"C:\Users\user\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    temp_audio_file = tempfile.mktemp(suffix=".wav")
    write(temp_audio_file, samplerate, audio)

    model = whisper.load_model("base")
    result = model.transcribe(temp_audio_file)

    os.remove(temp_audio_file)
    return result["text"]



def elevenlabs_tts(text, lang="en", voice_id="EXAVITQu4vr4xnSDxMaL"):
    try:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        client = ElevenLabs(api_key=api_key)

        model_id = "eleven_multilingual_v2" if lang != "en" else "eleven_monolingual_v1"

        response = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id
        )

        if hasattr(response, "__iter__") and not isinstance(response, (bytes, bytearray)):
            audio_bytes = b"".join(response)
        else:
            audio_bytes = response

        temp_audio_file = tempfile.mktemp(suffix=".mp3")
        with open(temp_audio_file, "wb") as f:
            f.write(audio_bytes)

        return temp_audio_file
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None





def extract_text_from_file(uploaded_file):
    file_content = uploaded_file.getvalue()
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            doc = fitz.open(tmp_file_path)
            text = "".join(page.get_text() for page in doc)

            os.unlink(tmp_file_path)
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    elif file_extension in ["txt", "md", "html"]:
        return file_content.decode("utf-8")
    else:
        st.warning(f"Unsupported file type: {file_extension}")
        return ""


def get_ai_response(query, lang="en"):
    vectorstore = load_vectorstore()
    if not vectorstore:
        return "Error loading knowledge base. Please try again.", []

    query_en = GoogleTranslator(source=lang, target='en').translate(query) if lang != "en" else query

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query_en)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query_en}"""

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        answer_en = response.choices[0].message.content.strip()
        answer = GoogleTranslator(source='en', target=lang).translate(answer_en) if lang != "en" else answer_en
        sources = [doc.page_content[:150] + "..." for doc in docs]

        st.session_state.conversation_history.append((query_en, answer_en))

        return answer, sources
    except Exception as e:
        return f"Error generating response: {str(e)}", []
