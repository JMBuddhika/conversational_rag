import os
import tempfile
from pathlib import Path
from uuid import uuid4
from operator import itemgetter

import streamlit as st

# LangChain community imports (modern locations)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory

# LCEL (modern LangChain composables)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="PDF RAG (with Memory)", page_icon="📄", layout="wide")
st.title("📄 PDF RAG — with Conversational Memory")

st.write(
    "Upload a PDF, embed with MiniLM + FAISS, and chat with a **local** model via Ollama. "
    "This app keeps a conversation memory so you can ask follow-ups."
)

# ---------------------------
# Sidebar: model + retriever options
# ---------------------------
st.sidebar.header("⚙️ Settings")
ollama_model = st.sidebar.text_input(
    "Ollama model tag",
    value="qwen2.5:1.5b",
    help="Pick an installed tag from `ollama list` (e.g., qwen2.5:0.5b-instruct / qwen2.5:7b-instruct).",
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
k = st.sidebar.slider("Top-k retrieved chunks", 1, 10, 4, 1)
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 100, 10)

# Base URL (helpful if Ollama isn’t on localhost)
ollama_base_url = "http://127.0.0.1:11434"

# ---------------------------
# Upload PDF
# ---------------------------
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriever_from_pdf(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})
    return retriever

def make_llm():
    return ChatOllama(
        model=ollama_model,
        temperature=float(temperature),
        base_url=ollama_base_url,
        # num_ctx=4096, # Uncomment / adjust if using larger models
    )

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Prompt with chat history
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers using ONLY the provided context. "
            "If the context is insufficient, say you don't know. Be concise."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# Session & memory wiring
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "stores" not in st.session_state:
    st.session_state.stores = {}  # {session_id: ChatMessageHistory()}

def get_history(session_id: str) -> ChatMessageHistory:
    stores = st.session_state.stores
    if session_id not in stores:
        stores[session_id] = ChatMessageHistory()
    return stores[session_id]

# UI: reset chat memory
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🧹 Reset conversation"):
        st.session_state.stores[st.session_state.session_id] = ChatMessageHistory()
        st.rerun()

# ---------------------------
# Build index (once PDF is uploaded)
# ---------------------------
retriever = None
if uploaded is not None:
    with st.spinner("Building FAISS index…"):
        retriever = build_retriever_from_pdf(uploaded.read())
        st.success("Index ready!")

    llm = make_llm()
    extract_q = itemgetter("question")

    # Keep ALL incoming keys (including chat_history) and just add 'context'
    base = RunnablePassthrough.assign(
        context=extract_q | retriever | format_docs
    )

    rag_chain = (
        base  # inputs : question + chat_history + context
        | PROMPT
        | llm
        | StrOutputParser()
    )

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # Render previous messages (from memory)
    st.subheader("💬 Chat")
    history = get_history(st.session_state.session_id).messages
    for m in history:
        role = "user" if m.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(m.content)

    # Chat input
    user_msg = st.chat_input("Ask about the PDF…")
    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.spinner("Thinking…"):
            answer = conversational_rag.invoke(
                {"question": user_msg},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            sources = retriever.invoke(user_msg)

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.markdown("**📚 Sources**")
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata or {}
                page = meta.get("page", "N/A")
                src = meta.get("source", "uploaded.pdf")
                st.markdown(f"- {i}. {Path(src).name} — page {page+1 if isinstance(page, int) else page}")
else:
    st.info("👆 Upload a PDF to begin.")
