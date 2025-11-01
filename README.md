Here’s a copy-paste **`README.md`** you can drop into your repo for the Streamlit Community Cloud deployment.

````md
# 📄 PDF RAG — Free CPU Deploy (Streamlit)

Chat with your own PDFs using a lightweight, **free-to-host** RAG pipeline on **Streamlit Community Cloud**.  
No Ollama or GPUs required — we run a **tiny Transformers chat model on CPU**, FAISS for retrieval, and maintain **conversational memory** between turns.

---

## ✨ Features

- **Upload a PDF** → split into chunks → **MiniLM** embeddings → **FAISS** vector store
- **Conversational memory** via LangChain `RunnableWithMessageHistory`
- **Local-friendly models** (CPU): `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default) or `Qwen/Qwen2.5-0.5B-Instruct`
- Streamlit **chat UI** with source citations (page numbers)

---

## 🧩 Architecture (RAG with memory)

1. **PDF Loader (PyPDF)** → **Chunking (RecursiveCharacterTextSplitter)**
2. **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
3. **Index:** FAISS (in-memory)
4. **Retriever:** top-k similarity
5. **LLM (CPU):** Transformers pipeline wrapped by `ChatHuggingFace`
6. **Memory:** `RunnableWithMessageHistory` + `MessagesPlaceholder("chat_history")`

---

## 🚀 One-click deploy (Streamlit Community Cloud)

1. **Push to GitHub** a repo with:
   - `app.py` (the Streamlit app using Transformers backend)
   - `requirements.txt`:
     ```
     streamlit
     langchain
     langchain-community
     langchain-text-splitters
     langchain-huggingface
     faiss-cpu
     sentence-transformers
     pypdf
     transformers>=4.44
     accelerate
     torch
     ```
   - *(optional)* `README.md` (this file)

2. Go to **Streamlit Community Cloud** → **New app** → select your repo & branch → set **Main file path** to `app.py` → **Deploy**.

3. Wait for the first build (installs dependencies and downloads the model).  
   You’ll get a **public URL** anyone can use.

> **Note:** If you pick a gated HF model (e.g., some Qwen variants), add an HF token.

---

## 🔐 (Optional) Hugging Face token

Some models (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) may require a token.

1. In your Streamlit app page, **Settings → Secrets → New secret**:
   - **Key:** `HF_TOKEN`
   - **Value:** your Hugging Face access token

2. The app reads it automatically and passes it to `from_pretrained`.

Pick **TinyLlama** to avoid tokens altogether.

---

## 🧪 Local development

### Using `uv` (recommended)

# 1) Create project
uv venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install
uv pip install -r requirements.txt

# 3) Run
uv run streamlit run app.py

### Using pip

python -m venv .venv
# activate it, then:
pip install -r requirements.txt
streamlit run app.py

---

## ⚙️ App settings

Inside the app sidebar you can control:

* **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fastest on free CPU) or `Qwen/Qwen2.5-0.5B-Instruct`
* **Temperature**: creativity vs. determinism
* **Top-k**: retrieved chunks
* **Chunk size / overlap**: affects recall vs. speed

---

## 🧠 Memory details

We use:

* `ChatPromptTemplate` with `MessagesPlaceholder("chat_history")`
* `RunnableWithMessageHistory` to persist turns in session
* Streamlit’s `st.session_state` to keep memory for the current browser session

Click **“🧹 Reset conversation”** to clear history.

---

## 🔍 FAQ

**Q: Cold start is slow.**
A: First run downloads the model and builds wheels; subsequent runs are faster thanks to caching.

**Q: App crashed with out-of-memory / very slow answers.**
A: Use **TinyLlama** (1.1B), keep `max_new_tokens` modest (the app defaults to 512), and avoid very large PDFs.

**Q: Do I need a GPU?**
A: No, this runs on CPU (free tier).

**Q: Can I keep Ollama/Qwen 1.5B?**
A: Streamlit Cloud free tier can’t run Ollama. Use a tiny Transformers model here, or host Ollama on your own VM (not free) and point a separate app at it.

---

## 🧱 Limitations

* Free CPU only — large models (≥7B) are not feasible here.
* In-memory FAISS is rebuilt after each reload; not a persistent database.
* Session memory is per user session; not a multi-user chat history store.

---

## 🔧 Tech stack

* **Streamlit** UI
* **LangChain** (LCEL runnables, memory)
* **FAISS** vector store
* **Sentence-Transformers** MiniLM embeddings
* **Transformers** (TinyLlama / Qwen 0.5B) with `ChatHuggingFace`

---

## 📜 License

MIT (or your preferred license). Add a `LICENSE` file if you need one.

---

## 🤝 Contributing

PRs welcome! Ideas:

* Add streaming tokens in the UI
* Persistent vector DB (Chroma, SQLite-FAISS)
* Multi-file ingestion & background indexing
* Rerankers (e.g., `bge-reranker-base`) for better retrieval quality

```

If you want, I can also generate a small **template repo** (all files ready) that you can push to GitHub directly.
::contentReference[oaicite:0]{index=0}
```
