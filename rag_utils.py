"""
rag_utils.py
Core RAG logic for StudyMate AI:
- process a PDF into a per-book Chroma vectorstore
- load an existing vectorstore
- build a RAG chain that returns both the answer and the source chunks
"""

import os
import hashlib
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE_DIR = "studymate_data"
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstores")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

_embedding_model = None


def get_embedding_model():
    """Cache the embedding model so it's loaded only once per process."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model


def make_book_id(file_name: str, file_bytes: bytes) -> str:
    """Stable id for a book: sanitized filename + short content hash."""
    safe_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(file_name)[0])
    content_hash = hashlib.md5(file_bytes).hexdigest()[:8]
    return f"{safe_name}_{content_hash}"


def book_vectorstore_dir(book_id: str) -> str:
    return os.path.join(VECTOR_DIR, book_id)


def is_book_processed(book_id: str) -> bool:
    return os.path.isdir(book_vectorstore_dir(book_id)) and len(
        os.listdir(book_vectorstore_dir(book_id))
    ) > 0


def process_pdf(file_name: str, file_bytes: bytes) -> str:
    """
    Save the uploaded PDF, split it into chunks, embed it, and persist a
    Chroma vectorstore for it. Returns the book_id. Skips reprocessing if
    this exact file was already processed before.
    """
    book_id = make_book_id(file_name, file_bytes)
    vs_dir = book_vectorstore_dir(book_id)

    if is_book_processed(book_id):
        return book_id  # already done, nothing to do

    # Save the raw PDF to disk
    pdf_path = os.path.join(PDF_DIR, f"{book_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    # Load + split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed + persist
    if os.path.isdir(vs_dir):
        shutil.rmtree(vs_dir)  # clean any partial previous attempt
    os.makedirs(vs_dir, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=vs_dir,
    )

    return book_id


def delete_book(book_id: str):
    """Remove a book's PDF and vectorstore from disk."""
    vs_dir = book_vectorstore_dir(book_id)
    pdf_path = os.path.join(PDF_DIR, f"{book_id}.pdf")
    if os.path.isdir(vs_dir):
        shutil.rmtree(vs_dir)
    if os.path.isfile(pdf_path):
        os.remove(pdf_path)


def load_retriever(book_id: str, k: int = 4, fetch_k: int = 10, lambda_mult: float = 0.5):
    vs_dir = book_vectorstore_dir(book_id)
    vectorstore = Chroma(
        persist_directory=vs_dir,
        embedding_function=get_embedding_model(),
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def format_docs(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Unknown page"
        parts.append(f"[{page_label}]\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain(retriever, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.2):
    """
    Returns a runnable that, given a plain question string, outputs a dict:
        {"question": ..., "docs": [Document, ...], "context": str, "answer": str}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are StudyMate, a helpful AI study assistant.
Use only the provided context from the uploaded book/document to answer the question.
If the answer is not present in the context, say exactly:
"I could not find the answer in the document."
Keep answers clear, well structured, and exam-friendly (use bullet points when useful).""",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    llm = ChatGroq(model=model_name, temperature=temperature)

    answer_chain = prompt | llm | StrOutputParser()

    chain = (
        RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(answer=answer_chain)
    )
    return chain