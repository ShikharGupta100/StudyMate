# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# embedding_model = HuggingFaceEmbeddings(

# )


# vectorstore = Chroma(
#     persist_directory="chroma_db"
#     ,
#     embedding_function=embedding_model
# )

# retriever = vectorstore.as_retriever(
#     search_type = "mmr",
#     search_kwargs = {
#         "k":4,
#         "fetch_k":10,
#         "lambda_mult":0.5
#     }
# )

# llm = ChatMistralAI(model ="mistral-small-2506")


# prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             """
#               You are a helpful AI assisstant.
#               Use only the provided context to answer the question.
#               If the answer is not present in the context,
#               say: "I could not find the answer in the document."
# """
#         ),
#         (
#             "human",
#             """
#             Context :{context}
#             Question:{question}
# """
#         )
#     ]
# )

import streamlit as st
from dotenv import load_dotenv

import rag_utils as ru
import history_utils as hu

load_dotenv()

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "active_book_id" not in st.session_state:
    st.session_state.active_book_id = None
if "retriever_cache" not in st.session_state:
    st.session_state.retriever_cache = {}  # book_id -> retriever
if "chain_cache" not in st.session_state:
    st.session_state.chain_cache = {}  # book_id -> chain

history = hu.load_history()

# ---------------------------------------------------------------------------
# Sidebar: upload + book library
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 StudyMate AI")
    st.caption("Chat with your books & notes")

    st.subheader("Upload a new PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("➕ Process & Add Book", use_container_width=True):
            file_bytes = uploaded_file.getvalue()
            with st.spinner(f"Reading and indexing '{uploaded_file.name}'... this can take a minute"):
                book_id = ru.process_pdf(uploaded_file.name, file_bytes)
                hu.register_book(book_id, uploaded_file.name)
                st.session_state.active_book_id = book_id
            st.success(f"'{uploaded_file.name}' is ready!")
            st.rerun()

    st.divider()
    st.subheader("Your books")

    history = hu.load_history()  # refresh after any add

    if not history:
        st.info("No books yet. Upload a PDF above to get started.")
    else:
        for book_id, meta in history.items():
            col1, col2 = st.columns([4, 1])
            is_active = book_id == st.session_state.active_book_id
            label = ("👉 " if is_active else "") + meta["name"]
            with col1:
                if st.button(label, key=f"select_{book_id}", use_container_width=True):
                    st.session_state.active_book_id = book_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{book_id}", help="Delete this book"):
                    ru.delete_book(book_id)
                    hu.remove_book(book_id)
                    if st.session_state.active_book_id == book_id:
                        st.session_state.active_book_id = None
                    st.session_state.retriever_cache.pop(book_id, None)
                    st.session_state.chain_cache.pop(book_id, None)
                    st.rerun()

    st.divider()
    if st.session_state.active_book_id:
        if st.button("🧹 Clear chat history for this book", use_container_width=True):
            hu.clear_book_chats(st.session_state.active_book_id)
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
active_id = st.session_state.active_book_id

if not active_id:
    st.title("Welcome to StudyMate AI 📚")
    st.write(
        "Upload a PDF (textbook, notes, research paper) from the sidebar to start "
        "asking questions about it. Everything you process stays saved so you can "
        "come back to it later."
    )
    st.stop()

active_name = history.get(active_id, {}).get("name", active_id)
st.title(f"💬 Chatting with: {active_name}")

# Build / fetch retriever + chain (cached per book for this session)
if active_id not in st.session_state.chain_cache:
    with st.spinner("Loading book..."):
        retriever = ru.load_retriever(active_id)
        chain = ru.build_chain(retriever)
        st.session_state.retriever_cache[active_id] = retriever
        st.session_state.chain_cache[active_id] = chain

chain = st.session_state.chain_cache[active_id]

# Render past chat history for this book
past_chats = hu.load_history().get(active_id, {}).get("chats", [])
for turn in past_chats:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])

# Chat input
question = st.chat_input("Ask something about this book...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke(question)
            answer = result["answer"]
            sources = result["docs"]

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources used"):
                for i, doc in enumerate(sources, start=1):
                    page = doc.metadata.get("page")
                    page_label = f"Page {page + 1}" if isinstance(page, int) else "Unknown page"
                    st.markdown(f"**{i}. {page_label}**")
                    st.caption(doc.page_content[:300] + "...")

    hu.add_chat(active_id, question, answer)