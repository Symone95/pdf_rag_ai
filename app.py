import streamlit as st
from rag_engine import add_documents, get_db_stats, reset_database, collection, get_file_hash
import os
from tools import agent_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="Local RAG Chat", layout="wide")  # Titolo del tab

st.sidebar.header("🗄️ Database")  # Titolo sidebar a sinistra

# --- STATS ---
doc_count = get_db_stats()
st.sidebar.write(f"Chunk salvati: **{doc_count}**")

# --- DOC FILTER ---
st.sidebar.subheader("📄 Filtra per documento")

docs = collection.get()["metadatas"]
doc_names = list(set([m["file"] for m in docs])) if docs else []

selected_doc = st.sidebar.selectbox(
    "Scegli documento",
    ["Tutti"] + doc_names
)

if selected_doc == "Tutti":
    selected_doc = None


# --- RESET DB ---
if st.sidebar.button("🗑️ Reset database"):
    reset_database()
    st.sidebar.success("Database cancellato!")
    st.rerun()

st.title("🤖 Chat con i tuoi PDF (Ollama + ChromaDB)")

# --- Sidebar Upload PDF ---
st.sidebar.header("📄 Carica documento")

uploaded_files = st.sidebar.file_uploader("Carica PDF", type="pdf", accept_multiple_files=True)

# memoria sessione per evitare doppia indicizzazione
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if uploaded_files:
    new_files = []
    file_hash_map = {}

    for f in uploaded_files:
        file_hash = get_file_hash(f)

        if file_hash not in st.session_state.processed_files:
            new_files.append(f)
            file_hash_map[f.name] = file_hash

    if new_files:
        with st.spinner("Indicizzazione automatica..."):
            add_documents(new_files, file_hash_map)

        for f in new_files:
            st.session_state.processed_files.add(f.name)

        st.sidebar.success("Documenti indicizzati!")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra messaggi precedenti
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input utente
query = st.chat_input("Fai una domanda sui documenti")

if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Sto pensando..."):

        response_placeholder = st.chat_message("assistant").empty()
        full_response = ""

        for chunk in agent_answer(query, selected_doc, st.session_state.messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})