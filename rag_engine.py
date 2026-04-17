import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re
import logging
from datetime import datetime

from pdf_loader import chunk_text, load_pdf_paginated

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Embedding model (caricato una volta sola)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma persistente
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

def get_files_with_upload_date():
    data = collection.get()

    if not data["metadatas"]:
        return {}

    file_dates = {}

    for m in data["metadatas"]:
        file = m["file"]
        date = m.get("uploaded_at")

        # prendiamo la prima occorrenza (tutti i chunk hanno la stessa)
        if file not in file_dates and date:
            file_dates[file] = date

    return file_dates

def group_by_file(structured_sources):
    files = {}

    for s in structured_sources:
        if s["file"] not in files:
            files[s["file"]] = s

    return list(files.values())

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

def make_source_link(file, page):
    # return f"[📄 {file} - pag.{page}](#)"
    return f'<a href="docs/{file}#page={page}" target="_blank">📄 {file} - pag.{page}</a>'

def extract_keywords(query):
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if len(w) > 4]


def get_files_in_db():
    data = collection.get(include=["metadatas"])

    files = {}
    for meta in data["metadatas"]:
        files[meta["file_hash"]] = meta["file"]

    # lista pulita
    return list(files.values())

def get_db_stats():
    data = collection.get()
    return len(data["ids"])

def reset_database():
    global collection
    client.delete_collection("docs")
    collection = client.get_or_create_collection("docs")

def add_documents(uploaded_files, file_hash_map):
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_hash = file_hash_map[file_name]

        # 1. estrazione + cleaning globale
        full_text = load_pdf_paginated(uploaded_file)

        # 2. chunking centralizzato
        for page_num, text in full_text:
            chunks = chunk_text(text, chunk_size=1200, overlap=200)

            # 3. embeddings
            embeddings = embed_model.encode(chunks)

            for i, chunk in enumerate(chunks):
                # chunk_id = str(uuid.uuid4())
                chunk_id = hashlib.md5(
                    (file_hash + str(page_num) + chunk).encode()
                ).hexdigest()
                all_chunks.append(chunk)
                all_embeddings.append(embeddings[i])
                all_ids.append(chunk_id)

                all_metadata.append({
                    "id": chunk_id,
                    "file": file_name,
                    "file_hash": file_hash,
                    "page": page_num,
                    "chunk": chunk,
                    "uploaded_at": datetime.now().isoformat()
                })

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

def merge_chunks_by_file(documents, metadatas):
    """
    Unisce i chunk dello stesso file,
    rimuove duplicati e ordina per pagina.
    """

    files = {}

    for doc, meta in zip(documents, metadatas):
        file = meta["file"]
        page = meta["page"]

        if file not in files:
            files[file] = {}

        # uso dict per evitare chunk duplicati
        files[file][(page, doc)] = doc

    merged_docs = []

    for file, chunks_dict in files.items():
        # (page, doc)
        chunks = list(chunks_dict.keys())

        # ordina per pagina
        chunks_sorted = sorted(chunks, key=lambda x: x[0])

        full_text = "\n\n".join([c[1] for c in chunks_sorted])

        merged_docs.append({
            "file": file,
            "text": full_text
        })

    return merged_docs

def search_context(query, selected_doc=None, k_chunks=20):
    """
    Retrieval document-centric:
    - cerca chunk semanticamente simili
    - raggruppa per file
    - ricostruisce i documenti unendo i chunk
    - restituisce CONTEXT numerato stile Perplexity
    """

    # 1️⃣ Embedding query
    query_embedding = embed_model.encode([query])[0]

    # 2️⃣ Query Chroma
    if selected_doc:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_chunks,
            where={"file": selected_doc}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_chunks
        )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 🔴 Se DB vuoto
    if not documents:
        return "", []

    # 3️⃣ GROUP + SORT + MERGE CHUNKS → DOCUMENTI
    merged_docs = merge_chunks_by_file(documents, metadatas)

    # 4️⃣ COSTRUZIONE CONTEXT + SOURCES
    context_blocks = []
    structured_sources = []

    print("merged_docs")
    print(merged_docs)
    for i, doc in enumerate(merged_docs):
        doc_index = i + 1
        file_name = doc["file"]
        full_text = doc["text"]

        # snippet per preview fonti
        snippet = full_text[:300].replace("\n", " ")

        structured_sources.append({
            "index": doc_index,
            "file": file_name,
            "page": "multi",
            "text": snippet
        })

        context_blocks.append(
            f"[{doc_index}] DOCUMENTO: {file_name}\n{full_text}"
        )

    context = "\n\n".join(context_blocks)

    return context, structured_sources


def ask_llm(query, context):
    """
    Funzione per chiamare llm one shot e avere una risposta caricata tutta insieme, se vuoi una risposta caricata parola per parola utilizza la funzione `stream_llm_answer`
    :param query:
    :param context:
    :return:
    """
    prompt = f"""
Usa il contesto per rispondere.

Contesto:
{context}

Domanda: {query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def stream_llm_answer(query, context, sources):
    """
    Funzione per caricare la risposta parola per parola e avere una fluidità di risposta sull'interfaccia
    :param query:
    :param context:
    :return:
    """

    # sources_text = "\n".join(structured_sources)
    sources_text = "\n".join(
        [f"[{s['index']}] {s['file']} - pag. {s['page']}" for s in sources]
    )

    prompt = f"""
Usa il contesto per rispondere.

Regole:
- Non inventare informazioni
- Usa solo il contesto
- Non parlare dello strumento.
- Non fare meta-commenti.
- cita le fonti con [1], [2], ecc.
- Alla fine mostra sempre le fonti usate con nome del file e pagina in cui hai trovato le informazioni fornite con [1], [2], ecc. e devono corrispondere ai numeri nel contesto

Contesto:
{context}

Domanda: {query}

Fonti:
{sources_text}
"""

    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        yield chunk["message"]["content"]


def direct_llm_answer(query):
    """
    Risposta diretta senza usare tools o RAG.
    Serve per small talk o domande generiche.
    """

    prompt = f"""
Sei un assistente AI utile e intelligente.
Rispondi normalmente alla domanda dell'utente simpaticamente.

Domanda: {query}
"""
    print("SONO QUI, la tua richiesta è: ", query)
    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    print(len(stream))
    for chunk in stream:
        print(chunk)
        yield chunk["message"]["content"]