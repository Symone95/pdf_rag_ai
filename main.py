from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# 1️⃣ Leggere PDF
reader = PdfReader("documento.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# 2️⃣ Dividere in chunk
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3️⃣ Creare embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)

# 4️⃣ Salvare in vector DB
client = chromadb.Client()
collection = client.create_collection("docs")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i]],
        ids=[str(i)]
    )

print("Documento indicizzato!")

# 5️⃣ Chat loop
while True:
    query = input("\nDomanda: ")

    query_embedding = embed_model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    context = "\n".join(results["documents"][0])

    prompt = f"""
Usa il contesto per rispondere alla domanda.

Contesto:
{context}

Domanda: {query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nRisposta:", response["message"]["content"])