import json

import ollama
from rag_engine import get_files_with_upload_date, get_files_in_db, direct_llm_answer, conversational_search, build_chat_history
import streamlit as st

TOOLS = [
    {
        "name": "search_documents",
        "description": "Cerca informazioni nei documenti caricati",
        "input": "query"
    },
    {
        "name": "list_documents",
        "description": "Restituisce la lista dei file presenti nel database",
        "input": "none"
    },
    {
        "name": "get_upload_dates",
        "description": "Restituisce quando sono stati caricati i documenti",
        "input": "none"
    }
]

def agent_answer(query, selected_doc = None, messages = None):

    print("AGENT ANSWER")
    # 1️⃣ planner decide tool
    plan_raw = tool_planner(query)

    try:
        plan = json.loads(plan_raw)
    except:
        plan = {"tool": "none"}

    print("AGENT PLAN", plan)

    # 2️⃣ se nessun tool → risposta diretta
    if plan["tool"] == "none":
        yield from direct_llm_answer(query)
        return

    # 3️⃣ esegui tool
    result = execute_tool(plan["tool"], query, selected_doc)

    # Recupero la history dei messaggi per passarla alla conversazione in modo tale che abbia un ricordo di quanto detto fin'ora
    chat_history = build_chat_history(messages) if messages else ""

    # 4️⃣ LLM finale usa risultato tool
    final_prompt = f"""
Usa questi dati per rispondere all'utente.

Regole:
- Rispondi in modo chiaro e diretto.
- Non inventare dati.
- Non parlare dello strumento.
- Non fare meta-commenti.

Conversazione:
{chat_history}

Domanda dell'utente: {query}

Lo strumento ha restituito questi dati:
{result}
"""

    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": final_prompt}],
        stream=True
    )

    for chunk in stream:
        yield chunk["message"]["content"]

    return None

def execute_tool(tool_name, query=None, selected_doc=None):
    print("tool_name")
    print(tool_name)
    if tool_name == "search_documents":
        context, structured_sources = conversational_search(query, st.session_state.messages, selected_doc)
        return {
            "context": context,
            "sources": structured_sources
        }

    if tool_name == "list_documents":
        files = get_files_in_db()
        return {"files": files}

    if tool_name == "get_upload_dates":
        dates = get_files_with_upload_date()
        return {"dates": dates}

    return {"error": "Tool non trovato"}


def tool_planner(query):
    tools_description = "\n".join(
        [f"{t['name']}: {t['description']}" for t in TOOLS]
    )

    prompt = f"""
Sei un AI che decide quale tool usare.

Tools disponibili:
{tools_description}

Regole:
- Rispondi SOLO in JSON
- Se serve un tool → {{ "tool": "nome_tool", "query": "..." }}
- Se NON serve tool → {{ "tool": "none" }}

Domanda: {query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
