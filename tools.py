import json

import ollama
from rag_engine import get_files_with_upload_date, get_files_in_db, direct_llm_answer, conversational_search, build_chat_history
import streamlit as st
from utils.general import extract_between

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
        yield from direct_llm_answer(query, messages)
        return

    # 3️⃣ esegui tool
    result = execute_tool(plan["tool"], query, selected_doc)

    # Recupero la history dei messaggi per passarla alla conversazione in modo tale che abbia un ricordo di quanto detto fin'ora
    chat_history = build_chat_history(messages) if messages else ""

    # 4️⃣ LLM finale usa risultato tool
    final_prompt = f"""
Usa questi dati per rispondere all'utente.

Regole:
- Alla fine mostra sempre le fonti usate con nome del file e pagina in cui hai trovato le informazioni fornite
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


def react_agent(query, messages, selected_doc=None):
    REACT_PROMPT = """
    Sei un agente AI che può usare strumenti.

    Devi seguire questo formato:

    Thought: pensa cosa fare
    Action: nome_tool oppure None
    Action Input: input per il tool
    Observation: risultato del tool
    ... (può ripetersi)
    Final Answer: risposta finale all'utente

    TOOLS DISPONIBILI:
    - search_documents(query)
    - list_documents()
    - get_upload_dates()

    REGOLE:
    - NON inventare dati
    - Usa tools quando servono
    - Se hai abbastanza informazioni, rispondi con Final Answer

    DOMANDA:
    {query}

    CONVERSAZIONE:
    {history}
    """

    chat_history = build_chat_history(messages)

    prompt = REACT_PROMPT.format(
        query=query,
        history=chat_history
    )

    max_steps = 5

    for step in range(max_steps):

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response["message"]["content"]

        print("🧠 REACT STEP:", output)

        # 1️⃣ Se finale → stop
        if "Final Answer:" in output:
            final = output.split("Final Answer:")[-1]
            yield final.strip()
            return

        # 2️⃣ Estrai Action
        if "Action:" in output:
            action = extract_between(output, "Action:", "\n").strip()
            action_input = extract_between(output, "Action Input:", "\n")

            # 3️⃣ esegui tool
            observation = execute_tool(action, action_input, selected_doc)

            # 4️⃣ aggiorna prompt con observation
            prompt += f"\n{output}\nObservation: {observation}\n"


def execute_tool(tool_name, query=None, selected_doc=None):
    print("tool_name", tool_name)
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
