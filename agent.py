from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
import tiktoken
import time


# -------------------------------
# Token Counter (compatível com LangChain 1.x)
# -------------------------------
def contar_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# -------------------------------
# Estado do agente
# -------------------------------
class State(TypedDict):
    input: str
    news: str
    sentiment: str
    steps: List[str]
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    metrics: dict


# -------------------------------
# TOOL 1 – Busca de notícias
# -------------------------------
search = DuckDuckGoSearchRun()

def buscar_noticias(state: State):
    t0 = time.perf_counter()
    query = state["input"]
    result = search.run(query)
    t1 = time.perf_counter()
    state["news"] = result
    state["steps"].append("Buscou notícias")
    state.setdefault("metrics", {})["buscar_noticias_ms"] = (t1 - t0) * 1000.0
    return state


# -------------------------------
# TOOL 2 – Analisador de Sentimento
# -------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def avaliar_sentimento(state: State):
    t0 = time.perf_counter()
    texto = state["news"]
    prompt = f"Classifique o sentimento como POSITIVE ou NEGATIVE:\n\n{texto}"
    res = None
    tentativa = 0
    ultimo_erro = None
    while tentativa < 3 and res is None:
        try:
            res = llm.invoke(prompt)
        except Exception as e:
            ultimo_erro = e
            time.sleep(0.5 * (tentativa + 1))
            tentativa += 1
    if res is None:
        state["sentiment"] = "ERROR"
    else:
        resposta = res.content
        monitorar_tokens_de_resposta(state, res)
        if not getattr(res, "usage_metadata", None):
            state["tokens_prompt"] += contar_tokens(prompt)
        state["sentiment"] = resposta.strip()
    t1 = time.perf_counter()
    state.setdefault("metrics", {})["avaliar_sentimento_ms"] = (t1 - t0) * 1000.0
    state["steps"].append("Classificou sentimento")
    return state


def monitorar_tokens_de_resposta(state: State, message):
    usage = getattr(message, "usage_metadata", None)
    if usage:
        state["tokens_prompt"] += int((usage.get("input_tokens") or 0))
        state["tokens_completion"] += int((usage.get("output_tokens") or 0))
        return
    meta = getattr(message, "response_metadata", None)
    if isinstance(meta, dict):
        usage2 = meta.get("token_usage") or meta.get("usage")
        if isinstance(usage2, dict):
            state["tokens_prompt"] += int((usage2.get("prompt_tokens") or 0))
            state["tokens_completion"] += int((usage2.get("completion_tokens") or 0))
            return
    content = getattr(message, "content", "")
    state["tokens_completion"] += contar_tokens(content)


# -------------------------------
# Fluxo do agente (LangGraph)
# -------------------------------
workflow = StateGraph(State)

workflow.add_node("buscar_noticias", buscar_noticias)
workflow.add_node("avaliar_sentimento", avaliar_sentimento)

workflow.set_entry_point("buscar_noticias")
workflow.add_edge("buscar_noticias", "avaliar_sentimento")
workflow.add_edge("avaliar_sentimento", END)

app = workflow.compile()


# -------------------------------
# Execução
# -------------------------------
def executar_agente(consulta: str):

    state = {
        "input": consulta,
        "news": "",
        "sentiment": "",
        "steps": [],
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "metrics": {}
    }

    t0 = time.perf_counter()
    result = app.invoke(state)
    t1 = time.perf_counter()

    result["tokens_total"] = result["tokens_prompt"] + result["tokens_completion"]
    result.setdefault("metrics", {})["total_ms"] = (t1 - t0) * 1000.0

    print("\n--- NOTÍCIA ENCONTRADA ---")
    print(result["news"])

    print("\n--- SENTIMENTO ---")
    print(result["sentiment"])

    print("\n--- PASSOS EXECUTADOS ---")
    print(result["steps"])

    print("\n--- USO DE TOKENS ---")
    print(f"Prompt tokens: {result['tokens_prompt']}")
    print(f"Completion tokens: {result['tokens_completion']}")
    print(f"Total tokens: {result['tokens_total']}")

    m = result.get("metrics", {})
    if m:
        print("\n--- MÉTRICAS ---")
        print(f"buscar_noticias_ms: {m.get('buscar_noticias_ms', 0):.2f}")
        print(f"avaliar_sentimento_ms: {m.get('avaliar_sentimento_ms', 0):.2f}")
        print(f"total_ms: {m.get('total_ms', 0):.2f}")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("consulta", nargs="?", default="Notícias sobre inflação no Brasil")
    args = p.parse_args()
    executar_agente(args.consulta)
