from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
import tiktoken


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


# -------------------------------
# TOOL 1 – Busca de notícias
# -------------------------------
search = DuckDuckGoSearchRun()

def buscar_noticias(state: State):
    query = state["input"]

    state["tokens_prompt"] += contar_tokens(query)

    result = search.run(query)

    state["tokens_completion"] += contar_tokens(result)

    state["news"] = result
    state["steps"].append("Buscou notícias")
    return state


# -------------------------------
# TOOL 2 – Analisador de Sentimento
# -------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def avaliar_sentimento(state: State):
    texto = state["news"]

    prompt = f"Classifique o sentimento como POSITIVE ou NEGATIVE:\n\n{texto}"
    state["tokens_prompt"] += contar_tokens(prompt)

    resposta = llm.invoke(prompt).content
    state["tokens_completion"] += contar_tokens(resposta)

    state["sentiment"] = resposta.strip()
    state["steps"].append("Classificou sentimento")
    return state


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
        "tokens_total": 0
    }

    result = app.invoke(state)

    result["tokens_total"] = result["tokens_prompt"] + result["tokens_completion"]

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

    return result


if __name__ == "__main__":
    executar_agente("Notícias sobre inflação no Brasil")
