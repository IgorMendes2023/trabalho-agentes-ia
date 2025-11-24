# Trabalho Agentes IA - Tópicos Especiais
## Projeto de Agentes IA com LangChain + OpenAI

Este projeto implementa um agente inteligente utilizando **LangChain** e **OpenAI**, executando um fluxo ReAct para tomada de decisões, ferramentas e respostas avançadas.

## Tecnologias Utilizadas
- Python 3.12+
- LangChain
- LangChain OpenAI
- OpenAI API
- Dotenv

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/seu-repo/trabalho-agentes-ia.git
cd trabalho-agentes-ia
```
Crie e ative o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
Instale as dependências:
```bash
pip install -r requirements.txt
```

Rode o projeto: 
```bash
python agent.py
```

## Configurando a Groq API (alternativa gratuita à OpenAI)
Este projeto também oferece suporte à Groq API, uma alternativa extremamente rápida e gratuita que permite rodar seu agente sem depender da OpenAI.

### Criando sua Groq API Key
Acesse o painel da Groq:
https://console.groq.com/keys
Clique em “Create API Key”
Copie a chave gerada (exemplo: gsk_xxxxxx)

### Configurando a variável de ambiente
Linux / Mac / Codespaces
```bash
export GROQ_API_KEY="sua_chave_aqui"