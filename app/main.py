from fastapi import FastAPI

from app.exemplo_04.rag_example import perguntar


app = FastAPI()

@app.get("/ask")
def ask(pergunta: str, custom_prompt: str):
    """Pergunte ao ChatGPT"""

    return perguntar(pergunta, custom_prompt)
