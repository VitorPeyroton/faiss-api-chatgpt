from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import openai
import numpy as np
import json

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

index = faiss.read_index("index.faiss")
with open("dados.json", "r") as f:
    textos = json.load(f)

app = Flask(__name__)
CORS(app, supports_credentials=True)

index = faiss.read_index("index.faiss")
with open("dados.json", "r") as f:
    textos = json.load(f)

@app.route("/buscar-contexto", methods=["POST", "OPTIONS"])
def buscar_contexto():
    data = request.get_json()
    pergunta = data["pergunta"]

    resposta = openai.Embedding.create(
        input=pergunta,
        model="text-embedding-ada-002"
    )
    vetor = np.array(resposta["data"][0]["embedding"]).astype("float32")

    D, I = index.search(np.array([vetor]), k=2)
    resultados = [textos[i] for i in I[0]]

    return jsonify({"trechos": resultados})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
