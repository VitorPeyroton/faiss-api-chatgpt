import faiss
import openai
import numpy as np
import json

import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Textos que simulam o conteúdo extraído do seu PDF
textos = [
    "RelaxHerbs é um suplemento natural com valeriana, passiflora e camomila. Ajuda na ansiedade e insônia leve.",
    "SleepWell contém melatonina e L-teanina. Ideal para quem sofre de insônia.",
    "EnergyBoost é um suplemento energético com cafeína, guaraná e ginseng. Melhora disposição física e mental."
]

vetores = []

for texto in textos:
    resposta = openai.Embedding.create(
        input=texto,
        model="text-embedding-3-small"
    )
    vetor = np.array(resposta['data'][0]['embedding'], dtype='float32')
    vetores.append(vetor)

index = faiss.IndexFlatL2(len(vetores[0]))
index.add(np.array(vetores))

faiss.write_index(index, "index.faiss")

# Salva os textos para depois retornar os correspondentes
with open("dados.json", "w") as f:
    json.dump(textos, f)
