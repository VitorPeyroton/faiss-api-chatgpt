import faiss
import openai
import numpy as np
import json

import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Textos que simulam o conteúdo extraído do seu PDF
textos = [
    "OmegaCalm é um suplemento com ômega-3, indicado para suporte ao humor e saúde mental.",
    "SonoZen combina triptofano e melatonina para auxiliar no sono profundo e relaxamento.",
    "NeuroPlus contém GABA e ashwagandha, ideal para reduzir o estresse e aumentar o foco.",
    "VitaMulher é um multivitamínico desenvolvido para mulheres, com ferro e ácido fólico.",
    "FocusFast é indicado para concentração e memória, com cafeína, taurina e colina."
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
