import faiss
import openai
import numpy as np
import json
import os
import fitz  # PyMuPDF

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 1. Extrai o texto do PDF ---
def extrair_textos_pdf(caminho_pdf):
    doc = fitz.open(caminho_pdf)
    textos = []
    for pagina in doc:
        texto = pagina.get_text().strip()
        if texto:
            textos.append(texto)
    return textos

# --- 2. Divide em blocos menores (evita estourar limite do embedding) ---
def dividir_blocos(textos, max_chars=500):
    blocos = []
    for texto in textos:
        paragrafos = texto.split('\n')
        bloco = ""
        for paragrafo in paragrafos:
            if len(bloco) + len(paragrafo) < max_chars:
                bloco += paragrafo.strip() + " "
            else:
                blocos.append(bloco.strip())
                bloco = paragrafo.strip() + " "
        if bloco:
            blocos.append(bloco.strip())
    return blocos

# --- 3. Embedding dos blocos e criação do índice FAISS ---
pdf_path = "base-de-conhecimento-vitor-peyroton.pdf"
textos_pdf = extrair_textos_pdf(pdf_path)
blocos = dividir_blocos(textos_pdf)

vetores = []
for bloco in blocos:
    resposta = openai.Embedding.create(
        input=bloco,
        model="text-embedding-3-small"
    )
    vetor = np.array(resposta['data'][0]['embedding'], dtype='float32')
    vetores.append(vetor)

index = faiss.IndexFlatL2(len(vetores[0]))
index.add(np.array(vetores))

faiss.write_index(index, "index.faiss")

# Salva os textos em JSON
with open("dados.json", "w") as f:
    json.dump(blocos, f)
