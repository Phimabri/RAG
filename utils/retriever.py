import torch
import numpy as np
from transformers import CamembertModel, AutoTokenizer
import csv
import faiss
import pandas as pd

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import os



def find_closest_chunks(question, faiss_index_path, embeddings_path, text_embeddings_path, model_name="dangvantuan/sentence-camembert-large", k=8):
    # Charger les données
    current_directory = "../"

    embeddings = pd.read_csv(current_directory+embeddings_path).drop('Unnamed: 0', axis=1)
    text_and_embeddings = pd.read_csv(current_directory+text_embeddings_path)

    # Charger l'index Faiss
    index = faiss.read_index(current_directory+faiss_index_path)

    # Initialiser le modèle Sentence Transformer
    model = SentenceTransformer(model_name)

    # Encoder la question
    question_embedding = model.encode(question).reshape(1, -1)

    # Recherche des chunks les plus proches
    D, I = index.search(question_embedding, k)

    # Collecter les chunks correspondants
    context = []
    for i in range(k):
        chunk_index = I[0][i]
        context.append(text_and_embeddings.iloc[chunk_index, 1])

    # Ajouter la question à la liste des chunks
    context.append(question)

    return context

if __name__ == "__main__":
    # Chemins vers les données et l'index Faiss
    faiss_index_path = "./camemBERT/faiss_index"
    embeddings_path = "./camemBERT/embeddings.csv"
    text_embeddings_path = "./camemBERT/text_and_embeddings.csv"

    print("Veuillez entrer votre question :")
    question = input()

    # Rechercher les chunks les plus proches
    closest_chunks = find_closest_chunks(question, faiss_index_path, embeddings_path, text_embeddings_path)

    # Afficher les chunks
    for chunk in closest_chunks:
        print(chunk)
