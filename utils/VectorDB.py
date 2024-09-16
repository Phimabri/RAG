import faiss
import pandas as pd
import numpy as np
import pandas as pd
import faiss

def create_vector_database(embeddings_path, index_path):
    """
    entrée : chemin vers le fichier d'embeddings
                chemin où on va enregistrer la vector database

    sortie : rien (on enregistre juste la VdB)

    """
    # Charger les données
    embeddings = pd.read_csv(embeddings_path).drop('Unnamed: 0', axis=1)

    # Taille de l'embedding
    n, size = embeddings.shape
    nlist = 50  # Nombre de cellules pour la cellule de Voronoi
    index = faiss.IndexFlatL2(size)

    index.add(embeddings)

    # Sauvegarder l'index
    faiss.write_index(index, index_path)

if __name__ == "__main__":
    # Chemins vers les données et l'index Faiss
    embeddings_path = "../embeddings/Nieztsch_embeddings.csv"
    text_embeddings_path = "../embeddings/Nieztsch_text_and_embeddings.csv"
    index_path = "../faiss_index/Nieztsch_faiss_index"

    # Créer la base de données vectorielle
    create_vector_database(embeddings_path, index_path)
