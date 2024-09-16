
""" ******************************** Partie 1 : Transformation du texte *********************************"""
# Importation des bibliothèques nécessaires
from llama_index.node_parser import SentenceSplitter  # Pour découper le texte en chunks
import fitz  # Bibliothèque pour travailler avec des fichiers PDF
import pandas as pd
import unicodedata


# Fonction pour nettoyer le texte en conservant les caractères spéciaux, les accents et les apostrophes
def clean_text(text):
    # Utilisation de la normalisation Unicode pour conserver les caractères spéciaux et les accents
    cleaned_text = unicodedata.normalize('NFKD', text)
    # Remplacer les caractères non ASCII par une chaîne vide, sauf les apostrophes
    cleaned_text = ''.join([char for char in cleaned_text if (not unicodedata.combining(char) and ord(char) < 128) or char in ("'", "’")])
    return cleaned_text.strip()  # Supprimer les espaces en début et fin de texte


# Chemin vers le fichier PDF
file_path="/Users/maloevain/Desktop/Perso/Clever_library/Documents/Par_dela_le_bien_et_le_mal.pdf"
doc = fitz.open(file_path)

# Initialisation de SentenceSplitter avec une taille de chunk de 512 caractères
text_parser = SentenceSplitter(chunk_size=512)

# Initialisation d'un dictionnaire pour stocker les informations sur les chunks de texte
chunks_info = {}
text_chunks=[]
compteur = 0

# Boucle sur chaque page du document
for doc_idx, page in enumerate(doc):
    # Extraction du texte de la page courante
    page_text = page.get_text("text")

    # Utilisation de SentenceSplitter pour découper le texte de la page en chunks
    cur_text_chunks = text_parser.split_text(page_text)

    # Nettoyage de chaque chunk de texte
    cur_text_chunks = [clean_text(chunk) for chunk in cur_text_chunks]

    # Ajout des chunks de texte à la liste text_chunks
    text_chunks.extend(cur_text_chunks)

    # Mise à jour du dictionnaire avec les informations sur les chunks
    for i, chunk_text in enumerate(cur_text_chunks):
        chunks_info[compteur + i] = [doc_idx, chunk_text]

    # Mise à jour du compteur
    compteur += len(cur_text_chunks)

# Création d'un DataFrame à partir du dictionnaire de chunks
df = pd.DataFrame.from_dict(chunks_info, orient='index', columns=['Page', 'Texte'])

# Sauvegarde du DataFrame au format CSV
df.to_csv("../Documents/Nieztsch_chunks.csv")






# À la fin de ce script, nous avons deux listes :
# 1. text_chunks contenant tous les segments de texte du document PDF
# 2. doc_idxs contenant les indices des pages correspondant à chaque chunk dans text_chunks
