import sys
import time
#On se place dans le répertoire parent
sys.path.append('./utils')

import ollama
from retriever import find_closest_chunks


import tensorflow as tf
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')

print("quelle est votre question ?")
question = input()

start_time = time.time() # Début du compteur de temps

#On récupère le context en rapport avec la question
context = find_closest_chunks(question ,"/faiss_index/Nieztsch_faiss_index","/embeddings/Nieztsch_embeddings.csv","/embeddings/Nieztsch_text_and_embeddings.csv")
context= "".join(context)

response = ollama.chat(model='phi3', messages=[{
    'role': 'user',
    'content': "Imagine que tu es un professeur de philosophie. En utilisant le contexte suivant : {} , réponds à la question suivante en français : {}\n ".format(context,question)
}])

end_time = time.time() # Fin du compteur de temps
execution_time = end_time - start_time # Calcul du temps d'exécution

print("Temps d'exécution: ", execution_time, "secondes")

print("le contexte sur lequel se base le LLM pour répondre à votre question est :", context)
print()
print()
print()
print()
print("votre réponse est :",response['message']["content"])
