


""" ********************************* Partie 2 Bis : Embedding ************************************ """

import pandas as pd



from sentence_transformers import SentenceTransformer
model =  SentenceTransformer("dangvantuan/sentence-camembert-large")



chunks=pd.read_csv('../documents/Nieztsch_chunks.csv')
text_chunks=chunks['Texte']

embeddings = model.encode(text_chunks)

df=pd.DataFrame(embeddings)
df.insert(loc=0, column='text',value=text_chunks)

df.to_csv('../embeddings/Nieztsch_text_and_embeddings.csv')

df=pd.DataFrame(embeddings)
df.to_csv('../embeddings/Nieztsch_embeddings.csv')
