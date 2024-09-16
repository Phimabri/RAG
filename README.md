# RAG

This project provides a pipeline to process large texts (like Nietzsche's *Beyond Good and Evil*) by chunking, embedding, and creating a vector database for semantic search. It also integrates a language model to answer questions in French based on the processed content.

## Key Features

1. **Text Chunking**: 
   - The system extracts text from PDF files and splits it into smaller chunks (512 characters) using the `SentenceSplitter`.
   - The chunks are cleaned to preserve special characters, accents, and apostrophes.
   - These chunks are then saved into a CSV file for further use.

2. **Text Embedding**: 
   - The text chunks are converted into numerical representations (embeddings) using the `CamemBERT` model from `sentence-transformers`.
   - The embeddings are saved into CSV files for easy access and reuse.

3. **Vector Database Creation**: 
   - The system builds a vector database using the `Faiss` library for efficient semantic search.
   - The embeddings are indexed in the Faiss index, which is stored for future queries.

4. **Question Answering with Context Retrieval**:
   - First of all, it is required to install Ollama (Meta's platform to run LLMs locally).
   - The system can retrieve the most relevant text chunks based on a user-provided question.
   - Using Faiss, it searches for the top-k chunks related to the question, encodes the query, and retrieves relevant context from the text chunks.
   - The `Phi3` language model is then used to generate a response based on this context.
   - It is also possible to use google Colab thanks to the file model_RAG.ipynb if you don't want to install Ollama 


