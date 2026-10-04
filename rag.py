import os 
from pathlib import Path
import faiss 
import numpy as np
import cohere

from sentence_transformers import SentenceTransformer
# from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

co = cohere.ClientV2(
    api_key = os.getenv("COHERE_API_KEY")
)

# Load the Document 

def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # print(f"Loading documents from {file_path}")
        documents = file.readlines()
        # print(documents)
    return [doc.strip() for doc in documents if doc.strip()]


# chunking the Document 
def chunk_documents(text , check_size=200):
    "simple function to chunk the documents into smaller pieces"

    
    chunks = []
    chunk_size = 0
    current_chunk = []
    
    for word in text:
        current_chunk.append(word) 
        chunk_size += len(word) + 1  # +1 for the space
        
        if chunk_size >= check_size:
            chunks.append(' '.join(current_chunk))
            
            current_chunk = []
            chunk_size = 0
        
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return  chunks
    
    
# Create Embeddings for the Chunks

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks , convert_to_numpy=True)

    return embeddings


# FAISS Search Index 

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


# retrive chunks from the index based on the query

def retrieve_chunks(query , index , chunks , top_k=3):
    """Convert query to embedding 
    search the index for the most similar chunks FAISS
    retrieve top_k chunks from the index
    """
    query_embedding  = embedding_model.encode([query] , convert_to_numpy=True)
    
    query_embedding = query_embedding.astype("float32")
    
    distances , indicies = index.search(
        query_embedding , 
        top_k 
    )
    
    retrived_chunks = []
    
    for i in indicies[0]:
        
        if i < len(chunks):
            retrived_chunks.append(chunks[i])

    return retrived_chunks


## Bulid a Prompt 

def create_prompt(query , retrived_chunks):
    
    context = " ".join(retrived_chunks)
    
    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the information
    provided in the context below.

    If the answer is not present in the context,
    say "I don't know based on the provided documents.
    
    Context:
    
    {context}
    
    
    Question:
    
    {query}
    
    Answer:
    """
    
    return prompt


# Generate answer from LLM 

def generate_answer(prompt):

    response = co.chat(
        model="command-a-03-2025",
        
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return response.message.content[0].text
def main():
    
    file_path = Path("./documents/growman.txt")
    print("\n==== Loading Documents ====")
    text  = load_documents(file_path)
    print(f"Loaded {len(text)} documents from {file_path}")




    print("\n==== Chunking Documents ====")
    chunks = chunk_documents(text)
    print(f"Created {len(chunks)} chunks from the documents")
    #cretaing embeddings for the chunks
    # for i , chunks in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunks}")
    
    
        
    print("\n==== Creating Embeddings ====")
    embeddings = create_embeddings(chunks)
    print(f"Created embeddings for {len(embeddings)} chunks")
    print(f"Embedding shape: {embeddings.shape}")


    # Create FAISS index
    print("\n==== Creating FAISS index ====")
    index = create_faiss_index(embeddings)
    print(f"Created FAISS index with {index.ntotal} vectors")
    
    query = input("\nEnter your query: ")
    
    retrive_chunks = retrieve_chunks(query, index , chunks ,3)
    
    
    for i , chunk in enumerate(retrive_chunks):
        print(f"\n Retrived {i+1}")
        print(chunk)
        


    print("=============Build prmopt=========")
    
    prompt = create_prompt(query , retrive_chunks)
    
    print("\n============Final prompt========")
    
    print(prompt)
    
    
    
    print("\n ============Generate Answer=========")
    
    answer = generate_answer(prompt)
    
    print(answer)
    
    
    
if __name__ == "__main__":
    main()
    