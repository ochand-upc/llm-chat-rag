#!/usr/bin/env python3

import os
import sys
import argparse
import chromadb
from chromadb.utils import embedding_functions
import openai
from openai import OpenAI
import textwrap
import time
from typing import List, Dict, Any, Optional

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
MODEL_NAME = "gpt-4o-mini"  # OpenAI 4o-mini model
MAX_TOKENS = 4096
TEMPERATURE = 0.7
TOP_K = 5  # Number of documents to retrieve

# Set up OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up ChromaDB
def setup_chroma():
    """Initialize and return ChromaDB client and collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Use OpenAI embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    
    # Get or create the collection
    try:
        collection = chroma_client.get_collection(name="markdown_docs", embedding_function=openai_ef)
        print(f"Connected to existing ChromaDB collection with {collection.count()} documents")
    except Exception:
        collection = chroma_client.create_collection(name="documents", embedding_function=openai_ef)
        print("Created new ChromaDB collection")
    
    return chroma_client, collection

# RAG Implementation
def retrieve_context(query: str, collection, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve relevant documents from ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = []
    if results and len(results["documents"]) > 0:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] and i < len(results["metadatas"][0]) else {}
            distance = results["distances"][0][i] if results["distances"] and i < len(results["distances"][0]) else 0.0
            
            documents.append({
                "content": doc,
                "metadata": metadata,
                "distance": distance
            })
    
    return documents

def generate_response(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate a response using OpenAI's 4o-mini model with context."""
    # Format the context
    context = ""
    sources = []
    
    for i, doc in enumerate(context_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
        if 'source' in doc['metadata']:
            sources.append(f"[{i+1}] {doc['metadata']['source']}")
    
    # Create the prompt
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don't know and try to help the user in another way. "
        "Include citations to the relevant documents in your answer using [Doc X] notation."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    # Generate the response
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        answer = response.choices[0].message.content
        
        # Add sources to the response if available
        if sources:
            answer += "\n\nSources:\n" + "\n".join(sources)
        
        return answer
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# CLI Interface
def print_welcome():
    """Print welcome message."""
    welcome = """
    ==============================================
            LLM Chat RAG - CLI Interface
    ==============================================
    Type your questions and get answers from documents.
    Commands:
      /help    - Show this help message
      /exit    - Exit the application
      /sources - Show sources for the last response
    ==============================================
    """
    print(welcome)

def format_output(text: str) -> str:
    """Format the output text."""
    wrapper = textwrap.TextWrapper(width=80, break_long_words=False, replace_whitespace=False)
    return "\n".join(["\n".join(wrapper.wrap(line)) for line in text.splitlines()])

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="LLM Chat RAG - A CLI chatbot with RAG capabilities")
    parser.add_argument("--setup", action="store_true", help="Setup ChromaDB and exit")
    args = parser.parse_args()
    
    # Check for API key
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Setup ChromaDB
    try:
        chroma_client, collection = setup_chroma()
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        sys.exit(1)
    
    # If just setting up, exit
    if args.setup:
        print("Setup complete. Exiting.")
        sys.exit(0)
    
    print_welcome()
    
    # Main chat loop
    last_sources = []
    while True:
        try:
            # Get user input
            user_input = input("\n> ")
            
            # Process commands
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Goodbye!")
                break
            elif user_input.lower() in ["/help", "help"]:
                print_welcome()
                continue
            elif user_input.lower() in ["/sources", "sources"]:
                if last_sources:
                    print("\nSources:")
                    for source in last_sources:
                        print(f"  - {source}")
                else:
                    print("No sources available for the last response.")
                continue
            elif not user_input.strip():
                continue
            
            # RAG process
            print("\nSearching for relevant information...")
            start_time = time.time()
            
            # Retrieve context
            context_docs = retrieve_context(user_input, collection)
            
            # Save sources
            last_sources = [doc["metadata"].get("source", "Unknown") for doc in context_docs if "source" in doc["metadata"]]
            
            # Generate response
            print("Generating answer...")
            response = generate_response(user_input, context_docs)
            
            # Print response
            elapsed_time = time.time() - start_time
            print(f"\nResponse (generated in {elapsed_time:.2f}s):")
            print("=" * 80)
            print(format_output(response))
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()