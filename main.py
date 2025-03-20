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
import json  # Add this import
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv


load_dotenv()

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

def retrieve_context_multi(queries: List[str], collection, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve relevant documents for multiple queries and combine results."""
    all_documents = []
    seen_docs = set()  # To track unique documents
    
    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if results and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] and i < len(results["metadatas"][0]) else {}
                distance = results["distances"][0][i] if results["distances"] and i < len(results["distances"][0]) else 0.0
                
                # Create a unique identifier for the document
                doc_id = str(doc) + str(metadata.get('source', ''))
                
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    all_documents.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance
                    })
    
    # Sort by relevance (distance)
    all_documents.sort(key=lambda x: x["distance"])
    
    # Limit to top_k most relevant documents
    return all_documents[:top_k]

def generate_response(query: str, context_docs: List[Dict[str, Any]], 
                      conversation_history: List[Dict[str, str]] = None,
                      mentioned_concepts: Dict = None) -> str:
    """Generate a response using OpenAI's model with context, conversation history, and concept info."""
    # Format the context
    context = ""
    sources = []
    
    for i, doc in enumerate(context_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
        if 'source' in doc['metadata']:
            sources.append(f"[{i+1}] {doc['metadata']['source']}")
    
    # Format concept information if available
    concept_info = ""
    if mentioned_concepts and len(mentioned_concepts) > 0:
        concept_info = "Relevant concept information:\n"
        for concept, details in mentioned_concepts.items():
            concept_info += f"- {concept}: {json.dumps(details)}\n"
    
    # Create the prompt
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context, conversation history, "
        "and any relevant concept information. "
        "If related documents were found in the knowledge base, use them as your primary source and cite them using [Doc X] notation. "
        "If no relevant documents were found but concept information is available, use that to provide a helpful response "
        "and clearly state that the answer is based on concept information, not document search results. "
        "If neither documents nor concept information is available, respond based on your general knowledge "
        "and clearly state that no specific information was found in the knowledge base."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if available
    if conversation_history and len(conversation_history) > 0:
        messages.extend(conversation_history)
    
    # Add the current query with context and concept info
    has_vdb_results = len(context_docs) > 0
    
    user_prompt = f"Question: {query}\n\n"
    
    if has_vdb_results:
        user_prompt += f"Context from knowledge base:\n{context}\n\n"
    else:
        user_prompt += "Note: No relevant documents were found in the knowledge base.\n\n"
    
    if concept_info:
        user_prompt += f"{concept_info}\n"
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Generate the response
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
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

def augment_query(query: str, num_questions: int = 3) -> tuple[List[str], dict]:
    """
    Generate multiple questions based on the original query.
    Also enriches the query with relevant information from the FIB JSON file
    if concepts are mentioned.
    
    Returns:
        tuple: (list of questions, dictionary of mentioned concepts)
    """
    # First check if any concepts from the JSON file are mentioned in the query
    mentioned_concepts = {}
    try:
        # Load the JSON file with concepts
        import json
        with open("dictionary.json", "r") as f:
            concepts_data = json.load(f)
        
        # Extract mentioned concepts
        for concept, details in concepts_data.items():
            if concept.lower() in query.lower():
                mentioned_concepts[concept] = details
            # Also check if the full name is mentioned
            elif 'name' in details and details['name'].lower() in query.lower():
                mentioned_concepts[concept] = details
        
        # Create concept enrichment text
        concept_info = ""
        if mentioned_concepts:
            concept_info = "Additional context for mentioned concepts:\n"
            for concept, details in mentioned_concepts.items():
                concept_info += f"- {concept}: {json.dumps(details)}\n"
        
        # Generate alternative questions with concept enrichment
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": 
                 "Generate alternative versions of the user's question to improve information retrieval. "
                 "Create varied questions that explore different aspects and phrasings of the original query. "
                 "If concept information is provided, use it to create more specific and relevant questions. "
                 "Return only the questions as a numbered list, without explanations or other text."},
                {"role": "user", "content": 
                 f"Original question: {query}\n\n"
                 f"{concept_info}\n\n"
                 f"Generate {num_questions} alternative questions:"}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        response_text = response.choices[0].message.content
        
        # Parse the numbered list of questions
        questions = []
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line:
                # Remove numbering like "1.", "2)", etc.
                cleaned_line = ' '.join(line.split(' ')[1:]) if any(line.startswith(f"{i}{sep}") for i in range(1, 10) for sep in ['.', ')', ':', '-']) else line
                questions.append(cleaned_line)
        
        # Add the original query to the list
        questions.insert(0, query)
        
        # Remove duplicates while preserving order
        unique_questions = []
        for q in questions:
            if q not in unique_questions:
                unique_questions.append(q)
        
        return unique_questions, mentioned_concepts
        
    except Exception as e:
        print(f"Error augmenting query: {str(e)}")
        # Return just the original query if there's an error
        return [query], mentioned_concepts

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
    conversation_history = []  # Store the conversation history
    
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
            
            # Add user message to history (without context)
            conversation_history.append({"role": "user", "content": user_input})
            
            # RAG process
            print("\nProcessing your question...")
            start_time = time.time()
            
            # Augment the query and get concept information
            print("Generating related questions...")
            augmented_queries, mentioned_concepts = augment_query(user_input)
            
            print(f"Searching for relevant information using {len(augmented_queries)} queries...")
            
            # Retrieve context using multiple queries
            context_docs = retrieve_context_multi(augmented_queries, collection)
            
            # Save sources
            last_sources = [doc["metadata"].get("source", "Unknown") for doc in context_docs if "source" in doc["metadata"]]
            
            # Generate response
            print("Generating answer...")
            
            # Extract the last 5 exchanges (10 messages or fewer) for the history context
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history[:]
            
            # Generate response with history context, VDB results, and concept information
            response = generate_response(
                user_input, 
                context_docs, 
                recent_history[:-1],
                mentioned_concepts
            )
            
            # Extract answer without sources for conversation history
            answer_only = response
            if "\n\nSources:" in response:
                answer_only = response.split("\n\nSources:")[0]
                
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": answer_only})
            
            # Keep only the last 5 exchanges (10 messages) in history
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
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