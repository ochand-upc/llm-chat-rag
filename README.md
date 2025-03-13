# LLM Chat RAG - CLI Interface

## Overview

The **LLM Chat RAG** (Retriever-Augmented Generation) is a Command Line Interface (CLI) tool that allows you to interact with documents stored in a ChromaDB database using OpenAI's GPT-4o-mini model. This system provides a way to query documents for relevant information, retrieve context, and generate AI-driven responses based on that context.

## Features

- **Retriever-Augmented Generation (RAG):** Retrieve context from a database and use it to generate more accurate and informative responses.
- **OpenAI GPT-4o-mini Model:** Uses OpenAI's GPT-4o-mini to generate answers based on the retrieved context.
- **ChromaDB Integration:** Uses ChromaDB for efficient document retrieval and context management.
- **Command-line interface:** Interact with the system through an intuitive CLI, allowing commands such as `/help`, `/exit`, and `/sources`.

## Prerequisites

- **Python 3.x**: Ensure Python 3.6 or later is installed.
- **OpenAI API Key**: You must have an OpenAI API key to use the GPT model.
- **ChromaDB**: A local or cloud-based ChromaDB instance for document storage and retrieval.

## Setup

### 1. Install Dependencies

Install required dependencies via `pip`:

```bash
pip install -r requirements.txt
```
### 2. Set Environment Variables

Set the following environment variables:
``` bash
OPENAI_API_KEY: Your OpenAI API key.
CHROMA_DB_PATH: (Optional) Path to the ChromaDB persistent database (default is ./chroma_db).
```
For example, on Linux or macOS, you can set the environment variables like this:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export CHROMA_DB_PATH="./chroma_db"
```

On Windows, use:

```bash
set OPENAI_API_KEY=your_openai_api_key
set CHROMA_DB_PATH=.\chroma_db
```

### 3. Run the Application

To start the CLI chatbot, run the following command:

```bash
python3 main.py
```

You will be prompted to interact with the chatbot, ask questions, and receive answers based on documents stored in ChromaDB.

### 4. Setup ChromaDB (Optional)

If you need to set up the ChromaDB collection before starting the application, run:

```bash
python3 main.py --setup
```

This will initialize the ChromaDB collection with OpenAI embeddings.
#### CLI Commands

    /help: Show the help message with available commands.
    /exit: Exit the application.
    /sources: Show the sources of the last response generated.

#### How it Works

- ChromaDB Setup: On running the script, it will either set up a new ChromaDB collection or connect to an existing one containing document embeddings.
- Retriever-Augmented Generation (RAG): When a user enters a question, the system will retrieve relevant documents from ChromaDB, and use the retrieved context along with OpenAI's GPT model to generate an answer.
- Response Generation: The generated response is displayed, and the sources (if any) are included at the end of the response.

### 5. Example Usage

```bash
==============================================
        LLM Chat RAG - CLI Interface
==============================================
Type your questions and get answers from documents.
Commands:
  /help    - Show this help message
  /exit    - Exit the application
  /sources - Show sources for the last response
==============================================

> What is the capital of France?
Searching for relevant information...
Generating answer...

Response (generated in 1.23s):
==================================================
The capital of France is Paris.
==================================================
```

#### Development

To contribute to this project, feel free to fork the repository, make changes, and create pull requests.
License

This project is licensed under the  GNU General Public License - see the LICENSE file for details.


### Key Sections:
- **Overview**: Describes the functionality of the tool.
- **Prerequisites**: Lists the required software and environment variables.
- **Setup**: Guides the user through installing dependencies, setting environment variables, and running the application.
- **CLI Commands**: Provides a list of available commands for interaction.
- **How it Works**: Explains the core functionality of the app.
- **Example Usage**: Shows an example of how to interact with the CLI.