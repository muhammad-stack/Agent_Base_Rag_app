# LLM-Powered Prompt Handler - ReadMe

This project creates a **Streamlit-based web application** that allows users to input a prompt, which is then processed by a **Large Language Model (LLM)** using **LangChain** and related tools such as FAISS for similarity search, web scraping, and more. The system integrates various components to search the web, retrieve documents, process input, and handle conversations with context, ultimately providing relevant responses to user prompts.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Tools and Components](#tools-and-components)
- [Customization](#customization)

## Overview

This application integrates **LLMs** with LangChain for a powerful prompt handler that can:
- Retrieve data from websites.
- Break down documents into chunks.
- Perform similarity-based searches using **FAISS**.
- Call external tools to perform additional functions like adding numbers.
- Manage conversational context and history.
  
It utilizes OpenAI-like models to process user prompts and return dynamic responses. The integration of multiple tools allows the agent to access real-time web data and leverage similarity search through embeddings.

## Features

- **Prompt-based User Interaction**: Users enter a prompt in a text box, and the app processes it using an LLM.
- **Document Retrieval**: Fetches documents from websites and processes them for further interaction.
- **Similarity Search with FAISS**: Uses FAISS for efficient similarity searches across retrieved documents.
- **Custom Tooling**: A tool to add two integers as an example of how to extend the agent with custom functions.
- **Agent with Function Calling**: Uses OpenAI-like functions with LangChain's `create_tool_calling_agent`.
- **Contextual Chat**: Tracks chat history to handle multi-turn conversations and provide context-aware responses.

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
```

### 2. Install Required Dependencies
Ensure you have Python installed. Then, install the required packages by running:
```bash
poetry install
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root with the necessary API keys and model configurations:
```bash
GEMINI_API_KEY=<your_gemini_key>
```

## How to Run

### Run the Streamlit App
Once the environment is set up and dependencies are installed, start the app by running:

```bash
streamlit run app.py
```

This will start the Streamlit app and open it in your default web browser.

## How It Works

1. **Prompt Input**: The user provides a prompt through a text box on the Streamlit UI.
2. **LLM Initialization**: The app uses the `ChatGoogleGenerativeAI` model to process the input prompt.
3. **Web Document Retrieval**: A web loader scrapes the content from the specified website (e.g., `daraz.com`), and FAISS is used to index the data for similarity search.
4. **Document Chunking**: The documents are split into smaller chunks for efficient processing using `RecursiveCharacterTextSplitter`.
5. **Tool Execution**: Custom tools (like a function to add two numbers) are registered, allowing the agent to call these functions as needed.
6. **Agent Setup**: The `create_tool_calling_agent` function creates the agent, which is capable of interacting with the tools and processing inputs from the LLM.
7. **Result Display**: The result is processed by the agent and displayed back to the user in the Streamlit interface.

## Tools and Components

### 1. **LangChain Components**:
   - **AgentExecutor**: Executes the agent with the tools and handles the flow of interactions.
   - **ChatGoogleGenerativeAI**: The language model used to handle user prompts.
   - **WebBaseLoader**: Retrieves and loads documents from specified URLs.
   - **FAISS Vector Store**: Performs efficient similarity searches on document embeddings.
   - **TavilySearchResults**: Searches for real-time results using an external search tool.
   - **ChatMessageHistory**: Tracks conversation history and context for multi-turn dialogues.
   
### 2. **Tools**:
   - **Retriever Tool**: Searches retrieved data using similarity matching.
   - **Custom Add Function**: Demonstrates how to integrate custom functions into the agent’s toolset.

### 3. **Streamlit for UI**:
   - Provides an interactive user interface to input prompts and display results.
   
## Customization

### Change the LLM Model
To use a different LLM model, modify the following line:
```python
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0
)
```
Replace `gemini-1.5-flash` with the desired model.

### Add More Tools
You can add additional tools by defining new functions and registering them in the `tools` list. For example:
```python
@tool
def multiply_numbers(a: int, b: int) -> int:
    return a * b
tools.append(multiply_numbers)
```

### Modify Search Settings
To change the FAISS vector store search behavior, adjust the settings in `retriever`:
```python
retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.8})
```

## Conclusion

This Streamlit app integrates LangChain's agent-based LLM processing with tools for web scraping, document retrieval, similarity search, and contextual chat history. It can easily be extended with additional tools, custom models, or more sophisticated document processing workflows.

