from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent, tool
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
import streamlit as st

load_dotenv() # Load the .env file

# Streamlit UI
st.title("LLM-Powered Prompt Handler")

# Input box for the prompt
prompt = st.text_area("Enter your prompt", height=200)

# Button to trigger LLM processing
if st.button("Submit"):
    if prompt:
        with st.spinner("Processing with LLM..."):
            # Initialize the ChatGoogleGenerativeAI model with specified parameters
            llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash",  # Specify the model to use
                temperature=0,  # Set the temperature for the model's output (0 means deterministic)
                max_retries=3  # Set the maximum number of retries for the model
            )

            # Initialize the chat message history to keep track of conversation context
            chat_history: ChatMessageHistory = ChatMessageHistory()

            # Initialize the Tavily search tool to search for results
            search: TavilySearchResults = TavilySearchResults()

            # Initialize the document loader to load documents from the website
            loader: WebBaseLoader = WebBaseLoader("https://www.daraz.com/")

            # Load the documents from the website into objects
            docs = loader.load()

            # Breaking the document into chunks
            document: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(documents=docs)

            # Initialize the FAISS vector store for efficient similarity search
            vector_store: FAISS = FAISS.from_documents(document, GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            ))

            retriever = vector_store.as_retriever(
                # search_type="similarity_score_threshold",
                # search_kwargs={'score_threshold': 0.8}
            )

            # Create a retriever tool using the vector store
            retriever_tool = create_retriever_tool(
                retriever=retriever, name="retriever", description="Retrieve the data from Daraz"
            )

            @tool
            def add_numbers(a: int, b: int) -> int:
                """ Add two integers and return the result """
                return a + b

            # List of tools the agent can use
            tools = [search, retriever_tool, add_numbers]

            # An open-source prompt to use for the agent for function calling purposes
            agent_prompt = hub.pull("hwchase17/openai-functions-agent")

            # Initialize the agent with the specified tools and prompt
            agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=agent_prompt)

            # Initialize the agent executor with the agent and tools and set verbose to True
            agent_exec: AgentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # Initialize the agent with the chat history
            agent_with_chat_history: RunnableWithMessageHistory = RunnableWithMessageHistory(
                agent_exec,
                lambda session_id: chat_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )

            # Run the agent with the chat history
            result = agent_with_chat_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "test009"}}
            )

            # Display the result in Streamlit
            st.write(result)
    else:
        st.warning("Please enter a prompt to get a response.")