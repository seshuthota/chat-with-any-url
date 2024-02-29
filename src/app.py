import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    return conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })["answer"]


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(document_chunks,
                                         embedding=embedding)
    return vector_store


def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation , generate a search query (ONLY SEARCH QUERY) to look up in the order to get "
         "information relevant the conversation"),
    ])

    if "llm" not in st.session_state:
        st.session_state.llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    retriever_chain = create_history_aware_retriever(st.session_state.llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context} KEEP THE ANSWERS CLEAR, "
         "PROVIDE DETAILED CODE WHEN NEEDED"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(st.session_state.llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


st.set_page_config(page_title="Chat with URL", page_icon=":tada:", layout="wide")
st.title("Chat with URL")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Webpage URL")
    api_key = st.text_input("Google API Key")

if (website_url is None or website_url == "") and api_key is None and api_key == "":
    st.session_state.api_key = api_key
    st.info("Enter the URL of the webpage you want to chat with")
else:
    if website_url and "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you today?"),
        ]

    user_query = st.chat_input("Type your message here...")

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        response = get_response(user_query)
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
