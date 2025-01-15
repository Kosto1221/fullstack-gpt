import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
import os
import openai

class ChatCallbackhandler(BaseCallbackHandler):

    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



st.set_page_config(
    page_title="Document GPT",
    page_icon="📃",
)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    embeddings_path = f"./.cache/embeddings/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(embeddings_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600 , chunk_overlap=100)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=api_key) 
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
     st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
       save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DON'T make anything up.

        Context: {context}
        """
         ),
        ("human", "{question}")
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about your files! 

    Upload your files on the sidebar.
    """
)

with st.sidebar:
    upload_disabled = True
    api_key = st.text_input("Insert OpenAI API Key", type="password")
    if api_key:
        try:
            openai.api_key = api_key
            openai.models.list()
            upload_disabled=False
        except openai.AuthenticationError:
            st.error("Invalid API key. Please check and try again.")
    
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"], disabled=upload_disabled)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.1, streaming=True, callbacks=[ChatCallbackhandler()])

if file:
    retriever = embed_file(file, api_key)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = {"context": retriever | RunnableLambda(format_docs),
                 "question": RunnablePassthrough()} | prompt | llm
        with st.chat_message("ai"):
            chain.invoke(message)

else: 
    st.session_state["messages"] = []
    

