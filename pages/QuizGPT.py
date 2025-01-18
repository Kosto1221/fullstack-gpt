import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import json
import openai
import os

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="❓",
)

st.title("Quiz GPT")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.

            Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

            Each question should have 4 answers, three of them must be incorrect and one should be correct.

            Use (o) to signal the correct answer.

            Question example:

            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)

            Question: What is the capital of Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut

            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998

            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model

            Your turn!

            Context: {context}
            """
        )
    ]
)



@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600 , chunk_overlap=100)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
        chain = {"context": format_docs} | questions_prompt | llm
        response = chain.invoke(_docs)
        response = response.additional_kwargs["function_call"]["arguments"]
        return json.loads(response)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(topic)
    return docs

with st.sidebar:
    docs = None
    topic = None
    change_disabled = True
    api_key = st.text_input("Insert OpenAI API Key", type="password")
    if api_key:
        try:
            openai.api_key = api_key
            openai.models.list()
            change_disabled=False
            llm  = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True, api_key=api_key, callbacks=[StreamingStdOutCallbackHandler()]).bind(function_call={"name":"create_quiz"}, functions=[function])
        except openai.AuthenticationError:
            st.error("Invalid API key. Please check and try again.")   

    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article"), disabled=change_disabled)
    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt or .pdf file", type=["txt", "docx", "pdf"], disabled=change_disabled)
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    st.link_button("Visit repository", "https://github.com/Kosto1221/fullstack-gpt")

if "grade" not in st.session_state:
    st.session_state.grade = []

if "clear_form" not in st.session_state:
    st.session_state.clear_form = False
                
if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.

        I will make a quiz from Wikipeida articles or files you upload to test your knowledge and help you study.

        Get started by uploading a file or seraching on Wikipedia in the sidebar.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    correct_count = 0
    with st.form("questions_form", clear_on_submit=st.session_state.clear_form):
        for i, question in enumerate(response["questions"]):
            st.write(f"{i+1} .",question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                label_visibility="collapsed",
                key=question["question"]
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct")
                correct_count += 1
                st.session_state.grade.append("correct")
            elif value is not None:
                st.error("Wrong")
                st.session_state.grade.append("wrong")
            st.divider()

        col1, col2, col3 = st.columns([1, 5.7, 1], vertical_alignment="center")

        with col1:
            submit_button = st.form_submit_button(disabled=len(st.session_state.grade) == 10)
            if submit_button:
                if len(st.session_state.grade) == 10:
                    if all(answer == "correct" for answer in st.session_state.grade):
                        st.balloons()
                    else:
                        st.snow()
                del st.session_state.grade

        with col2:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    height: 100%;
                    font-size: 20px;
                    font-weight: bold;
                    margin-top: -15px;
                ">
                    <span style="margin-right: 10px;">SCORE</span>
                    <span style="margin-right: 10px;">:</span>
                    <span style="margin-right: 10px;">{correct_count}</span>
                    <span style="margin-right: 10px;">/</span>
                    <span>10</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            restart_button =  st.form_submit_button("Restart")
            if restart_button:
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

