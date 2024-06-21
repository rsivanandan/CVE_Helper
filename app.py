import streamlit as st
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import openai
import re

regex = r"CVE-\d{4}-\d{4,7}"

@st.cache_resource(show_spinner=False)
def init_llm_ollama():
    llm = Ollama(model="llama3", request_timeout=600.0)
    return llm


@st.cache_resource(show_spinner=False)
def init_llm_hf():

    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.3", token=HF_Token
    )
    return llm


def init_llm_OpenAI():

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5)
    return llm


def bot(llm) -> None:
        user_input = st.text_input("Provide the CVE Number",placeholder="CVE-2021-44832")
        template_1 = (
            "Using this context : \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Give a brief description of what this BRIEF : {query_str}\n"
            "Under the heading Mitigation: Answer should include ways on how this CVE can be mitigated"
            "If you are not able to get any context, ONLY then say 'I don’t know, Sorry'. » "
        )

        template_2 = (
            "Using this context : \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Write a powershell script to automate the mitigation of the vulnerability and give only the script: {query_str}\n"
            
        )

        qa_template_1 = PromptTemplate(template_1)
        qa_template_2 = PromptTemplate(template_2)
        prompt_1 = qa_template_1.format(
            context_str=updateCVE(user_input), query_str=user_input
        )

        if st.button("Get Details"):
            with st.spinner("Thinking..."):
                with st.container(border=True):
                    output = llm.complete(prompt_1)
                    st.markdown(output)
                    prompt_2 = qa_template_2.format(context_str=output, query_str="")
                with st.expander("PowerShell Script"):
                    st.code(llm.complete(prompt_2))
            st.write(
                "Caution : ***AI generated script, Do test it diligently before rolling it out*"
            )
            st.write("*CVE Details fetched from https://cvedetails.com*")



st.set_page_config(
    page_title="CVE Helper",
    page_icon="./CVE.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)


@st.cache_resource(show_spinner=False)
def updateCVE(CVE) -> str:
    url = "https://www.cvedetails.com/cve/{}".format(CVE)
    session = HTMLSession()
    response = session.get(url)
    soup = BeautifulSoup(response.content, "lxml").text
    cleantext = soup.replace("\n", "").replace("\t", "")
    return cleantext[:250]


# Streamlit UI
st.title("CVE Helper Bot")
st.sidebar.image("CVE.png")

llmtype = st.radio(
    "Choose your LLM",
    ["Huggingface Mistral", "Open AI gpt-3.5-turbo", "Ollama llama3 (local)"],
    horizontal=True,
)
if llmtype == "Ollama llama3 (local)":
    st.sidebar.write(
        "*Ensure that you're running Ollama locally and also have downloaded the llama3 model*"
    )
    bot(init_llm_ollama())

elif llmtype == "Open AI gpt-3.5-turbo":
    with st.sidebar.container(border=True):
        st.sidebar.write(
            "*Signup for a free account at https://platform.openai.com/. Create a free API key and input below*"
        )
        # os.environ['OPENAI_API_KEY'] = st.sidebar.text_input("Open API Key", type="password")
    openai.api_key = st.sidebar.text_input("Open API Key", type="password")
    st.sidebar.image("openai.webp")
    if openai.api_key:
        bot(init_llm_OpenAI())

elif llmtype == "Huggingface Mistral":
    with st.sidebar.container(border=True):
        st.sidebar.write(
            "*Signup for a free account at https://huggingface.co/. Create a free API key and input below*"
        )
    HF_Token = st.sidebar.text_input("Huggingface Token", type="password")
    st.sidebar.image("hf.svg")
    # init_llm_hf()
    if HF_Token:
        bot(init_llm_hf())


