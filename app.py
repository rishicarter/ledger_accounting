import pandas as pd
import numpy as np
import os
import logging

import streamlit as st
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
import chromadb

from src.controllers import excelController as exCon
from src.controllers import ragController as ragCon

chromadb.api.client.SharedSystemClient.clear_system_cache()

st.set_page_config(
    page_title="LedgersGPT",
    page_icon="💹", 
    layout="wide",
    initial_sidebar_state='expanded',
    menu_items={
        'About': "# App to ask, explore about Account Transactions!"
    }
)

st.title("💹 LedgersGPT")
st.subheader("RAG with knowledge about account transactions. Ask any question related to past transactions!")

## Set Session states
st.session_state.HF_TOKEN = st.secrets['HF_TOKEN']
st.session_state.GOOGLE_API_KEY =  st.secrets['GOOGLE_API_KEY']
MODEL_NAME = "BAAI/bge-small-en-v1.5"
flag = False
ledger_df = pd.DataFrame()
RETRY_COUNTER = 3

DATA_PATH, OUTPUT_PATH, LOG_PATH, LOG_FILENAME = "./data/", "./output/", "./logs/", "activity.log"

if "init" not in st.session_state:
    st.session_state.init = True  
    st.session_state['counter'] = 0
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    # Logging
    logging.basicConfig(filename=LOG_PATH+LOG_FILENAME,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

    logging.info("Running LedgersGPT..")

uploaded_file = st.file_uploader("Upload a Ledger file", type=['xlsx'])
if uploaded_file is not None:
    try:
        ledger_df = exCon.handle_uploaded_file(uploaded_file)
    except Exception as ex:
        st.write("Excel format not compatible...")
        logging.error(f"Upload Error: {ex}", exc_info=True)

if st.session_state.HF_TOKEN and st.session_state.GOOGLE_API_KEY and ledger_df is not None and (not ledger_df.empty):
    try:
        rag_chain = ragCon.rag_generater(ledger_df)
        flag = True
    except Exception as ex:
        st.write("Something went wrong...")
        logging.error(f"RAG Error: {ex}", exc_info=True)
        flag = False
    

if flag == True:
    question = st.text_input("Ask me anything regarding your past transactions...")
    
    
    ledger_df['amount'] = ledger_df['amount'].astype(float)

    if question:
        logging.info(f"Question asked: {question}")
        with st.spinner('Searching....'):
            while st.session_state['counter'] != RETRY_COUNTER:
                try:
                    st.session_state['counter'] += 1
                    query = rag_chain.invoke(question)
                    logging.info(f"Response {st.session_state['counter']}:{query}")
                    result = eval(query)
                except Exception as ex:
                    logging.error(f"Response Error: {ex}")
                    result = "Unable to find reference from context."
            st.session_state['counter'] = 0
        st.write(result)

    st.divider()