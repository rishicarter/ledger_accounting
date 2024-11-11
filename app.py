import pandas as pd
import numpy as np
import os

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

HF_TOKEN = st.secrets['HF_TOKEN']
GOOGLE_API_KEY =  st.secrets['GOOGLE_API_KEY']
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_PATH, OUTPUT_PATH, OUT_FILENAME = "./data/", "./output/", "autoledgers.csv"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

ledger_df = pd.DataFrame()

st.title("💹 LedgersGPT")
st.subheader("RAG with knowledge about account transactions. Ask any question related to past transactions!")

@st.cache_data
def load_data(csv_paths):
    loader = CSVLoader(file_path=OUTPUT_PATH+OUT_FILENAME)
    docs = loader.load()
    return docs

def preprocess(df: pd.DataFrame):
    df = df.copy()
    index = df[df.iloc[:,0] == 'Date'].index[0]
    df1 = df.iloc[index+1:,]
    current_cols = df1.columns
    modified_cols = ['date', 'direction', 'transaction_desc', 'vch_type', 'vch_no', 'debit', 'credit']
    df1 = df1.rename(columns=dict(zip(current_cols, modified_cols)))
    df1['transaction_details'] = df1.apply(lambda row: row['transaction_desc'] if pd.isna(row['date']) and pd.isna(row['direction']) else None, axis=1)
    df1['transaction_details'] = df1['transaction_details'].bfill()
    df1 = df1[~df1['transaction_desc'].str.contains('Opening Balance|Closing Balance', case=False, na=False)]
    df1 = df1.dropna(subset=['date', 'direction', 'transaction_desc', 'vch_type', 'vch_no'])
    df1['direction'] = df1['direction'].apply(lambda x: "No" if x == "By" else "Yes")
    df1['debit'] = df1['debit'].fillna(df1['credit'])
    df1.rename(columns={"debit": "amount", "direction": "is_amount_debited"}, inplace=True)
    df1.drop('credit', axis=1, inplace=True)
    
    return df1

def load_df(dfs_list):
    dfs = []
    for df in dfs_list:
        df = preprocess(df)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    # df.to_csv(OUTPUT_PATH+OUT_FILENAME, index=False)
    return df
    

uploaded_file = st.file_uploader("Upload a Ledger file", type=['xlsx'])
if uploaded_file is not None:
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names
    dfs = []
    for sheet_name in sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        dfs.append(df)
    try:
        ledger_df = load_df(dfs)
    except:
        st.write("Excel format not compatible...")

if HF_TOKEN and GOOGLE_API_KEY and (not ledger_df.empty):
    with st.expander("Sample Questions"):
        st.write("""
        1. What all amounts were paid to Ramesh babu and on which dates?
        2. What amounts got debited during 19th October 2024?
        3. What were the cash amounts debited or credited to Aishwarya?
        4. How much total amount was credited?
        5. What were the Registration Charges?
        """)
    # ledger_df['amount'] = ledger_df['amount'].astype(float)
    question = st.text_input("Ask me anything regarding your past transactions...")
    docs = load_data(DATA_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50})
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GOOGLE_API_KEY)
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    template = """You are provided with ledger data stored in a DataFrame with the following columns:

    1. `date`: The date and time of the transaction. Dtype: Timestamp
    2. `is_amount_debited`: Indicates if the transaction is a debit ("Yes") or a credit ("No"). Dtype: String
    3. `transaction_desc`: The account involved in the transaction (e.g., Cash, Bank Name, Individual’s Name). Dtype: String
    4. `vch_type`: The type of voucher associated with the transaction (e.g., Payment, Receipt, Journal, Contra). Dtype: String
    5. `vch_no`: The voucher number associated with the transaction, if available. Dtype: float
    6. `amount`: The amount debited or credited in the transaction. Dtype: float
    7. `transaction_details`: Additional details related to the transaction, including unstructured notes or recipient names. Dtype: String
    
    Generate a Python code snippet to retrieve data from this DataFrame (named `ledger_df`) based on the provided question. The code should apply filters and transformations according to the question and should be written as a single line without line breaks. If any entity is specified in the question (such as a name, organization, or account name), ensure the code filters for this entity in both `transaction_desc` and `transaction_details` columns, using a case-insensitive search. Date-based filters should apply to the `date` column, and amount-based filters should apply to the `amount` column. Treat amount column as float. Always return a dataframe query like mentioned in the examples.
    
    Examples:
    
    1. Question: "Retrieve all debit transactions made to HDFC Bank in June 2024."
       - Code: ledger_df[(ledger_df['is_amount_debited'] == 'Yes') & (ledger_df['transaction_desc'].str.lower().str.contains('hdfc bank') | ledger_df['transaction_details'].str.lower().str.contains('hdfc bank')) & (ledger_df['date'].between('2024-06-01 00:00:00', '2024-06-30 00:00:00'))]
    
    
    2. Question: "List all transactions where more than 500,000 was credited, including transaction descriptions."
       - Code: ledger_df[(ledger_df['is_amount_debited'] == 'No') & (ledger_df['amount'] > 500000)]
    
    3. Question: "Find the total amount debited for each voucher type in July 2024."
       - Code: ledger_df[(ledger_df['is_amount_debited'] == 'Yes') & (ledger_df['date'].between('2024-07-01 00:00:00', '2024-07-31 00:00:00'))].groupby('vch_type')['amount'].sum()
    
    4. Question: "List all transactions to Ramesh Choupardu."
       - Code: ledger_df[ledger_df['transaction_desc'].str.lower().str.contains('ramesh choupardu') | ledger_df['transaction_details'].str.lower().str.contains('ramesh choupardu')]
    
    Now, here’s the below the context and question for the DataFrame query generation:
    
    context: {context}
    
    Question: {question}"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    ledger_df['amount'] = ledger_df['amount'].astype(float)

    # system_prompt = "You are an accountant, who answers questions based on the ledgers found in the csvs. Each ledger has date, the amount that was debited, type of payment cash or anything else. Based on the context answer the quesetion."
    if question:
        # pipeline = generator.Generate(question=question,
        #                               retriever=retriever, 
        #                               llm=llm, 
        #                               system_prompt=system_prompt)
        # response = pipeline.call()

        with st.spinner('Searching....'):
            try:
                query = rag_chain.invoke(question)
                result = eval(query)
            except:
                query = rag_chain.invoke(question)
                result = eval(query)
        st.write(result)

    st.divider()