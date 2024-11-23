import pandas as pd
from io import StringIO
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

PROMPT_TEMPLATE = """You are provided with ledger data stored in a DataFrame with the following columns:

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

OUTPUT_PATH, OUT_FILENAME = "./output/", "autoledgers.csv"
def load_data(df: pd.DataFrame):
    logging.info("OUT_FILE: ", OUTPUT_PATH+OUT_FILENAME)
    df.to_csv(OUTPUT_PATH+OUT_FILENAME, index=False)
    loader = CSVLoader(file_path=OUTPUT_PATH+OUT_FILENAME)
    docs = loader.load()
    return docs

def rag_generater(ledger_df: pd.DataFrame):
    """
    Function to generate RAG model.
    ------
    PARAMS
    ------
    ledger_df: Dataframe containing the ledgers
    ------
    RETURNS
    -------
    Rag_chain
    """
    docs = load_data(ledger_df)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=st.session_state.HF_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50})
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=st.session_state.GOOGLE_API_KEY)
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    template = PROMPT_TEMPLATE

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain