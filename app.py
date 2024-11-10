from beyondllm import source,retrieve,generator,embeddings,llms

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

HF_TOKEN = st.secrets['HF_TOKEN']
GOOGLE_API_KEY =  st.secrets['GOOGLE_API_KEY']
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_PATH = "./data/"

# os.environ['GROQ_API_KEY'] = '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'
# os.environ['GROQ_MODEL'] = 'mixtral-8x7b-32768'
# os.environ['GROQ_TEMP'] = '0.1'


@st.cache_data
def load_data(csv_paths):
    data = source.fit(path=csv_paths, dtype="csv")
    return data

if HF_TOKEN and GOOGLE_API_KEY:
    data = load_data(DATA_PATH)
    embed_model = embeddings.GeminiEmbeddings(api_key=google_api_key,
                                          model_name="models/embedding-001")
    retriever = retrieve.auto_retriever(data=data, 
                                        embed_model=embed_model,
                                        type="hybrid", 
                                        top_k=5,
                                        mode="OR")
    llm = llms.GeminiModel(model_name="gemini-pro",
                       google_api_key=GOOGLE_API_KEY)
    question = st.text_input("Ask me anything regarding your past transactions...")
    with st.expander("Sample Questions"):
        st.write("""
        1. What amount was paid to Ramesh babu and on which date?
        2. What transaction happened on 19th October 2024?
        3. What amount was paid to Ramesh Choupardu?
        """)

    system_prompt = "You are an accountant, who answers questions based on the ledgers found in the csvs. Each ledger has date, the amount that was debited, type of payment cash or anything else. Based on the context answer the quesetion."
    if question:
        pipeline = generator.Generate(question=question,
                                      retriever=retriever, 
                                      llm=llm, 
                                      system_prompt=system_prompt)
        response = pipeline.call()
        st.write(response)

    st.divider()