
import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB
from hdbcli import dbapi
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
port = int(os.environ.get('PORT', 4000))
openai.api_key = os.environ.get('OPENAI_API_KEY')
# Initialize HANA connection and components
def init_hana_components():
    # HANA connection settings
    HANA_CONNECTION = dbapi.connect(
    address="xxxxxxxxxxxxxxx.hana.trial-us10.hanacloud.ondemand.com",
    port="443",
    user=os.environ.get('HANA_USER'),
    password=os.environ.get('HANA_PASSWORD'),
    autocommit=True,
    sslValidateCertificate=False
    )
     
    # Initialize embeddings model
    
    embeddings = OpenAIEmbeddings()
    
    # Initialize HANA vector store
    db = HanaDB(
    embedding=embeddings, connection=HANA_CONNECTION, table_name="UPLOAD_DOCUMENT_VECTORS"
    )
   

    # Create a retriever instance of the vector store
    retriever = db.as_retriever()

    prompt_template = """
    You are an expert in state of the union topics. You are provided multiple context items that are related to the prompt you have to answer.
    Use the following pieces of context to answer the question at the end.

    '''
    {context}
    '''

    Question: {question}
    """

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(
       memory_key="chat_history", output_key="answer", return_messages=True
    )
    
      
    # Create retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    
    return qa_chain

    # Initialize components
qa_chain = init_hana_components()

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       try:
           user_query = request.form['user_input']
           result = qa_chain({"question": user_query})
           
           output = {
               'answer': result['answer'],
               #'sources': [doc.metadata.get('source', 'Unknown') for doc in result.get('source_documents', [])],
               'source_documents': result.get('source_documents', []),
               'query': user_query
           }
           
           return render_template('ragindex.html', output=output)
           
       except Exception as e:
           return render_template('ragindex.html', error=f"Error: {str(e)}")
           
   return render_template('ragindex.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=port, debug=True)
