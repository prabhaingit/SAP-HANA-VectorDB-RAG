
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

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
port = int(os.environ.get('PORT', 4000))
openai.api_key = os.environ.get('OPENAI_API_KEY')
# Initialize HANA connection and components
def init_hana_components():
    # HANA connection settings

    HANA_CONNECTION = dbapi.connect(
    address="927fbf89-6ab2-4d71-9901-06bc6c363fc0.hana.trial-us10.hanacloud.ondemand.com",
    port="443",
    user="DBADMIN",
    password="ObSq6arqi31!",
    autocommit=True,
    sslValidateCertificate=False
    )
     
    # Initialize embeddings model
    
    embeddings = OpenAIEmbeddings()
    
    # Initialize HANA vector store
    vectorstore = HanaDB(
        connection_info=HANA_CONNECTION,
        embedding_function=embeddings,
        table_name="STATE_OF_THE_UNION"  # Your vector table name
    )

    # Create a retriever instance of the vector store
    retriever = vectorstore.as_retriever()
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai.api_key
    )
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

# Initialize components
qa_chain = init_hana_components()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_query = request.form['user_input']
            
            # Get response from RAG chain
            result = qa_chain({"question": user_query})
            
            # Extract answer and sources
            answer = result['answer']
            sources = [doc.metadata.get('source', 'Unknown') 
                      for doc in result.get('source_documents', [])]
            
            # Prepare output
            output = {
                'answer': answer,
                'sources': sources,
                'query': user_query
            }
            
            return render_template('ragindex.html', output=output)
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            return render_template('ragindex.html', error=error_message)
            
    return render_template('ragindex.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
