
import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request
#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB
from hdbcli import dbapi
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_core.prompts import PromptTemplate


load_dotenv()

app = Flask(__name__)
port = int(os.environ.get('PORT', 4000))
openai.api_key = os.environ.get('OPENAI_API_KEY')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    # Read PDF
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    
    return chunks

def store_in_hana(chunks, filename):
    # Initialize HANA connection
    HANA_CONNECTION = dbapi.connect(
        address="927fbf89-6ab2-4d71-9901-06bc6c363fc0.hana.trial-us10.hanacloud.ondemand.com",
        port="443",
        user=os.environ.get('HANA_USER'),
        password=os.environ.get('HANA_PASSWORD'),
        autocommit=True,
        sslValidateCertificate=False
    )
    
    # Initialize embeddings model
    
    embeddings = OpenAIEmbeddings()
    
    
    # Initialize vector store
    vectorstore = HanaDB(
        connection=HANA_CONNECTION,
        embedding=embeddings,
        table_name="UPLOAD_DOCUMENT_VECTORS"
    )
    
    # Create documents with metadata
    texts = chunks
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
    
    # Add documents to vector store
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    
    return len(chunks)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('ragindexupload.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('ragindexupload.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Process PDF and store in HANA
                chunks = process_pdf(file_path)
                chunk_count = store_in_hana(chunks, filename)
                
                # Clean up uploaded file
                os.remove(file_path)
                
                return render_template('ragindexupload.html', 
                                     success=True,
                                     filename=filename,
                                     chunk_count=chunk_count)
            
            except Exception as e:
                return render_template('ragindexupload.html', 
                                     error=f'Error processing file: {str(e)}')
        
        return render_template('ragindexupload.html', 
                             error='Invalid file type. Please upload a PDF.')
    
    return render_template('ragindexupload.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=port, debug=True)
