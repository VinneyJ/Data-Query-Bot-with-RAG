import os
import time
from filetype import guess
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import pinecone


open_ai= os.getenv("OPENAI_API_KEY")


def detect_document_type(document_path):
    """
    Detects the file type
    """
    guess_file = guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    
    if(guess_file.extension.lower() == "pdf"):
        file_type = "pdf"
        
    elif(guess_file.extension.lower() in image_types):
        file_type = "image"
        
    else:
        file_type = "unkown"
        
    return file_type



def extract_file_content(file_path):
    """
    Exracts content from file
    """
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
    documents = loader.load()
    documents_content = '\n'.join(doc.page_content for doc in documents)
    return documents_content
   



text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 150,
    chunk_overlap  = 10,
    length_function = len,
)


chain = load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)



def save_to_db(index, embeddings, file_path):
    """
    Saves embeddings to FAISS DB
    """

    file_content = extract_file_content(file_path)
    file_splitter = text_splitter.split_text(file_content)

    try:
        faiss_db = FAISS.from_documents(file_splitter, embeddings) 
    except Exception as e:
        faiss_db = FAISS.from_texts(file_splitter, embeddings)
  
    if os.path.exists(index):
        local_db = FAISS.load_local(index, embeddings)
        #merging the new embedding with the existing index store
        local_db.merge_from(faiss_db)
        print("Merge completed")
        local_db.save_local(index)
        print("Updated index saved")
    else:
        faiss_db.save_local(folder_path=index)
        print("New store created...")

    
def search_index(query, embeddings):
    """
    Handles Search query
    """

    # Load the Faiss index
    index = FAISS.load_local("faiss_index", embeddings)

    # Search the index
    documents = index.similarity_search(query)
    results = chain({
                        "input_documents":documents, 
                        "question": query
                    }, 
                    return_only_outputs=True)
    results = results['intermediate_steps'][0]
    
    return results


def read_pdf_files(folder_path):
    """
    Reads pdf Files folder path
    """
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai)
    
    index = "faiss_index"

    pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    print(pdf_files)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        time.sleep(25)
        save_to_db(index, embeddings, pdf_path)
        
