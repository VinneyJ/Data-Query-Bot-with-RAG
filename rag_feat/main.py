import os
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

research_paper_path = "TaxProceduresAct29of2015.pdf"

def extract_file_content(file_path):
    
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
    documents = loader.load()
    # documents_content = '\n'.join(doc.page_content for doc in documents)
    # return documents_content
    return documents



text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
#research_paper_chunks = text_splitter.split_documents(research_paper_content)


chain = load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)



# def get_doc_search(text_splitter):
#     db = FAISS.from_documents(text_splitter, embeddings)
#     db.save_local("faiss_index")

#     new_db = FAISS.load_local("faiss_index", embeddings)
#     return new_db

def save_to_db(index, embeddings, file_path):
    current_dir = os.getcwd()

   # Check if the "faiss_index" directory exists in the current directory
    if not os.path.exists(os.path.join(current_dir, "faiss_index")):
       # If not, create it
       os.makedirs(os.path.join(current_dir, "faiss_index"))
    file_content = extract_file_content(file_path)
    file_splitter = text_splitter.split_documents(file_content)
    db = index.from_documents(file_splitter, embeddings)
        
    db.save_local("faiss_index")
    print("Indexes saved to Faiss Index DB.")
    
def search_index(query):
    # Load the Faiss index
    index = FAISS.load_local("faiss_index")

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
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai)
    index = FAISS.load_local("./faiss_index", embeddings)

    pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    print(pdf_files)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        save_to_db(index, embeddings, pdf_path)
        
     

# def chat_with_file(file_path, query):
    
#     file_content = extract_file_content(file_path)
#     file_splitter = text_splitter.split_documents(file_content)
#     #print(file_splitter)
    
#     document_search = get_doc_search(file_splitter)
#     documents = document_search.similarity_search(query)
    
#     results = chain({
#                         "input_documents":documents, 
#                         "question": query
#                     }, 
#                     return_only_outputs=True)
#     results = results['intermediate_steps'][0]
    
#     return results






#query = "Why is the self-attention approach used in this document?"
# query = str(input("Ask Something..."))

# results = chat_with_file(research_paper_path, query)
# print(f"THese are results {results}")

# answer = results["answer"]
# confidence_score = results["score"]

# print(f"Answer: {answer}\n\nConfidence Score: {confidence_score}")