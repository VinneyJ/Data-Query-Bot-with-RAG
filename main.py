import os
from filetype import guess
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.vectorstores import FAISS
import pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  
    environment=os.getenv("PINECONE_ENV"),  
)
open_ai = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=open_ai)


embeddings = OpenAIEmbeddings(openai_api_key=open_ai)

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

research_paper_path = "transformer.pdf"

def extract_file_content(file_path):
    
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
    documents = loader.load()
    #print(f"dOCUMENTS cONTENT: {documents}")
    documents_content = '\n'.join(doc.page_content for doc in documents)
    return documents_content

research_paper_content = extract_file_content(research_paper_path)
nb_characters = 400

text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
research_paper_chunks = text_splitter.split_text(research_paper_content)

# Connect Vector 
# def get_doc_search(documents):
#     index_name = "langchain-demo"
#     if index_name not in pinecone.list_indexes():
#         pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
#     texts = [d for d in documents]
#     docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
#     return docsearch

chain = load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)



def get_doc_search(text_splitter):
    
    return FAISS.from_texts(text_splitter, embeddings)

def chat_with_file(file_path, query):
    
    file_content = extract_file_content(file_path)
    file_splitter = text_splitter.split_text(file_content)
    #print(file_splitter)
    
    document_search = get_doc_search(file_splitter)
    documents = document_search.similarity_search(query)
    
    results = chain({
                        "input_documents":documents, 
                        "question": query
                    }, 
                    return_only_outputs=True)
    results = results['intermediate_steps'][0]
    
    return results
#query = "Why is the self-attention approach used in this document?"
query = str(input("Ask Something..."))

results = chat_with_file(research_paper_path, query)
print(f"THese are results {results}")

answer = results["answer"]
confidence_score = results["score"]

print(f"Answer: {answer}\n\nConfidence Score: {confidence_score}")