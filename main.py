# import re
# import requests
# # from unstructured.documents.elements import Text
# # from unstructured.partition.pdf import partition_pdf
# from unstructured.cleaners.core import replace_unicode_quotes
# from unstructured.cleaners.core import clean


# from unstructured.cleaners.core import group_broken_paragraphs

# para_split_re = re.compile(r"(\s*\n\s*){3}")

# url = "https://api.unstructured.io/general/v0/general"

# headers = {
#     "accept": "application/json",
#     "unstructured-api-key": "yGqfsn0x5vm41J6BTUsc5XbNXiRsJO"
# }

# data = {
#     "strategy": "hi_res",
#     "pdf_infer_table_structure": "true"
# }

# file_path = "resume/Vincent-Otieno-Resume.pdf"
# file_data = {'files': open(file_path, 'rb')}

# response = requests.post(url, headers=headers, files=file_data, data=data)

# file_data['files'].close()

# json_response = response.json()

# elem = "\n\n"
# for el in json_response:
#     s = str(el['text'])




# new_file = "\n\n".join([ clean(replace_unicode_quotes(str(el['text'])),bullets=True, lowercase=True, trailing_punctuation=True,extra_whitespace=True) for el in json_response])

# replace_unicode_quotes(new_file)


# print(new_file)
# element = Text(new_file)
# element.apply(replace_unicode_quotes)
# print(element)




# #filename = 'Vincent-Otieno-Resume.pdf'
# #filename = os.path.join(resume, "layout-parser-paper-fast.pdf")
# elements = partition(filename='Vincent-Otieno-Resume.pdf', content_type="application/pdf")
# print("\n\n".join([str(el) for el in elements][:10]))

# from unstructured.partition.auto import partition

# elements = partition(filename="Vincent-Otieno-Resume.pdf")


# # with open("mydoc.docx", "rb") as f:
# #     elements = partition(file=f)
# #     print("\n\n".join([str(el) for el in elements][:10]))

# print(element)
# import re
# from unstructured.cleaners.core import group_broken_paragraphs

# para_split_re = re.compile(r"(\s*\n\s*){3}")

# text = """The big brown fox

# was walking down the lane.


# At the end of the lane, the

# fox met a bear."""
# print(f"Before: \n{text}")

# text2 = group_broken_paragraphs(text, paragraph_split=para_split_re)
# print(f"After: \n{text2}")
import getpass
import os
import re
import uuid
from typing import Any
from PIL import Image
import fitz
import gradio as gr
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

enable_box = gr.Textbox(value=None, placeholder='Upload your OpenAI API key', interactive=True)
disable_box = gr.Textbox(value = 'OpenAI API key is Set', interactive = False)

def set_apikey(api_key: str):
        app.OPENAI_API_KEY = api_key        
        return disable_box

def enable_api_box():
        return enable_box

def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY: str = None ) -> None:
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0

    def __call__(self, file: str) -> Any:
        if self.count==0:
            self.chain = self.build_chain(file)
            self.count+=1
        return self.chain

    def chroma_client(self):
        #create a chroma client
        client = chromadb.Client()
        #create a collecyion
        collection = client.get_or_create_collection(name="my-collection")
        return client
    # def pinecone_client(self):
    #     # create a Pinecone client
    #     client = pinecone.Client(api_key='your-api-key')
    #     # create a collection
    #     collection = client.get_or_create_collection(name="my-collection")
    #     return client

    def process_file(self,file: str):
        loader = PyPDFLoader(file.name)
        documents = loader.load()  
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        return documents, file_name

    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
                retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True,)
        return chain
    # def build_chain(self, file: str):
    #     documents, file_name = self.process_file(file)
    #     #Load embeddings model
    #     embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
    #     pdfsearch = Pinecone.from_documents(documents, embeddings, collection_name= file_name,)
    #     chain = ConversationalRetrievalChain.from_llm(
    #             ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
    #             retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
    #             return_source_documents=True,)
    #     return chain


def get_response(history, query, file): 
        if not file:
            raise gr.Error(message='Upload a PDF')    
        chain = app(file)
        result = chain({"question": query, 'chat_history':app.chat_history},return_only_outputs=True)
        app.chat_history += [(query, result["answer"])]
        app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['answer']:
           history[-1][-1] += char
           yield history,''

def render_file(file):
        doc = fitz.open(file.name)
        page = doc[app.N]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

def render_first(file):
        doc = fitz.open(file.name)
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]

app = my_app()
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(placeholder='Enter OpenAI API key', show_label=False, interactive=True)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')
        with gr.Row():           
            chatbot = gr.Chatbot(value=[], elem_id='chatbot', height=650)
            show_img = gr.Image(label='Upload PDF', height=680)
    with gr.Row():
        with gr.Column(scale=0.60):
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False
                    )
        with gr.Column(scale=0.20):
            submit_btn = gr.Button('submit')
        with gr.Column(scale=0.20):
            btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"])

    api_key.submit(
            fn=set_apikey, 
            inputs=[api_key], 
            outputs=[api_key,])
    change_api_key.click(
            fn= enable_api_box,
            outputs=[api_key])
    btn.upload(
            fn=render_first, 
            inputs=[btn], 
            outputs=[show_img,chatbot],)

    submit_btn.click(
            fn=add_text, 
            inputs=[chatbot,txt], 
            outputs=[chatbot, ], 
            queue=False).success(
            fn=get_response,
            inputs = [chatbot, txt, btn],
            outputs = [chatbot,txt]).success(
            fn=render_file,
            inputs = [btn], 
            outputs=[show_img]
    )



  

if __name__ == "__main__":
    demo.queue()
    demo.launch(show_api=False, share=True)   