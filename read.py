import os
import fitz
def read_pdf_files(folder_path):
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    print(pdf_files)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(pdf_path)
        doc = fitz.open(pdf_path)
        print(f"Docs: {doc.page_count}")

    #     for page_number in range(doc.page_count):
    #         page = doc[page_number]
    #         text = page.get_text("text")
            
    #         # Process the text as needed (e.g., analyze, extract information, etc.)
    #         # Example: print the text of each page
    #         print(f"Text from {pdf_file}, Page {page_number + 1}:\n{text}\n")

        doc.close()
        
        
folder_path = "./resume"
read_pdf_files(folder_path)