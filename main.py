import requests

url = "https://api.unstructured.io/general/v0/general"

headers = {
    "accept": "application/json",
    "unstructured-api-key": "<YOUR API KEY>"
}

data = {
    "strategy": "hi_res",
    "pdf_infer_table_structure": "true"
}

file_path = "/Path/To/File"
file_data = {'files': open(file_path, 'rb')}

response = requests.post(url, headers=headers, files=file_data, data=data)

file_data['files'].close()

json_response = response.json()