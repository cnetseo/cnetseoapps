import requests
import langchain
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv
import asyncio
import json 

serpapikey = '342639d3011a832274782f0b02e492371835e6f436d09268978f7a827ecb741b'
url_to_test = "https://cnetdev-37797.nonprod.cnet.com/tech/services-and-software/best-vpn/"
api_key = "Basic Y2Nhc2F6emFAcmVkdmVudHVyZXMuY29tOjMwNGViMWI5MDQ3ZTZjOTc="

def getIndexInfo(url_to_test):
    loader = AsyncHtmlLoader(url_to_test)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=0
        )
    splits = splitter.split_documents(docs_transformed)
    page_contents = [split.page_content for split in splits]
    return page_contents

def extract_results(items):
    urls = []
    count = 0  
    for item in items:
        if item.get('type') == 'organic': 
            urls.append(item.get('url'))
            
            count += 1  
            if count == 5:  
                break  
    print(urls)
    return urls

def getSERPInfo(keyword, url_to_test):
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    try:
        payload = json.dumps([{
            "keyword": keyword,
            "location_code": 2826,
            "language_code": "en",
            "device": "desktop",
            "os": "windows",
            "depth": 100
        }])
    except UnicodeEncodeError:
        try:
            # Try encoding the keyword in UTF-8
            encoded_keyword = keyword.encode('utf-8')
            payload = json.dumps([{
                "keyword": encoded_keyword.decode('utf-8'),
                "location_code": 2826,
                "language_code": "en",
                "device": "desktop",
                "os": "windows",
                "depth": 100
            }])
        except:
            print(f"Error encoding keyword: {keyword}. Skipping this keyword.")
            return {
                'keyword': keyword,
                'Indexed?': 'Error - Encoding Issue'
            }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = response.json()

        if response_data["status_code"] == 20000:
            for result in response_data["tasks"][0]["result"]:
                urls = extract_results(result['items'])
                indexed = "Yes" if url_to_test in urls else "No"
                return {
                    'keyword': keyword,
                    'Indexed?': indexed
                }
        else:
            print(f"API error. Code: {response_data['status_code']} Message: {response_data['status_message']}")
            return {
                'keyword': keyword,
                'Indexed?': 'Error - API Issue'
            }
    except Exception as e:
        print(f"Error processing keyword: {keyword}. Error: {str(e)}")
        return {
            'keyword': keyword,
            'Indexed?': 'Error - Processing Issue'
        }

def main():
    chunks = getIndexInfo(url_to_test)
    #print(chunks)
    print(len(chunks))
    results = []
    for chunk in chunks[20:30]:
        print(chunk)
        result = getSERPInfo(chunk, url_to_test)
        if result:
            results.append(result)

    # Write results to CSV
    with open('serp_results.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['keyword', 'Indexed?']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Results have been written to serp_results.csv")

if __name__ == "__main__":
    asyncio.run(main())