import requests
import json
import os
import streamlit as st
from serpapi import GoogleSearch
from urllib.parse import urlparse
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langsmith.run_trees import RunTree
from langsmith import Client

serpapikey =  st.secrets['serpapi']['SERPAPIKEY']
openai_api_key = st.secrets['openai']['openai_api_key']
token = st.secrets['screenshot']['screenshot_api_key']
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY= st.secrets['langsmith']['langsmithapi']
LANGCHAIN_PROJECT="content_gaps"
output = "json"
file_type = "jpeg"

#client = Client()

headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
            }

def run_google_search(keyword,user_url):
    response_urls = []
    
    try:
        if serpapikey is None:
            raise ValueError("API Key (serpapikey) is missing")

        params = {
            "q": keyword,
            "api_key": serpapikey
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        if not organic_results:
            raise ValueError("No organic results found")

        urls = [result["link"] for result in organic_results if "reddit.com" not in result["link"] and "youtube.com" not in result["link"]]

        if not urls:
            raise ValueError("No URLs found excluding reddit.com and youtube.com")
        
        urls.insert(0, user_url)
        print(f"Printing the urls to be analyze: {urls}")

        return urls

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def getContent(keyword,user_url):
    response_urls = []
    urls = run_google_search(keyword)
    urls.insert(0, user_url)

    for url in urls:
        url = requests.utils.quote(url)
        keyword = keyword
        query = "https://shot.screenshotapi.net/screenshot"
        query += "?token=%s&url=%s&full_page=true&extract_text=true&output=%s&file_type=%s&wait_for_event=load" % (token, url, output, file_type)
        #print(query)
        response = requests.get(query, stream=True)

        if response.status_code == 200:
            data = json.loads(response.text)
            screenshot_url = data['screenshot']
            text_url = data['extracted_text']
            response_urls.append({'screenshot_url': screenshot_url, 'text_url': text_url})
        else:
            print(f"Request failed with status code {response.status_code}")

    return response_urls

def get_content_dict(response_urls):
    content_dict = {}
    url_domain_names = []

    for data in response_urls:
        if 'text_url' in data:
            url = data['text_url']
            response = requests.get(url)
            if response.status_code == 200:
                text_content = response.text
                # Parse the URL
                parsed_url = urlparse(url)
                # Extract the path
                path = parsed_url.path
                # Split the path
                split_path = path.split('_')
                # Extract and reformat the domain
                domain = '.'.join(split_path[0:3])
                url_domain_names.append(domain)
                # Store the text content and domain in a dictionary
                content_dict[domain] = text_content

    return content_dict

def summarize_page(urls):

    llm = ChatOpenAI(temperature=0)
    content_dict = {}
    url_domain_names = []

    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

     # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    for url in urls:
        print("Scraping {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()

        parsed_url = urlparse(url)
        # Extract the path
        path = parsed_url.path
        # Split the path
        split_path = path.split('_')
        # Extract and reformat the domain
        domain = '.'.join(split_path[0:3])
        url_domain_names.append(domain)

        docs = loader.load()
        summary = stuff_chain.run(docs)
        print("Printing summary for {domain} : {summary}")
        content_dict[domain] = summary
    
    return content_dict



st.cache_data(ttl=3600)
def content_gaps_module(urls,keyword, headers):
    content_dict = summarize_page(urls)
    length = len(content_dict)
    print(f"This dictionary has {length} entries")

    content_response = []

    # Get the first domain and its content
    items_list = list(content_dict.items())
    first_domain, first_url_content = items_list[0]

    for domain, url_content in items_list[1:]:
        payload = {
            "model": "gpt-4-0125-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                                    You are being provided the scrapped text of two web pages: the first is from {first_domain} & the 2nd is from {domain}. I would like you to find the content gaps between the first page and the second page. Return your answer in the following format: 
                                    Content Gap 1:  
                                    (Describe content gap 1)
                                    Content Gap 2:  
                                    (Describe content gap 2)

                                    Rules to follow: 
                                    - Content gaps are defined as key information, facts or concepts that are present in page 2 but absent in page 1
                                    - The main topic of the page is related to  {keyword} so use that to guide what you consider a content gap.
                                    - Use the domain names instead of "page 1" or "page 2" for example 
                                """
                        },
                        {
                            "type": "text",
                            "text": first_url_content
                        },
                        {
                            "type": "text",
                            "text": url_content
                        }
                    ]
                }
            ],
            "max_tokens": 3000
        }

        print(f"processing {domain}")
        airesponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(airesponse)
        # Check if the request was successful
        if airesponse.status_code == 200:
            # Append the content of the response to the list
            content_response.append(airesponse.json()['choices'][0]['message']['content'])
        else:
            print(f"Failed to get response for URL, Status Code: {airesponse.status_code}")

    

    consolidated_content_gaps = "\n".join(content_response)
    print(consolidated_content_gaps)

    prompt_template = """You are being given a series of content gap observations comparing {first_domain} to several other pages. First summarize each of the findings, lumping any similar ones into a common theme. Second, give a name to each content gap. 
    Third, output the content gaps in the schema example shown below:

    -----
    These are the content gaps: 
    {text} 
    -------
    This is the output schema example. Structure the summarized output into this json schema: 
    {format_instructions}
    """

    os.environ["OPENAI_API_KEY"] = st.secrets['openai']["openai_api_key"]
    
    class ContentGap(BaseModel):
        content_gap_name: str = Field(description="name of content gap between page i.e Content Gap 1, Content Gap 2")
        content_gap_description: str = Field(description="description of content gap between pages")

    parser = SimpleJsonOutputParser(pydantic_object=ContentGap)
    
    prompt = PromptTemplate(template = prompt_template,input_variables=["first_domain","text"], partial_variables={"format_instructions": parser.get_format_instructions()},)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    #stuff_chain = PromptTemplate(pattern=prompt_template) | LLMChain(llm=llm) | SimpleJsonOutputParser(pydantic_model=ContentGap)

    stuff_chain = prompt | llm | parser


    result = stuff_chain.invoke({"first_domain":first_domain,"text": consolidated_content_gaps})
    print(result)
    return result

def main():
    st.set_page_config(page_title="Content Gaps Analysis", page_icon="📈")
    st.sidebar.header("Experimental")
    st.title('Content Gaps Analysis')
    user_url = st.text_input("Enter URL", "")
    keyword = st.text_input("Enter Keyword", "")
    if st.button('Find Content Gaps'):  # Ensure both user_url and keyword are entered
        results = run_google_search(keyword,user_url)
        content_gaps = content_gaps_module(results,keyword,headers)
        print(content_gaps)
        for i, text in enumerate(content_gaps):
            st.write(f'Content Gap {i+1}:')
            st.write(text)

if __name__ == "__main__":
    main()
