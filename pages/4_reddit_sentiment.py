import praw
import os
from langchain.chains.openai_functions.base import (
    create_openai_fn_chain,
    create_structured_output_chain,
    create_openai_fn_runnable,
    create_structured_output_runnable,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import csv
import streamlit as st
import pandas as pd
import base64
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY'] = st.secrets['openai']["openai_api_key"]
apikey = st.secrets['serpapi']["SERPAPIKEY"]
reddit_client_id=st.secrets['reddit']["reddit_client_id"]
reddit_client_secret=st.secrets['reddit']["reddit_client_secret"]

#client = Client()

from serpapi import GoogleSearch
import re

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators= "\n\n",
    chunk_size = 50,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
)

def run_google_search(query):
    params = {
        "q": query,
        "api_key": apikey
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    urls = [result["link"] for result in organic_results]
    reddit_urls = [url for url in urls if re.search(r'^(http(s)?:\/\/)?(www\.)?reddit\.com', url)]
    return reddit_urls
        
reddit = praw.Reddit(
   client_id=reddit_client_id,
   client_secret=reddit_client_secret,
   user_agent="User-Agent: android:com.example.myredditapp:v1.2.3 (by /u/ccasazza)"
)
st.set_page_config(page_title="Reddit Sentiment Beta", page_icon="📈")
st.sidebar.header("Reddit Sentiment Beta")
st.title('Reddit Sentiment Analysis')

search_term = st.text_input('Enter the search term')
search_term = search_term + " reddit"
num_urls = st.number_input('Enter the number of reddit threads to explore', min_value=1, max_value=10, value=3)

print(search_term)
print("Running")

json_schema = {
    "title": "Sentiment",
    "description": "Identifying the sentiment of reddit threads",
    "type": "object",
    "properties": {
        "sentiment": {"title": "Sentiment Score", "description": "How positive, neutral or negative is this statement about the {search_term} out of a score of 1-5 with 5 being overwhelmingly positive. Focus on sentiment related to just the {search_term}", "type": "string"},
        "reasoning": {"title": "reasoning", "description": "Why you scored the sentiment of this statement", "type": "string"},
        "comment_type": {"title": "comment_type","description": "What type of comment is this? Is it an opinion, factual statement, irrelevant statement"}
    },
    "required": ["sentiment", "reasoning","comment_type"],
}

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world class algorithm for extracting information in structured formats about the sentiment around certain products.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {input}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

def chunk_by_tokens(text, tokens_count):
    words = text.split(' ')
    chunks = [' '.join(words[i:i + tokens_count]) for i in range(0, len(words), tokens_count)]
    return chunks

# Define prompt & LLM chain
prompt_template = """You are being given comments from a reddit thread about a {product}:
    "{text}"
    Identify what people really like about the product, what they criticize about the product and where the product's value lies:"""



prompt_stuffchain = PromptTemplate(input_variables=["text","product"],template=prompt_template)


#prompt_stuffchain = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")
llm_chain = LLMChain(llm=llm, prompt=prompt_stuffchain)
# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# Search term for Google


def run_analysis():
    with st.spinner('Running analysis...'):
        print("run_analysis function has started")
        reddit_urls = run_google_search(search_term)
        scores = []
        total_sentiment = 0  # Initialize the total sentiment score
        total_comments = 0  # Initialize the count of comments
        for url in reddit_urls[:num_urls]:
            if "/comments/" in url:  # Check if it's a thread link
                try:
                    submission = reddit.submission(url=url)
                    submission.comments.replace_more(limit=None)
                    count = 0
                    print(submission.comments.list())
                    for comment in submission.comments.list():
                        if count >= 4:
                            break
                        if comment.score > 2 and len(comment.body) > 200:
                            runnable = create_structured_output_runnable(json_schema, llm, prompt)
                            score = runnable.invoke({"input": comment.body})
                            if isinstance(score, dict) and score.get('comment_type') == 'opinion': 
                                score['original_comment'] = comment.body  
                                sentiment = float(score['sentiment'])  # Convert sentiment score to a float
                                total_sentiment += sentiment  # Add sentiment score to the total
                                total_comments += 1  # Increment the count of comments
                                score['reddit url'] = url
                                scores.append(score)
                        count += 1
                except Exception as e:
                    st.error("Error processing URL {}".format(url))
                    st.error("Error message: {}".format(str(e)))
            else:
                st.warning("The URL {} is a subreddit link, not a thread link, and will be skipped.".format(url))

        average_sentiment = total_sentiment / total_comments if total_comments > 0 else 0  # Calculate average sentiment
        print("Average sentiment score: ", average_sentiment)
        print(scores)
       

    st.success('Analysis completed.')
   
    # Convert scores to a dataframe
    df = pd.DataFrame(scores)

    # Show only a handful of rows
    st.metric("Reddit Sentiment Score (5 is best)",round(average_sentiment,2))
    st.dataframe(df.head())

    st.success('Done!')
   
    
    # Download CSV file
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="reddit_scores.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    scores_text = "\n\n".join(score['original_comment'] for score in scores)
    #encoded_text = scores_text.encode("utf-8")
    #print(encoded_text)
    #encoded_text = str(encoded_text)
    with open("output.txt", "w") as file:
        file.write(scores_text)
    
    with open("output.txt", "r") as file:
        content = file.read()

    print(content)
    texts = text_splitter.create_documents(content)
    #print(texts)

    output_text = stuff_chain.invoke({"input_documents": texts,"product":search_term})

    text_to_display = output_text['output_text']

    st.text_area('Output Text:', value=text_to_display, height=500)

if st.button('Run Analysis'):
    run_analysis()



