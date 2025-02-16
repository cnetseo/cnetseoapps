import os
import logging
import requests
import json
import streamlit as st
import pandas as pd
import base64
from langchain.chains.openai_functions.base import (
    create_openai_fn_chain,
    create_structured_output_chain,
    create_openai_fn_runnable,
    create_structured_output_runnable,
)
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
openai_api_key = st.secrets["openai"]["openai_api_key"]
unwrangle_api_key = st.secrets["unwrangle"]["uwapi_key"]

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai_api_key}"
}

json_schema = {
    "title": "Sentiment",
    "description": "Identifying the sentiment of Amazon reviews",
    "type": "object",
    "properties": {
        "sentiment": {"title": "Sentiment Score", "description": "How positive, neutral or negative is this statement about the product out of a score of 1-5 with 5 being overwhelmingly positive. Focus on sentiment related to just the product", "type": "string"},
        "reasoning": {"title": "reasoning", "description": "Why you scored the sentiment of this statement", "type": "string"},
        "comment_type": {"title": "comment_type","description": "What type of comment is this? Is it an opinion, factual statement, irrelevant statement"}
    },
    "required": ["sentiment", "reasoning","comment_type"],
}

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


llm = ChatOpenAI(model="gpt-4", temperature=0)

st.cache_data(ttl=3600)
def get_amazon_reviews(amazon_url, product_name):
        try:
            # Get reviews from Unwrangle API
            response = requests.get(f"https://data.unwrangle.com/api/getter?platform=amazon_reviews&url={amazon_url}&page=1&api_key={unwrangle_api_key}")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while trying to get reviews from Unwrangle API: {e}")
            return [], 0

        data = response.json()

        reviews = data.get('reviews', [])
        scores = []
        total_sentiment = 0
        total_reviews = 0
        count = 0

        for review in reviews:
            review_text = review['review_text']
            print(review_text)
            runnable = create_structured_output_runnable(json_schema, llm, prompt)
            score = runnable.invoke({"input": review_text})
            if isinstance(score, dict) and score.get('comment_type') == 'opinion': 
                score['original_comment'] = review_text  
                sentiment = float(score['sentiment'])  # Convert sentiment score to a float
                total_sentiment += sentiment  # Add sentiment score to the total
                total_reviews += 1  # Increment the count of comments
                score['amazon url'] = amazon_url
                scores.append(score)
                count += 1
           

        average_sentiment = total_sentiment / total_reviews if total_reviews > 0 else 0  # Calculate average sentiment
        print("Average sentiment score: ", average_sentiment)
        print(scores)

         # Extract the review_text for each review
        review_texts = [review['review_text'] for review in reviews]
        # Join all the review_texts together with a space in between
        joined_review_texts = ' '.join(review_texts)
       

        prompt_summary = [
            {"role": "system", "content": f"You are an AI assistant for helping write product reviews for a review website. You will be given context on the product in the form of user reviews from amazon to help formulate the review. Summarize the sentiment around {product_name}"},
            {"role": "assistant", "content": joined_review_texts},
            {"role": "user", "content": f"Base your review on the provided context. Format your response like this: {product_name} Overall Sentiment: \n What People Love\n What People Don't Like\n What People are Saying:"},
        ]

        payload = {
            "model" :'gpt-3.5-turbo-16k',
            "messages" : prompt_summary,
            "temperature": 0,
            "max_tokens" : 600 
        }

        try:
            # Get completion from OpenAI API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while trying to get completion from OpenAI API: {e}")
            return [], 0, ""

        data = response.json()
    # Extract answers from response
        answers = data["choices"][0]["message"]["content"]
        return scores, average_sentiment,  answers


def main():
    st.set_page_config(page_title="Amazon Sentiment", page_icon="📈")
    st.title('Amazon Review Sentiment Analyzer')
    amazon_url = st.text_input("Enter Amazon URL", "")
    product_name = st.text_input("Enter Product Name", "")
    if st.button('Analyze Reviews'):
        scores, average_sentiment,summary = get_amazon_reviews(amazon_url, product_name)
        
        # Convert scores to a dataframe
        df = pd.DataFrame(scores)

                # Show only a handful of rows
        st.metric("Amazon Sentiment Score (5 is best)",round(average_sentiment,2))
        st.dataframe(df.head())

        st.success('Done!')
        
        # Download CSV file
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  
        href = f'<a href="data:file/csv;base64,{b64}" download="amazon_scores.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.text_area('Output Text:', value=summary, height=500)
        
if __name__ == "__main__":
    main()