import os
import logging
import requests
import json
import streamlit as st
import pandas as pd
import base64

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

def get_amazon_reviews(amazon_url, product_name):
    try:
        response = requests.get(f"https://data.unwrangle.com/api/getter?platform=amazon_reviews&url={amazon_url}&page=1&api_key={unwrangle_api_key}")
        response.raise_for_status()
        data = response.json()

        reviews = data.get('reviews', [])
        scores = []
        total_sentiment = 0
        total_reviews = 0

        for review in reviews:
            review_text = review['review_text']

            prompt = [
                {"role": "system", "content": f"You are an AI assistant for helping write product reviews for a review website. You will be given context on the product in the form of user reviews from amazon to help formulate the review. Summarize the sentiment around {product_name}"},
                {"role": "assistant", "content": review_text},
                {"role": "user", "content": f"Base your review on the provided context. Format your response like this: {product_name} Overall Sentiment: \n What People Love\n What People Don't Like\n What People are Saying:"},
            ]

            payload = {
                "model" :'gpt-3.5-turbo-16k',
                "messages" : prompt,
                "temperature": 0,
                "max_tokens" : 600 
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            score = data["choices"][0]["message"]["content"]
            if isinstance(score, dict): 
                score['original_review'] = review_text  
                sentiment = float(score['sentiment'])  
                total_sentiment += sentiment  
                total_reviews += 1  
                score['amazon_url'] = amazon_url
                scores.append(score)
        
        average_sentiment = total_sentiment / total_reviews if total_reviews > 0 else 0  
        print("Average sentiment score: ", average_sentiment)
        print(scores)
        
        return scores, average_sentiment

    except Exception as e:
        logging.error(f"An error occurred while processing URL {amazon_url}: {e}")
        return [], 0

def main():
    st.set_page_config(page_title="Amazon Sentiment", page_icon="📈")
    st.title('Amazon Review Sentiment Analyzer')
    amazon_url = st.text_input("Enter Amazon URL", "")
    product_name = st.text_input("Enter Product Name", "")
    if st.button('Analyze Reviews'):
        scores, average_sentiment = get_amazon_reviews(amazon_url, product_name)
        
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
        
if __name__ == "__main__":
    main()