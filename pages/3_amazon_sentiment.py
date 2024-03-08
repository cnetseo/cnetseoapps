import os
import logging
import requests
import json
import streamlit as st

openai_api_key = st.secrets["openai"]["openai_api_key"]
unwrangle_api_key = st.secrets["unwrangle"]["uwapi_key"]


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai_api_key}"
}

def get_amazon_reviews(amazon_url, product_name):
    try:
        response = requests.get(f"https://data.unwrangle.com/api/getter?platform=amazon_reviews&url={amazon_url}&page=1&api_key={unwrangle_api_key}")
        response.raise_for_status()
        data = response.json()

        reviews = data.get('reviews', [])
        review_texts = [review['review_text'] for review in reviews]
        joined_review_texts = ' '.join(review_texts)

        prompt = [
            {"role": "system", "content": f"You are an AI assistant for helping write product reviews for a review website. You will be given context on the product in the form of user reviews from amazon to help formulate the review. Summarize the sentiment around {product_name}"},
            {"role": "assistant", "content": joined_review_texts},
            {"role": "user", "content": f"Base your review on the provided context. Format your response like this: {product_name} Overall Sentiment: \n What People Love\n What People Don't Like\n What People are Saying:"},
        ]

        payload = {
            "model" :'gpt-3.5-turbo-16k',
            "messages" : prompt,
            "temperature": 0,
            "max_tokens" : 600 }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        answers = data["choices"][0]["message"]["content"]

        return json.dumps({"reviews": answers})

    except Exception as e:
        logging.error(f"An error occurred while processing URL {amazon_url}: {e}")
        return json.dumps({"error": "Failed to process any reviews"}), 500

def main():
    st.set_page_config(page_title="Amazon Sentiment", page_icon="📈")
    st.title('Amazon Review Sentiment Analyzer')
    amazon_url = st.text_input("Enter Amazon URL", "")
    product_name = st.text_input("Enter Product Name", "")
    if st.button('Analyze Reviews'):
        result = get_amazon_reviews(amazon_url, product_name)
        st.write(result)
if __name__ == "__main__":
    main()