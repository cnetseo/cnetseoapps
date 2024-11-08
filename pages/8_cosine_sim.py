import streamlit as st
import openai
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import io

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')

openai_api_key = st.secrets['openai']['openai_api_key']

# Initialize the OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=openai_api_key)

@st.cache_data
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    client = get_openai_client()
    return client.embeddings.create(input=[text], model=model).data[0].embedding

@st.cache_data
def get_wayback_content(url, timestamp):
    wayback_url = f"http://web.archive.org/web/{timestamp}/{url}"
    response = requests.get(wayback_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('body')
        if main_content:
            return ' '.join(p.get_text() for p in main_content.find_all('p'))
    return None

@st.cache_data
def parse_date(date_string):
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_string}")

@st.cache_data
def get_wayback_content_cached(url, timestamp):
    return get_wayback_content(url, timestamp)

def calculate_change_score(embeddings_a, embeddings_b):
    avg_embedding_a = np.mean(embeddings_a, axis=0)
    print(avg_embedding_a)
    avg_embedding_b = np.mean(embeddings_b, axis=0)
    print(avg_embedding_b)
    similarity = cosine_similarity([avg_embedding_a], [avg_embedding_b])[0][0]
    print(similarity)
    change_score = round((1 - similarity) * 100, 2)
    return change_score

def detect_content_changes(content_a, content_b, similarity_threshold=0.8, progress_bar=None):
    sentences_a = sent_tokenize(content_a)
    sentences_b = sent_tokenize(content_b)

    total_sentences = len(sentences_a) + len(sentences_b) 
    processed_sentences = 0

    embeddings_a = []
    for sentence in sentences_a:
        embeddings_a.append(get_embedding(sentence))
        processed_sentences += 1
        if progress_bar:
            progress_bar.progress(processed_sentences / total_sentences)

    embeddings_b = []
    for sentence in sentences_b:
        embeddings_b.append(get_embedding(sentence))
        processed_sentences += 1
        if progress_bar:
            progress_bar.progress(processed_sentences / total_sentences)

    embeddings_a_np = np.array(embeddings_a)
    embeddings_b_np = np.array(embeddings_b)
    change_score = calculate_change_score(embeddings_a_np, embeddings_b_np)
    
    return change_score

def process_url(url, date1, date2, progress_bar=None):
    """Process a single URL and return its change score."""
    try:
        timestamp1 = date1.strftime("%Y%m%d")
        timestamp2 = date2.strftime("%Y%m%d")
        
        content_a = get_wayback_content_cached(url, timestamp1)
        time.sleep(1)  # Be nice to the Wayback Machine API
        content_b = get_wayback_content_cached(url, timestamp2)
        
        if content_a is None or content_b is None:
            return None
            
        change_score = detect_content_changes(content_a, content_b, progress_bar=progress_bar)
        return change_score
        
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")
        return None

def main():
    st.title("Wayback Machine Content Comparison")
    
    # Add file upload widget
    uploaded_file = st.file_uploader("Upload CSV file with URLs (single column with header 'url')", type=['csv'])
    
    col1, col2 = st.columns(2)
    with col1:
        date1 = st.date_input("Select first date:")
    with col2:
        date2 = st.date_input("Select second date:")
    
    # Single URL input remains as alternative
    url_input = st.text_input("Or enter a single URL:", "")
    
    if st.button("Compare Content"):
        download_nltk_data()
        
        if uploaded_file is not None:
            # Process CSV file
            df = pd.read_csv(uploaded_file)
            
            if 'url' not in df.columns:
                st.error("CSV must contain a column named 'url'")
                return
                
            results = []
            progress_text = "Processing URLs..."
            my_bar = st.progress(0, text=progress_text)
            
            for index, row in df.iterrows():
                url = row['url']
                st.write(f"Processing {url}...")
                change_score = process_url(url, date1, date2, my_bar)
                
                results.append({
                    'url': url,
                    'date1': date1.strftime("%Y-%m-%d"),
                    'date2': date2.strftime("%Y-%m-%d"),
                    'change_score': change_score
                })
                
                my_bar.progress((index + 1) / len(df))
            
            # Create results DataFrame and download button
            results_df = pd.DataFrame(results)
            
            # Create download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="wayback_comparison_results.csv",
                mime="text/csv"
            )
            
            # Display results in the app
            st.subheader("Results:")
            st.dataframe(results_df)
            
        elif url_input:
            # Process single URL
            st.write(f"Processing single URL: {url_input}")
            progress_bar = st.progress(0)
            change_score = process_url(url_input, date1, date2, progress_bar)
            
            if change_score is not None:
                st.subheader(f"Change Score: {change_score}")
                st.write("0 means no change, 100 means completely different.")
            else:
                st.error("Failed to process URL. Please check the URL and dates.")
                
        else:
            st.warning("Please either upload a CSV file or enter a URL.")

if __name__ == "__main__":
    main()