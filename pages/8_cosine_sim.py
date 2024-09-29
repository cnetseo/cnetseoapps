import streamlit as st
import openai
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from wayback import get_wayback_content, parse_date
import time

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)

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
def get_wayback_content_cached(url, timestamp):
    return get_wayback_content(url, timestamp)

def calculate_change_score(embeddings_a, embeddings_b):
    # Calculate the average embedding for each set of sentences
    avg_embedding_a = np.mean(embeddings_a, axis=0)
    avg_embedding_b = np.mean(embeddings_b, axis=0)
    
    # Calculate cosine similarity between the average embeddings
    similarity = cosine_similarity([avg_embedding_a], [avg_embedding_b])[0][0]
    print(similarity)
    
    # Convert similarity to change score (0 similarity -> 100 score, 1 similarity -> 0 score)
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

    similarity_matrix = cosine_similarity(embeddings_b_np, embeddings_a_np)

    added_content = []
    for i, row in enumerate(similarity_matrix):
        max_similarity = np.max(row)
        if max_similarity < similarity_threshold:
            added_content.append(sentences_b[i])

    removed_content = []
    for i, col in enumerate(similarity_matrix.T):
        max_similarity = np.max(col)
        if max_similarity < similarity_threshold:
            removed_content.append(sentences_a[i])

    # Calculate change score using the sentence embeddings
    change_score = calculate_change_score(embeddings_a_np, embeddings_b_np)

    return added_content, removed_content, change_score

def compare_wayback_content(url, date1_str, date2_str, similarity_threshold=0.8):
    date1 = parse_date(date1_str)
    date2 = parse_date(date2_str)

    timestamp1 = date1.strftime("%Y%m%d")
    timestamp2 = date2.strftime("%Y%m%d")

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    try:
        with st.spinner("Retrieving content for the first date..."):
            content_a = get_wayback_content_cached(url, timestamp1)
        time.sleep(1)  # Be nice to the Wayback Machine API
        with st.spinner("Retrieving content for the second date..."):
            content_b = get_wayback_content_cached(url, timestamp2)

        if content_a is None or content_b is None:
            return "Failed to retrieve content for one or both dates.", None, None, None

        with st.spinner("Analyzing content differences..."):
            added, removed, change_score = detect_content_changes(content_a, content_b, similarity_threshold, my_bar)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None, None, None
    finally:
        my_bar.empty()

    return f"Changes between {date1.date()} and {date2.date()} for {url}", added, removed, change_score

def main():
    st.title("Wayback Machine Content Comparison")

    url = st.text_input("Enter URL to compare:", "https://www.example.com")
    col1, col2 = st.columns(2)
    with col1:
        date1 = st.date_input("Select first date:")
    with col2:
        date2 = st.date_input("Select second date:")

    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.01)

    if st.button("Compare Content"):
        download_nltk_data()  # Download NLTK data only when needed
        result, added, removed, change_score = compare_wayback_content(url, date1.strftime("%Y-%m-%d"), date2.strftime("%Y-%m-%d"), similarity_threshold)
        
        if result:
            st.write(result)
            
            # Display the change score
            st.subheader(f"Overall Change Score: {change_score}")
            st.write(f"0 means no change, 1 means completely different.")
            
            if added:
                st.subheader("Added or Significantly Changed Content:")
                for i, sentence in enumerate(added, 1):
                    st.write(f"{i}. {sentence}")
            else:
                st.write("No significant additions detected.")
            
            if removed:
                st.subheader("Removed Content:")
                for i, sentence in enumerate(removed, 1):
                    st.write(f"{i}. {sentence}")
            else:
                st.write("No significant removals detected.")
        else:
            st.error("Failed to compare content. Please try again.")

if __name__ == "__main__":
    main()