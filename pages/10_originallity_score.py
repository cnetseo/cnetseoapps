import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import json
import tiktoken


api_key = st.secrets["dataforseoapikey"]["api_key"]
# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets['openai']['openai_api_key'])

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text, chunk_size=150):
    """Split text into chunks of approximately chunk_size tokens"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

@st.cache_data
def get_embedding(text, model="text-embedding-3-small"):
    """Get OpenAI embedding for text"""
    text = text.replace("\n", " ")
    client = get_openai_client()
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def check_relevancy(chunk_embedding, query_embedding, threshold=0.85):
    """Check if chunk is relevant to search query"""
    similarity = cosine_similarity([chunk_embedding], [query_embedding])[0][0]
    return similarity > threshold

def verify_information_gain(your_chunk, competitor_chunk, keyword):
    """Use LLM to verify true information gain"""
    client = get_openai_client()
    
    prompt = f"""Compare these two text chunks and identify:
1. Unique facts in chunk 1 not present in chunk 2
2. Verifiable facts (with specific data, studies, or citations)
3. Facts relevant to the keyword: {keyword}
4. Specific metrics/measurements
5. Novel insights or analysis

Chunk 1:
{your_chunk}

Chunk 2:
{competitor_chunk}

Output as JSON:
{{
    "unique_facts": ["fact1", "fact2"],
    "verifiable_facts": ["fact1", "fact2"],
    "relevant_facts": ["fact1", "fact2"],
    "specific_metrics": ["metric1", "metric2"],
    "novel_insights": ["insight1", "insight2"]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    except:
        return None

def calculate_chunk_gain_score(llm_analysis):
    """Calculate information gain score based on LLM analysis"""
    if not llm_analysis:
        return 0
    
    weights = {
        'unique_facts': 2.0,
        'verifiable_facts': 1.5,
        'relevant_facts': 1.0,
        'specific_metrics': 1.5,
        'novel_insights': 1.0
    }
    
    total_score = 0
    max_score = sum(weights.values()) * 5  # Assuming max 5 items per category
    
    for category, weight in weights.items():
        items = llm_analysis.get(category, [])
        total_score += len(items) * weight
    
    normalized_score = min(1.0, total_score / max_score)
    return normalized_score

def get_serp_results(keyword, location_code=2840):
    """Fetch SERP results using DataForSEO API"""
    try:
        url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
        headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }
        payload = f"[{{\"keyword\":\"{keyword}\", \"location_code\":{location_code}, \"language_code\":\"en\", \"device\":\"desktop\", \"os\":\"windows\", \"depth\":100}}]"
        
        response = requests.post(url, headers=headers, data=payload)
        data = response.json()
        
        
        if data["status_code"] == 20000:
            print("Starting to fetch SERP results")
            for result in data["tasks"][0]["result"]:
                #print(result['items'])
                organic_results = []
                count = 0
            for item in result['items']:
                if item.get('type') == 'organic':
                    organic_results.append({
                        'url': item.get('url'),
                        'title': item.get('title', '')
                    })
                    count += 1
                    
                    if count == 10:
                        break
            
            return organic_results
        else:
            st.error(f"API Error: {data['status_code']} - {data.get('status_message', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Error fetching SERP results: {str(e)}")
        return []


def get_page_content(url, timeout=30):
    """Fetch and extract main content and title from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = ' '.join(p.get_text().strip() for p in main_content.find_all('p') if p.get_text().strip())
            
            if not text:
                st.warning(f"No paragraph content found for {url}")
                # Try getting all text as fallback
                text = ' '.join(main_content.stripped_strings)
            
            st.write(f"Retrieved {len(text)} characters from {url}")
            
            if len(text) < 100:
                st.warning(f"Very short content ({len(text)} chars) from {url}")
            
            return {'title': title, 'content': text}
        st.error(f"No main content found for {url}")
        return None
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

def analyze_content(target_content, competitor_contents, keyword):
    """Analyze content for information gain"""
    # Get query embedding
    query_embedding = get_embedding(keyword)
    
    # Check title relevancy first
    title_embedding = get_embedding(target_content['title'])
    relevancy_score = cosine_similarity([title_embedding], [query_embedding])[0][0]
    print(f"Title relevancy score: {relevancy_score:.3f}")
    
    if relevancy_score <= 0.4:  # Title not relevant enough
        st.warning("Content title doesn't appear relevant enough to the search query")
        return {
            'final_score': 0,
            'relevant_chunks': 0,
            'chunks_with_gains': 0,
            'chunk_details': []
        }
    
    # If we get here, title is relevant, so analyze all chunks
    target_chunks = chunk_text(target_content['content'])
    print(f"Total chunks created: {len(target_chunks)}")
    
    chunk_results = []
    chunks_with_gains = 0
    
    # Analyze each chunk
    for i, target_chunk in enumerate(target_chunks):
        print(f"\nAnalyzing chunk {i+1}/{len(target_chunks)}")
        print(f"Chunk length: {len(target_chunk)} characters")
        chunk_embedding = get_embedding(target_chunk)
        chunk_gain = False
        
        # Compare against each competitor
        for comp_idx, comp_content in enumerate(competitor_contents):
            comp_chunks = chunk_text(comp_content['content'])
            print(f"Comparing against competitor {comp_idx+1} ({len(comp_chunks)} chunks)")
            
            for comp_chunk in comp_chunks:
                comp_embedding = get_embedding(comp_chunk)
                similarity = cosine_similarity([chunk_embedding], [comp_embedding])[0][0]
                
                # If similarity is low enough, verify with LLM
                if similarity < 0.3:
                    print(f"Found potentially unique content (similarity: {similarity:.3f})")
                    llm_analysis = verify_information_gain(target_chunk, comp_chunk, keyword)
                    gain_score = calculate_chunk_gain_score(llm_analysis)
                    
                    if gain_score > 0:
                        print(f"✨ Confirmed information gain: {gain_score:.2f}")
                        chunk_gain = True
                        chunk_results.append({
                            'text': target_chunk,
                            'gain_score': gain_score,
                            'analysis': llm_analysis
                        })
                        break
            
            if chunk_gain:
                chunks_with_gains += 1
                break
    
    print(f"\nFinal Statistics:")
    print(f"- Total chunks: {len(target_chunks)}")
    print(f"- Chunks with gains: {chunks_with_gains}")
    
    # Calculate final score based on chunks with gains vs total chunks
    final_score = (chunks_with_gains / len(target_chunks)) * 10 if target_chunks else 0
    
    return {
        'final_score': round(final_score, 2),
        'relevant_chunks': len(target_chunks),  # All chunks considered relevant if title is relevant
        'chunks_with_gains': chunks_with_gains,
        'chunk_details': chunk_results
    }

def main():
    st.title("Content Information Gain Analyzer")
    
    # User inputs
    keyword = st.text_input("Enter search keyword:")
    target_url = st.text_input("Enter your content URL:")
    
    if st.button("Analyze Information Gain") and keyword and target_url:
        with st.spinner("Analyzing content..."):
            # Get SERP results
            serp_results = get_serp_results(keyword)
            
            if not serp_results:
                st.error("Failed to fetch SERP results. Please try again.")
                return
                
            # Fetch target content
            target_content = get_page_content(target_url)
            if not target_content:
                st.error("Failed to fetch target content. Please check the URL.")
                return
            
            st.write(f"Analyzing content: {target_content['title']}")
            
            # Fetch competitor content
            competitor_contents = []
            for result in serp_results:
                content = get_page_content(result['url'])
                if content:
                    competitor_contents.append(content)
                time.sleep(1)  # Rate limiting
            
            # Analyze content
            analysis = analyze_content(target_content, competitor_contents, keyword)
            
            # Display results
            st.header("Analysis Results")
            
            # Display score with color coding
            score = analysis['final_score']
            score_color = "red" if score < 4 else "orange" if score < 7 else "green"
            st.markdown(f"### Information Gain Score: <span style='color:{score_color}'>{score}/10</span>", unsafe_allow_html=True)
            
            # Display statistics
            st.markdown(f"""
            * Total content chunks: {analysis['relevant_chunks']}
            * Chunks with unique information: {analysis['chunks_with_gains']}
            """)
            
            # Display detailed chunk analysis
            st.subheader("Detailed Analysis of Unique Content")
            for chunk_detail in analysis['chunk_details']:
                with st.expander(f"Content Section (Score: {chunk_detail['gain_score']:.2f})"):
                    st.markdown("**Content:**")
                    st.markdown(f"*{chunk_detail['text']}*")
                    
                    st.markdown("**Analysis:**")
                    for category, items in chunk_detail['analysis'].items():
                        if items:
                            st.markdown(f"*{category.replace('_', ' ').title()}:*")
                            for item in items:
                                st.markdown(f"- {item}")
if __name__ == "__main__":
    main()