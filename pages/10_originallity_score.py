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
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

class ContentAnalysis(BaseModel):
    unique_facts: List[str]
    verifiable_facts: List[str]
    relevant_facts: List[str]
    specific_metrics: List[str]
    novel_insights: List[str]
    expert_insights: List[str]



api_key = st.secrets["dataforseoapikey"]["api_key"]


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets['openai']['openai_api_key'])

@st.cache_resource
def get_google_client():    
    return genai.Client(api_key=st.secrets["google"]["google_api"])

client = get_google_client()

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text, chunk_size=500):
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
def get_embedding(text, model="text-embedding-004"):
    text = text.replace("\n", " ")
    result = client.models.embed_content(
    model=model,
    contents=text)
    return result.embeddings[0].values

def check_relevancy(chunk_embedding, query_embedding, threshold=0.85):
    """Check if chunk is relevant to search query"""
    similarity = cosine_similarity([chunk_embedding], [query_embedding])[0][0]
    return similarity > threshold

def verify_information_gain(your_chunk, competitor_chunk, keyword, use_google=False):
    """Use LLM to verify true information gain"""
    
    prompt = f"""
    You are a content analyst determining the level of 'information gain' compared to competitor content i.e. the amount of novel information. 

    Compare these two text chunks and identify:
    1. Unique facts in chunk 1 not present in chunk 2 that are not just an opinion or general knowledge
    2. Verifiable facts (with specific data, studies, or citations) in chunk 1 not present in chunk 2
    3. Facts relevant to the keyword: {keyword} in chunk 1 not present in chunk 2
    4. Specific metrics/measurements in chunk 1 not present in chunk 2
    5. Novel insights or analysis in chunk 1 not present in chunk 2
    6. Hands on-testing or insights that can only be derived by personal experience or expertise in chunk 1 but not present in chunk 2

    Only highlight unique information that is not present in the competitor's content.

    Chunk 1:
    {your_chunk}

    Chunk 2:
    {competitor_chunk}
    """

    if use_google:
        try:
            print("Starting Gemini analysis...")
            client = get_google_client()
            response = client.models.generate_content(
                model="gemini-2.5-pro-preview-03-25",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ContentAnalysis,
                }
            )
            
            # Get parsed response
            analysis: ContentAnalysis = response.parsed
            return analysis.model_dump()
        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
            return None
    else:
        try:
            print("Starting OpenAI analysis...")
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            print(f"Error in OpenAI analysis: {str(e)}")
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
        'novel_insights': 1.0,
        'expert_insights': 1.0
    }
    
    total_score = 0
    max_score = sum(weights.values()) * 5  # Assuming max 5 items per category
    
    for category, weight in weights.items():
        items = llm_analysis.get(category, [])
        total_score += len(items) * weight
    
    normalized_score = min(1.0, total_score / max_score)
    return normalized_score

def get_serp_results(keyword, target_url, location_code=2840):
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
            organic_results = []
            count = 0
            items = data["tasks"][0]["result"][0]["items"]
            for item in items:
                if item.get('type') == 'organic' and item.get('url') != target_url:
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
            
            print(f"Retrieved {len(text)} characters from {url}")
            
            if len(text) < 100:
                st.warning(f"Very short content ({len(text)} chars) from {url}")
            
            return {'title': title, 'content': text}
        st.error(f"No main content found for {url}")
        return None
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

def analyze_content(target_content, competitor_contents, keyword,use_google=False):
    """Analyze content for information gain"""
    query_embedding = get_embedding(keyword)
    
    # Check title relevancy
    title_embedding = get_embedding(target_content['title'])
    print(title_embedding)
    relevancy_score = cosine_similarity([title_embedding], [query_embedding])[0][0]
    print(f"Title relevancy score: {relevancy_score:.3f}")
    
    if relevancy_score <= 0.4:
        st.warning("Content title doesn't appear relevant enough to the search query")
        return {
            'final_score': 0,
            'target_gain': 0,
            'competitor_gains': [],
            'chunk_details': []
        }

    def calculate_page_gain(page_content, all_other_content):
        """Calculate gain against concatenated competitor content"""
        total_gain = 0
        chunks = chunk_text(page_content['content'])
        chunk_details = []
        
        print(f"Analyzing content: {page_content['title'][:50]}...")
        
        for chunk in chunks:
            # Only run LLM analysis once per chunk against all content
            llm_analysis = verify_information_gain(chunk, all_other_content, keyword,use_google=use_google)
            chunk_gain = calculate_chunk_gain_score(llm_analysis)
            
            if chunk_gain > 0:
                total_gain += chunk_gain
                chunk_details.append({
                    'text': chunk,
                    'gain_score': chunk_gain,
                    'analysis': llm_analysis
                })
        
        return total_gain, chunk_details

    # Calculate target gain against all competitor content combined
    print("Analyzing target content...")
    all_competitor_content = " ".join([comp['content'] for comp in competitor_contents])
    target_gain, target_details = calculate_page_gain(target_content, all_competitor_content)
    
    # Calculate each competitor's gain against other competitors
    competitor_gains = []
    print("Analyzing competitor content...")
    for i, comp in enumerate(competitor_contents):
        # Combine all other competitor content except current one
        other_comps_content = " ".join(c['content'] for j, c in enumerate(competitor_contents) if j != i)
        comp_gain, _ = calculate_page_gain(comp, other_comps_content)
        competitor_gains.append(comp_gain)
        print(f"Competitor {i+1} gain score: {comp_gain:.2f}")
    
    # Calculate relative score
    avg_competitor_gain = np.mean(competitor_gains) if competitor_gains else 0
    std_competitor_gain = np.std(competitor_gains) if competitor_gains else 1
    
    # Convert to 0-10 score using standard deviations from mean
    if avg_competitor_gain == 0:
        final_score = 5 if target_gain > 0 else 0
    else:
        z_score = (target_gain - avg_competitor_gain) / std_competitor_gain
        final_score = min(10, max(0, 5 + (z_score * 2)))
    
    # Create rankings
    all_scores = [{'url': target_content['title'], 'gain': target_gain, 'is_target': True}]
    all_scores.extend([
        {'url': comp['title'], 'gain': gain, 'is_target': False}
        for comp, gain in zip(competitor_contents, competitor_gains)
    ])
    
    # Sort by gain score
    ranked_pages = sorted(all_scores, key=lambda x: x['gain'], reverse=True)
    
    return {
        'final_score': round(final_score, 2),
        'target_gain': target_gain,
        'competitor_gains': competitor_gains,
        'avg_competitor_gain': avg_competitor_gain,
        'std_competitor_gain': std_competitor_gain,
        'chunk_details': target_details,
        'rankings': ranked_pages,
        'competitor_analyses': {  # New field
            comp['title']: {
                'gain': gain,
                'analysis': verify_information_gain(comp['content'], target_content['content'], keyword,use_google=use_google)
            }
            for comp, gain in zip(competitor_contents, competitor_gains)
        }
    }

def show_competitor_analysis(competitor_score, competitor_details):
    st.markdown("### Why This Competitor Ranks Higher")
    
    st.markdown(f"""
    This competitor achieved an Information Gain score of {competitor_score} by providing unique information in these areas:
    """)
    
    # Group unique information by type
    unique_facts_count = len(competitor_details.get('unique_facts', []))
    verifiable_facts_count = len(competitor_details.get('verifiable_facts', []))
    metrics_count = len(competitor_details.get('specific_metrics', []))
    insights_count = len(competitor_details.get('novel_insights', []))
    
    # Show summary first
    st.markdown(f"""
    - {unique_facts_count} unique facts not found in other content
    - {verifiable_facts_count} verifiable facts with data/citations
    - {metrics_count} specific measurements or data points
    - {insights_count} novel insights or analysis
    """)
    
    # Show detailed examples
    with st.expander("See Examples of Unique Information"):
        for category, items in competitor_details.items():
            if items:
                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                for item in items:
                    st.markdown(f"- {item}")
        
        st.info("💡 Adding similar types of information to your content could improve your Information Gain score.")

def summarize_improvement_opportunities(target_content, higher_ranked_competitors, keyword, use_google=False):
    """Generate a comprehensive improvement summary based on all higher-ranking content"""
    
    # Combine all higher-ranking content for analysis
    all_better_content = "\n\n".join([comp['content'] for comp in higher_ranked_competitors])
    
    prompt = f"""
    As a content strategist, analyze how the target content could be improved based on competitors that achieved higher information gain scores.
    
    Target Content Topic: {keyword}
    
    Higher Ranking Content:
    {all_better_content}

    Target Content:
    {target_content}

    Provide a strategic summary that includes:
    1. Key themes or types of information present in higher-ranking content but missing from target
    2. Specific examples of how competitors are delivering more unique information
    3. Actionable suggestions for improving information gain (be specific but don't directly copy competitor content)
    4. Additional research angles or data points that could be explored
    5. Content structure or presentation insights

    Focus on concrete, actionable insights rather than general advice.
    """

    if use_google:
        client = get_google_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    else:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

def main():
    st.title("Content Information Gain Analyzer")
    
    # Add model selection
    model_choice = st.radio(
        "Select AI Model:",
        ["OpenAI GPT-4", "Google Gemini"],
        help="Choose which AI model to use for analysis"
    )
    
    # User inputs
    keyword = st.text_input("Enter search keyword:")
    target_url = st.text_input("Enter your content URL:")

    if st.button("Analyze Information Gain") and keyword and target_url:
        use_google = model_choice == "Google Gemini"
        with st.spinner("Analyzing content..."):
            # Get SERP results
            serp_results = get_serp_results(keyword,target_url)
            
            if not serp_results:
                st.error("Failed to fetch SERP results. Please try again.")
                return
                
            # Fetch target content
            target_content = get_page_content(target_url)
            if not target_content:
                st.error("Failed to fetch target content. Please check the URL.")
                return
            
            print(f"Analyzing content: {target_content['title']}")
            
            # Fetch competitor content
            competitor_contents = []
            for result in serp_results:
                content = get_page_content(result['url'])
                if content:
                    competitor_contents.append(content)
                time.sleep(1)  # Rate limiting
            
            # Analyze content
            analysis = analyze_content(target_content, competitor_contents, keyword,use_google=use_google)
            
           # In main():
            # Display results
            st.header("Analysis Results")
            
            score = analysis['final_score']
            score_color = "red" if score < 4 else "orange" if score < 7 else "green"
            st.markdown(f"### Information Gain Score: <span style='color:{score_color}'>{score}/10</span>", unsafe_allow_html=True)
            
            # Show comparison stats
            st.markdown("### Comparison with Competitors")
            st.markdown(f"""
            * Your content's information gain: {analysis['target_gain']:.2f}
            * Average competitor gain: {analysis['avg_competitor_gain']:.2f}
            * Standard deviation: {analysis['std_competitor_gain']:.2f}
            """)
            
            st.markdown("### Content Rankings by Information Gain")

            # Create DataFrame for rankings
            rankings_df = pd.DataFrame({
                'Rank': range(1, len(analysis['rankings']) + 1),
                'Content': [r['url'] for r in analysis['rankings']],
                'Type': ['Your Content' if r['is_target'] else 'Competitor' for r in analysis['rankings']],
                'Information Gain': [round(r['gain'], 2) for r in analysis['rankings']]
            })

            # Add emoji or marker to make your content stand out
            rankings_df['Type'] = rankings_df['Type'].map({
                'Your Content': '🎯 Your Content',
                'Competitor': 'Competitor'
            })

            # Display the rankings
            st.dataframe(
                rankings_df.set_index('Rank'),
                use_container_width=True
            )

            # After displaying the rankings table
            rankings_df = rankings_df.reset_index()  # Make sure Rank is a column
            your_content_rank = rankings_df[rankings_df['Type'] == '🎯 Your Content']['Rank'].iloc[0]

            if your_content_rank > 1:
                st.markdown("## Improvement Opportunities")
                
                # Get all higher-ranking competitors
                better_competitors = [
                    competitor_contents[int(r['Rank'])-1] 
                    for r in rankings_df[rankings_df['Rank'] < your_content_rank].to_dict('records')
                ]
                
                # Get strategic summary
                improvement_summary = summarize_improvement_opportunities(
                    target_content['content'],
                    better_competitors,
                    keyword,
                    use_google=use_google
                )
                
                st.markdown("### Strategic Analysis")
                st.markdown(improvement_summary)
                
                # Show individual competitor breakdowns in expanders
                st.markdown("### Detailed Competitor Analysis")
                for i, comp in enumerate(better_competitors):
                    with st.text_area(f"Competitor #{i+1}: {comp['title']}"):
                        if comp['title'] in analysis['competitor_analyses']:
                            comp_analysis = analysis['competitor_analyses'][comp['title']]
                            show_competitor_analysis(comp_analysis['gain'], comp_analysis['analysis'])


if __name__ == "__main__":
    main()