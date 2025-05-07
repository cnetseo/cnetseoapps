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
import firebase_admin
from firebase_admin import credentials, db  # Changed from firestore to db
import datetime
import uuid

# Keep the existing ContentAnalysis class
class ContentAnalysis(BaseModel):
    unique_facts: List[str]
    verifiable_facts: List[str]
    relevant_facts: List[str]
    specific_metrics: List[str]
    novel_insights: List[str]
    expert_insights: List[str]

# Firebase initialization
def initialize_firebase():
    """Initialize Firebase with properly fixed database reference"""
    try:
        # Check if app is already initialized and delete it if it exists
        try:
            app = firebase_admin.get_app()
            print(f"Found existing Firebase app, deleting it: {app.name}")
            firebase_admin.delete_app(app)
            print("Existing app deleted")
        except ValueError:
            print("No existing Firebase app found")
        
        # Now initialize a fresh app with explicit database URL
        print("Initializing new Firebase app")
        
        # Get Firebase credentials from Streamlit secrets
        firebase_config = dict(st.secrets["firebase"])
        
        # Fix private key format if needed
        if "private_key" in firebase_config:
            key = firebase_config["private_key"]
            if '\\n' in key and '\n' not in key:
                firebase_config["private_key"] = key.replace('\\n', '\n')
        
        # Create credential certificate
        cred = credentials.Certificate(firebase_config)
        
        # Use explicit hardcoded database URL
        database_url = 'https://originalitycalc-default-rtdb.firebaseio.com/'
        print(f"Using database URL: {database_url}")
        
        # Initialize app with options object containing the URL
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url
        })
        print("New Firebase app initialized successfully")
        
        # Get and verify database reference
        db_ref = db.reference()
        print("Successfully got database reference")
        
        # Verify connection works by writing a test value
        try:
            # Simple write test - set then update
            test_ref = db_ref.child("test").child("connection")
            test_ref.set({"timestamp": {".sv": "timestamp"}})
            # Instead of remove(), use update to clear the value
            test_ref.update({"test_complete": True})
            print("Verified database connection with test write")
        except Exception as test_e:
            print(f"Warning: Database connection test failed: {str(test_e)}")
        
        return db_ref
    
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}")
        st.error(f"Firebase initialization error: {str(e)}")
        return None

# Enhanced user authentication
def authenticate_user(username, password):
    """Authenticate user for database access"""
    # You could use a more secure approach in production
    valid_credentials = {
        "admin": "CNETSEO2025",  # Change this to a strong password
              # Add additional users as needed
    }
    
    if username in valid_credentials and password == valid_credentials[username]:
        return username
    return None

# Add this function to check if user is authenticated
def require_auth():
    """Require authentication for accessing protected features"""
    if not st.session_state.get("authenticated_user"):
        st.warning("⚠️ Authentication required to access this feature")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = authenticate_user(username, password)
                if user:
                    st.session_state["authenticated_user"] = user
                    st.success(f"Welcome, {user}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
        
        return False
    return True

# Save analysis results to Realtime Database
def save_analysis_with_summary(keyword, target_url, analysis_results, improvement_summary=None, user_id=None):

    """Save analysis results with improvement summary"""
    try:
        # Get database reference
        db_ref = initialize_firebase()
        
        # If database reference is None, return early
        if db_ref is None:
            st.error("Could not connect to database. Analysis will not be saved.")
            return None
        
        
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create a streamlined data structure with just the essential information
        analysis_data = {
            "keyword": keyword,
            "target_url": target_url,
            "score": float(analysis_results["final_score"]),
            "timestamp": {".sv": "timestamp"},
        }

        # Add user ID to the analysis data
        if user_id:
            analysis_data["user_id"] = user_id
        
        
        # Add rankings data
        if "rankings" in analysis_results and analysis_results["rankings"]:
            # Convert rankings to simple array for storage
            rankings = []
            for r in analysis_results["rankings"]:
                rankings.append({
                    "url": r["url"],
                    "gain": float(r["gain"]),
                    "is_target": bool(r["is_target"])
                })
            analysis_data["rankings"] = rankings
        
        # Add improvement summary if provided
        if improvement_summary:
            analysis_data["improvement_summary"] = improvement_summary
        
        # Try to save the data
        try:
            print(f"Saving analysis with ID: {analysis_id}")
            db_ref.child("content_analyses").child(analysis_id).set(analysis_data)
            print("Successfully saved analysis data")
            
            return analysis_id
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")
            st.error(f"Failed to save analysis: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error preparing analysis data: {str(e)}")
        print(f"Firebase error: {str(e)}")
        return None

# Get past analyses from Realtime Database
def get_past_analyses(user_id=None, limit=20):
    """Retrieve past analyses for a specific user"""
    db_ref = initialize_firebase()
    
    # If database reference is None, return early
    if db_ref is None:
        st.error("Could not connect to database.")
        return []
    
    analyses = []
    try:
        # Get all analyses
        all_analyses = db_ref.child("content_analyses").get()
        
        if all_analyses:
            # Convert to list and filter by user_id if specified
            for analysis_id, analysis in all_analyses.items():
                # Only include analyses for the authenticated user
                if not user_id or analysis.get("user_id") == user_id:
                    analysis["id"] = analysis_id
                    analyses.append(analysis)
            
            # Sort by timestamp (descending)
            analyses.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Limit results
            analyses = analyses[:limit]
    except Exception as e:
        st.error(f"Error fetching analyses: {str(e)}")
    
    return analyses

# Get a single analysis by ID
def get_analysis_by_id(analysis_id):
    """Retrieve a specific analysis by ID from Realtime Database"""
    db_ref = initialize_firebase()
    
    try:
        analysis = db_ref.child("content_analyses").child(analysis_id).get()
        if analysis:
            analysis["id"] = analysis_id
            return analysis
    except Exception as e:
        st.error(f"Error fetching analysis: {str(e)}")
    
    return None

# Delete an analysis
def delete_analysis(analysis_id):
    """Delete an analysis from Realtime Database"""
    db_ref = initialize_firebase()
    
    try:
        db_ref.child("content_analyses").child(analysis_id).remove()
        return True
    except Exception as e:
        st.error(f"Error deleting analysis: {str(e)}")
        return False

# Test connection to Realtime Database
def test_firebase_connection():
    """Test the Firebase Realtime Database connection"""
    try:
        # Initialize Firebase
        db_ref = initialize_firebase()
        
        # Create a test node
        test_id = str(uuid.uuid4())
        test_ref = db_ref.child("connection_tests").child(test_id)
        
        # Write test data
        test_ref.set({
            "timestamp": {".sv": "timestamp"},
            "test_id": test_id,
            "status": "success"
        })
        
        # Read it back
        test_data = test_ref.get()
        
        # Delete the test data (cleanup)
        test_ref.remove()
        
        if test_data and test_data.get("status") == "success":
            return True, "Firebase Realtime Database connection successful!"
        else:
            return False, "Test data was written but could not be read back correctly."
            
    except Exception as e:
        return False, f"Firebase connection failed: {str(e)}"

# Display timestamps in a readable format
def format_timestamp(timestamp):
    """Convert Firebase timestamp to readable format"""
    if not timestamp:
        return "N/A"
    
    try:
        # Realtime Database timestamps are milliseconds since epoch
        dt = datetime.datetime.fromtimestamp(timestamp / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return "N/A"

# Update the show_history_tab function
def show_history_tab():
    """Display the history of past analyses with authentication"""
    st.header("Analysis History")
    
    # Get the authenticated user
    user_id = st.session_state.get("authenticated_user")
    
    # Get analyses for the authenticated user
    analyses = get_past_analyses(user_id=user_id, limit=30)
    
    if not analyses:
        st.info("No analysis history found")
    else:
        # Create a DataFrame for better display
        history_data = []
        for analysis in analyses:
            history_data.append({
                "Date": format_timestamp(analysis.get("timestamp")),
                "Keyword": analysis.get("keyword", "N/A"),
                "URL": analysis.get("target_url", "N/A"),
                "Score": analysis.get("score", "N/A"),
                "ID": analysis.get("id", "N/A")
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Allow viewing a specific analysis
        selected_id = st.selectbox("Select an analysis to view details:", 
                                  options=[a["id"] for a in analyses],
                                  format_func=lambda x: f"{next((a['keyword'] for a in analyses if a['id'] == x), 'Unknown')} - {next((a['target_url'] for a in analyses if a['id'] == x), 'Unknown')}")
        
        if selected_id:
            selected_analysis = get_analysis_by_id(selected_id)
            if selected_analysis:
                st.subheader(f"Analysis Details: {selected_analysis.get('keyword', 'Unknown')}")
                
                # Display basic information
                st.markdown(f"""
                **Target URL:** {selected_analysis.get('target_url', 'N/A')}  
                **Score:** {selected_analysis.get('score', 'N/A')}/10  
                **Date:** {format_timestamp(selected_analysis.get('timestamp'))}
                """)
                
                # Display rankings if available
                if "rankings" in selected_analysis and selected_analysis["rankings"]:
                    st.markdown("### Content Rankings")
                    rankings_df = pd.DataFrame({
                        'Rank': range(1, len(selected_analysis["rankings"]) + 1),
                        'Content': [r.get('url', 'Unknown') for r in selected_analysis["rankings"]],
                        'Type': ['Your Content' if r.get('is_target', False) else 'Competitor' for r in selected_analysis["rankings"]],
                        'Information Gain': [round(r.get('gain', 0), 2) for r in selected_analysis["rankings"]]
                    })
                    
                    # Add emoji to make your content stand out
                    rankings_df['Type'] = rankings_df['Type'].map({
                        'Your Content': '🎯 Your Content',
                        'Competitor': 'Competitor'
                    })
                    
                    st.dataframe(rankings_df.set_index('Rank'), use_container_width=True)
                
                # Display improvement summary if available
                if "improvement_summary" in selected_analysis and selected_analysis["improvement_summary"]:
                    st.markdown("### Strategic Analysis")
                    st.markdown(selected_analysis["improvement_summary"])
                
                # Add option to delete this analysis (only for admins)
                if user_id:  # Changed from is_admin to user_id
                    if st.button("Delete This Analysis"):
                        if delete_analysis(selected_id):
                            st.success("Analysis deleted successfully")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to delete analysis")



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
                model="gemini-2.0-flash",
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
    This competitor achieved an Information Gain score of {competitor_score:.2f} by providing unique information in these areas:
    """)
    
    # Ensure competitor_details is a dictionary
    if not isinstance(competitor_details, dict):
        st.error(f"Invalid competitor analysis data type: {type(competitor_details)}")
        st.json(competitor_details)  # Display what we received for debugging
        return
    
    # Group unique information by type
    unique_facts = competitor_details.get('unique_facts', [])
    verifiable_facts = competitor_details.get('verifiable_facts', [])
    specific_metrics = competitor_details.get('specific_metrics', [])
    novel_insights = competitor_details.get('novel_insights', [])
    expert_insights = competitor_details.get('expert_insights', [])
    
    # Show summary counts
    st.markdown(f"""
    - {len(unique_facts)} unique facts not found in other content
    - {len(verifiable_facts)} verifiable facts with data/citations
    - {len(specific_metrics)} specific measurements or data points
    - {len(novel_insights)} novel insights or analysis
    - {len(expert_insights)} expert insights based on experience
    """)
    
    # Show detailed examples - WITHOUT USING EXPANDERS
    st.markdown("### Examples of Unique Information")
    
    # Display categories without nested expanders
    if unique_facts:
        st.markdown("**Unique Facts:**")
        for fact in unique_facts:
            st.markdown(f"- {fact}")
    
    if verifiable_facts:
        st.markdown("**Verifiable Facts with Data/Citations:**")
        for fact in verifiable_facts:
            st.markdown(f"- {fact}")
    
    if specific_metrics:
        st.markdown("**Specific Measurements & Data Points:**")
        for metric in specific_metrics:
            st.markdown(f"- {metric}")
    
    if novel_insights:
        st.markdown("**Novel Insights & Analysis:**")
        for insight in novel_insights:
            st.markdown(f"- {insight}")
    
    if expert_insights:
        st.markdown("**Expert Insights:**")
        for insight in expert_insights:
            st.markdown(f"- {insight}")
    
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

# Update the main function to ensure rich output is preserved
def main():
    st.title("Content Information Gain Analyzer")
    
    # Create tabs for analysis and history
    tab1, tab2 = st.tabs(["Analyze Content", "View History"])
    
    with tab1:
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
                serp_results = get_serp_results(keyword, target_url)
                
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
                analysis = analyze_content(target_content, competitor_contents, keyword, use_google=use_google)
                
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
                    better_competitors = []
                    for idx, row in rankings_df[rankings_df['Rank'] < your_content_rank].iterrows():
                        rank = row['Rank']
                        # Find the corresponding competitor content
                        for i, comp in enumerate(competitor_contents):
                            if comp['title'] == row['Content']:
                                better_competitors.append(comp)
                                break
                    
                    # Make sure there are better competitors before analyzing
                    if better_competitors:
                        # Get strategic summary
                        improvement_summary = summarize_improvement_opportunities(
                            target_content['content'],
                            better_competitors,
                            keyword,
                            use_google=use_google
                        )
                        
                        st.markdown("### Strategic Analysis")
                        st.markdown(improvement_summary)
                        
                        st.markdown("### Detailed Competitor Analysis")
                        for i, comp in enumerate(better_competitors):
                            with st.expander(f"Competitor #{i+1}: {comp['title']}"):
                                if comp['title'] in analysis['competitor_analyses']:
                                    comp_analysis = analysis['competitor_analyses'][comp['title']]
                                    show_competitor_analysis(comp_analysis['gain'], comp_analysis['analysis'])
                    
                # Always try to save results automatically
                try:
                    # Check if user is authenticated before saving
                    if st.session_state.get("authenticated_user"):
                        analysis_id = save_analysis_with_summary(
                            keyword, 
                            target_url, 
                            analysis, 
                            improvement_summary=improvement_summary,
                            user_id=st.session_state["authenticated_user"]  # Add user ID to save
                        )
                        
                        if analysis_id:
                            st.success(f"Analysis saved successfully! You can access it in the History tab.")
                    else:
                        # If not authenticated, prompt to login for saving
                        st.warning("Login required to save analysis results")
                        if st.button("Login to Save Results"):
                            st.session_state["show_login"] = True
                            st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error saving analysis: {str(e)}")
        
    with tab2:
        # Require authentication for viewing history
        if require_auth():
            show_history_tab()






if __name__ == "__main__":
    main()