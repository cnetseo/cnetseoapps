import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

# Rate limiting decorator - 15 requests per minute
@sleep_and_retry
@limits(calls=15, period=60)
def rate_limited_request(url: str) -> requests.Response:
    return http.get(url)

@st.cache_data
def parse_date(date_string: str) -> datetime:
    """Parse date string in various formats."""
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_string}")

def get_wayback_content_with_headers(url: str, timestamp: str, selector: str) -> Optional[List[str]]:
    """Fetch content from Wayback Machine with rate limiting and retries."""
    wayback_url = f"http://web.archive.org/web/{timestamp}/{url}"
    logger.info(f"Fetching content from: {wayback_url}")
    
    try:
        response = rate_limited_request(wayback_url)
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            logger.info("Successfully received response")
        else:
            logger.error(f"Failed to fetch URL. Status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        headers = soup.select(selector)
        logger.info(f"Found {len(headers)} headers with selector '{selector}'")
        
        if headers:
            header_texts = [header.get_text(strip=True) for header in headers]
            for i, text in enumerate(header_texts[:3]):
                logger.info(f"Sample header {i + 1}: {text}")
            return header_texts
        else:
            all_h4s = soup.find_all('h4')
            logger.warning(f"Found {len(all_h4s)} total h4 elements, but none with target selector")
            if all_h4s:
                logger.warning("Sample h4 elements found:")
                for h4 in all_h4s[:3]:
                    logger.warning(f"H4 classes: {h4.get('class', 'no-class')} | Text: {h4.get_text(strip=True)}")
            return []
            
    except requests.RequestException as e:
        logger.error(f"Request Exception for {wayback_url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {wayback_url}: {str(e)}")
        return None

def process_urls(urls_input: str) -> List[str]:
    """Process URLs from text input, handling various formats."""
    urls = []
    for line in urls_input.split('\n'):
        # Clean and validate each URL
        url = line.strip()
        if url and (url.startswith('http://') or url.startswith('https://')):
            urls.append(url)
    return urls

def generate_comparison_csv(urls: List[str], 
                          old_date: str, 
                          new_date: str,
                          old_selector: str,
                          new_selector: str) -> Tuple[str, pd.DataFrame]:
    """Compare content between dates with rate limiting."""
    logger.info(f"Starting comparison between dates: {old_date} and {new_date}")
    
    old_timestamp = parse_date(old_date).strftime("%Y%m%d")
    new_timestamp = parse_date(new_date).strftime("%Y%m%d")
    output_data = []
    
    for url in urls:
        logger.info(f"\nProcessing URL: {url}")
        
        old_headers = get_wayback_content_with_headers(url, old_timestamp, old_selector)
        old_headers = old_headers if old_headers is not None else []
        old_set = set(old_headers)
        logger.info(f"Old version items count: {len(old_headers)}")
        
        new_headers = get_wayback_content_with_headers(url, new_timestamp, new_selector)
        new_headers = new_headers if new_headers is not None else []
        new_set = set(new_headers)
        logger.info(f"New version items count: {len(new_headers)}")
        
        added = list(new_set - old_set)
        removed = list(old_set - new_set)
        unchanged = list(old_set & new_set)
        total_text_changes = len(added) + len(removed)
        
        logger.info(f"Changes found - Added: {len(added)}, Removed: {len(removed)}, Unchanged: {len(unchanged)}")
        
        output_data.append({
            'URL': url,
            f'New Date ({new_date})': ' | '.join(new_headers),
            f'Old Date ({old_date})': ' | '.join(old_headers),
            'Added Items': ' | '.join(added) if added else 'None',
            'Removed Items': ' | '.join(removed) if removed else 'None',
            'Unchanged Items': ' | '.join(unchanged) if unchanged else 'None',
            'Total Text Changes': total_text_changes,
            'New Count': len(new_headers),
            'Old Count': len(old_headers),
            'Count Difference': len(new_headers) - len(old_headers)
        })
    
    df = pd.DataFrame(output_data)
    output_file = 'comparison_results.csv'
    df.to_csv(output_file, index=False)
    return output_file, df

def main():
    st.title("Wayback Machine Content Comparison")
    
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

    class StreamlitHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            st.session_state.log_messages.append(log_entry)

    streamlit_handler = StreamlitHandler()
    streamlit_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(streamlit_handler)
    
    # URL input area
    urls_text = st.text_area("Enter URLs (one per line):", height=150)
    
    # Configure dates and selectors
    st.write("Configure dates and selectors:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Earlier Version**")
        old_date = st.text_input("Date (YYYY-MM-DD)", "2024-06-01")
        old_selector = st.text_input("CSS Selector", "h4.c-bestListProductListing_hed")
        
    with col2:
        st.markdown("**Later Version**")
        new_date = st.text_input("Date (YYYY-MM-DD)", "2024-12-01")
        new_selector = st.text_input("CSS Selector", "h4.c-bestListListicle_hed")
    
    if st.button("Compare and Generate CSV"):
        try:
            logger.info("Starting new comparison run...")
            urls = process_urls(urls_text)
            
            if not urls:
                st.error("No valid URLs found. Please enter URLs starting with http:// or https://")
                return
                
            output_file, df = generate_comparison_csv(
                urls, old_date, new_date, old_selector, new_selector
            )
            
            st.subheader("Comparison Results")
            st.dataframe(df)
            
            with open(output_file, 'rb') as f:
                st.download_button(
                    label="Download CSV Results",
                    data=f,
                    file_name='wayback_comparison_results.csv',
                    mime='text/csv'
                )
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            st.error(f"Error processing data: {str(e)}")
        
        st.subheader("Debug Logs")
        for log in st.session_state.log_messages:
            st.text(log)

if __name__ == "__main__":
    main()