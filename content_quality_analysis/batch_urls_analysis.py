
# --- Configuration --
import os
import datetime
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import logging
import re
import sys
import time
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import threading # For potential future use with locks if needed

API_KEY = 'AIzaSyBhq9dqHAxX9Idn4FFVC_q1JnpW5BvcSGE'

# --- Concurrency Configuration ---
MAX_WORKERS = 3 # Number of parallel threads

# --- LLM Configuration ---
MODEL_NAME = "gemini-2.0-flash" # Or "gemini-pro" etc.
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
# GENERATION_CONFIG = {"max_output_tokens": 2048} # Optional

# --- Scraping Configuration ---
REQUEST_TIMEOUT = 20 # Increased slightly for potentially slower responses under load
REQUEST_DELAY = 0.1 # Optional small delay INSIDE scrape_text_content if needed for politeness to specific servers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- File Configuration ---
INPUT_CSV_FILE = "content_quality_analysis/Pages to Prune - Sheet10.csv"
OUTPUT_CSV_FILE = f"content_analysis_results_threaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# --- Setup Logging ---
# Add thread name to log format for clarity
log_format = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

csv_write_lock = threading.Lock()

OUTPUT_COLUMNS = [
    'Analysis Timestamp', 'URL', 'Scraping Status', 'LLM Model Used',
    'Page Topic (Extracted/Provided)', 'Is YMYL (Provided/Default)',
    'Quality Score (1-10)', 'Overall Assessment',
    'Key Strengths (Alignment with Guidelines)',
    'Key Weaknesses / Red Flags (Deviation from Guidelines)',
    'NoIndex Recommendation (to Improve Overall Site Quality)'
]

# --- Helper Functions (scrape, construct_prompt, get_analysis, parse) ---
# These functions remain largely the same as the previous non-threaded version
# Ensure they are robust and handle their own exceptions where possible.

def scrape_text_content(url):
    """Fetches a URL and extracts the main text content using BeautifulSoup."""
    logging.info(f"Attempting to scrape URL: {url}")
    try:
        # Optional: Add a small delay here if needed to be polite to specific servers
        # time.sleep(REQUEST_DELAY)
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main')
        if not main_content: main_content = soup.body
        if not main_content:
            logging.warning(f"Could not find main content/body for {url}")
            return None, "Scraping Error: No body tag found"
        text = main_content.get_text(separator=' ', strip=True)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        page_title = soup.title.string.strip() if soup.title else "No Title Found"
        logging.info(f"Successfully scraped ~{len(text)} characters from {url}")
        return text, page_title
    except requests.exceptions.Timeout:
        logging.error(f"Scraping Timeout for {url} after {REQUEST_TIMEOUT}s")
        return None, f"Scraping Error: Timeout ({REQUEST_TIMEOUT}s)"
    except requests.exceptions.RequestException as e:
        logging.error(f"Scraping Request Error for {url}: {e}")
        return None, f"Scraping Error: {e}"
    except Exception as e:
        logging.error(f"Unexpected scraping/parsing error for {url}: {e}")
        return None, f"Scraping Error: Unexpected ({e})"

def construct_prompt(page_topic, is_ymyl, content_to_analyze):
    """Constructs the detailed analysis prompt for Gemini."""
    # Determine the Yes/No string for YMYL
    ymyl_string = 'Yes (Emphasize Expertise & Trustworthiness scrutiny)' if is_ymyl else 'No'

    # --- Truncate content BEFORE putting it in the prompt if necessary ---
    if len(content_to_analyze) > 30000: # Adjust length limit as needed
         logging.warning(f"Content for topic '{page_topic}' truncated for LLM input.")
         content_to_analyze = content_to_analyze[:30000] + "... [Content Truncated]"

    # --- Construct the prompt using f-string formatting correctly ---
    prompt = f"""**Role:** You are an expert SEO Content Quality Analyst. Your sole focus is evaluating web content based on Google's guidelines for creating helpful, reliable, people-first content, as outlined in their official documentation. You prioritize identifying content created to benefit people over content made primarily for search engine rankings.

**Goal:** Analyze the provided website content *strictly* through the lens of Google's helpful content self-assessment questions. Determine its potential quality level from this perspective, identify specific strengths and weaknesses according to these guidelines, and provide a numerical quality score.

**Context:**
*   **Page Topic/Main Keyword(s):** {page_topic}
*   **Intended Audience (as perceived from content):** [LLM should infer this]
*   **Apparent Page Goal (what the content tries to help the user do/understand):** [LLM should infer this]
*   **Is this YMYL Content?** {ymyl_string}

**Content to Analyze:**
--- BEGIN CONTENT ---
{content_to_analyze}
--- END CONTENT ---

**Analysis Instructions:**
Evaluate the content *based only on the text provided* against the following criteria derived directly from Google's helpful content guidelines. For each point, provide brief reasoning and examples from the text where possible.

**1. Originality, Value & Substance:**
    *   **Original Info/Analysis:** Does the content provide original information, reporting, research, or insightful analysis beyond the obvious? Does it avoid simply copying or rewriting other sources without substantial added value?
    *   **Substantial & Complete:** Does the content offer a substantial, complete, or comprehensive description of the topic?
    *   **Value vs. Alternatives:** Does the content seem to provide substantial value when compared to hypothetical other pages on the same topic (based on its depth and uniqueness)?

**2. Expertise, Experience & Trustworthiness (E-E-A-T Signals in Content):**
    *   **Demonstrated Expertise/Experience:** Does the content clearly demonstrate first-hand expertise (e.g., from using a product/service, visiting a place) or a depth of knowledge? Does it *sound* like it was written by an expert or enthusiast?
    *   **Trustworthiness Cues:** Does the content present information in a way that builds trust (e.g., clear reasoning, supporting evidence within the text, balanced views where appropriate)? Are there any easily verifiable factual errors apparent *within the text itself*?
    *   *(Note: You cannot assess external author pages or site reputation, focus only on signals within the provided text).*

**3. People-First Focus & Satisfying Experience:**
    *   **Audience Focus:** Does the content seem created for a specific, existing or intended audience, or does it feel generic?
    *   **Goal Achievement & Satisfaction:** After reading this content, would someone likely feel they've learned enough to achieve their goal? Would they likely leave feeling they've had a satisfying experience, or would they feel the need to search again for better info?
    *   **Recommendation Worthiness:** Is this the sort of content (based on its quality and helpfulness) that a user might bookmark, share, or recommend?

**4. Presentation & Production Quality:**
    *   **Readability & Clarity:** Is the content well-written, clear, and easy to understand? Does it have significant spelling or stylistic issues?
    *   **Production Care:** Does the content appear well-produced and cared for, or does it seem sloppy or hastily produced?
    *   **Title Alignment (Inferred):** Does the *beginning* of the content align with what a descriptive, helpful, non-exaggerated, non-shocking page title would promise?

**5. Potential "Search-Engine First" Red Flags (Evaluate based on content patterns):**
    *   **Primary Motivation:** Does the content seem created *primarily* to attract search engine visits rather than to inform a specific audience?
    *   **Value Add vs. Summarization:** Is it mainly summarizing what others have said without adding much original value or insight?
    *   **Topic Selection:** Does the topic feel authentically chosen for the audience, or potentially chosen just because it's trending or perceived to be easy traffic (especially if lacking demonstrated expertise)?
    *   **Completeness/Need to Re-Search:** Does the content feel incomplete, potentially leaving readers needing to search again?
    *   **Unnatural Language/Structure:** Are there signs of writing to a specific word count, excessive automation patterns, or unnatural keyword usage?
    *   **Unanswered Questions/False Promises:** Does the content promise information it doesn't deliver (e.g., speculating on unconfirmed details)?

**Output Request:**
Structure your response *exactly* like this, using these specific headers:

### Quality Score (1-10):
[Provide a single integer score from 1 (very low quality, likely search-first) to 10 (very high quality, clearly helpful & people-first) reflecting the overall assessment based *specifically* on the Google guidelines evaluated above.]

### Overall Assessment:
[Provide one qualitative assessment summarizing how well the content aligns with Google's helpful, reliable, people-first content guidelines (e.g., Strong Alignment - People-First, Moderate Alignment - Some Issues, Weak Alignment - Likely Search-First Concerns, Poor Alignment - Significant Issues).]

### Key Strengths (Alignment with Guidelines):
[List 2-3 specific positive points based on the analysis instructions, citing examples where possible.]

### Key Weaknesses / Red Flags (Deviation from Guidelines):
[List the 2-3 most significant areas where the content deviates from the guidelines or raises potential "search-engine first" red flags, citing examples.]

### NoIndex Recommendation (to Improve Overall Site Quality):
[Based on your assessment of the content give a one sentence recommendation on whether this page should be noindexed or not. Use "Yes" or "No" only. For context, we do not have resources to improve many pages, but ideally want to only noindex content we truly feel is low quality and can lower the overall quality of the site.]
"""
    return prompt

def get_gemini_analysis(prompt_text, model):
    """Sends the prompt to the Gemini API and returns the response text. Assumes model is thread-safe or handles internal locking."""
    # This function largely assumes the google-generativeai client library handles thread safety for API calls.
    try:
        # Note: Logging here will show interleaved messages from different threads
        logging.info(f"Sending request to Google Gemini API...")
        response = model.generate_content(
            prompt_text,
            safety_settings=SAFETY_SETTINGS
            # generation_config=GENERATION_CONFIG
        )
        # Error/Block handling (same as before)
        if not response.parts:
             # ... (rest of error handling logic from previous version) ...
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 logging.error(f"Gemini request blocked. Reason: {block_reason}")
                 return f"Error: Content blocked by safety filters ({block_reason})"
             elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                 finish_reason = response.candidates[0].finish_reason
                 logging.warning(f"Gemini response finished unexpectedly. Reason: {finish_reason}")
                 return response.text if hasattr(response, 'text') else f"Error: Finished due to {finish_reason}, no text returned."
             else:
                 logging.error("Gemini returned an empty response.")
                 return "Error: Empty response from Gemini."

        analysis_text = response.text
        logging.info("Received response from Google Gemini API.")
        return analysis_text
    except Exception as e:
        # Log exceptions happening during the API call within the thread
        logging.error(f"Error calling Google Gemini API: {e}")
        return f"Error: Exception during API call - {e}" # Return error message

def parse_llm_response(response_text):
    """Parses the LLM's structured response into a dictionary."""
    # Use keys that exactly match the intended output headers for clarity
    parsed_data = {
        "Quality Score (1-10)": "Parsing Error",
        "Overall Assessment": "Parsing Error",
        "Key Strengths (Alignment with Guidelines)": "Parsing Error",
        "Key Weaknesses / Red Flags (Deviation from Guidelines)": "Parsing Error",
        "NoIndex Recommendation (to Improve Overall Site Quality)": "Parsing Error"
    }

    if not response_text or response_text.startswith("Error:"):
        error_msg = response_text if response_text else "Parsing Error: Empty response received"
        # Ensure all keys (matching the final desired output) get the error message
        for key in parsed_data:
            parsed_data[key] = error_msg
        # Attempt to extract score even if other parts failed due to error message format
        score_match = re.search(r"### Quality Score \(1-10\):\s*([\d.]+)", response_text or "", re.IGNORECASE)
        if score_match:
             parsed_data["Quality Score (1-10)"] = score_match.group(1).strip()
        return parsed_data

    # Define patterns using the EXACT headers from the prompt's Output Request
    # Use re.escape to handle special characters like ( ) / in headers safely
    header1 = r"### Quality Score \(1-10\):" # Score doesn't need escape as () are handled
    header2 = r"### Overall Assessment:"
    header3 = r"### Key Strengths \(Alignment with Guidelines\):"
    header4 = r"### Key Weaknesses / Red Flags \(Deviation from Guidelines\):"
    header5 = r"### NoIndex Recommendation \(To Improve Overall Site Quality\):"

    patterns = {
        # Key: Matches the key in parsed_data dict above
        # Value: Regex pattern to capture content for that section
        "Quality Score (1-10)": rf"{header1}\s*([\d.]+)\s*{header2}",
        "Overall Assessment": rf"{header2}\s*(.*?)\s*{header3}",
        "Key Strengths (Alignment with Guidelines)": rf"{header3}\s*(.*?)\s*{header4}",
        "Key Weaknesses / Red Flags (Deviation from Guidelines)": rf"{header4}\s*(.*?)\s*{header5}",
        "NoIndex Recommendation (to Improve Overall Site Quality)": rf"{header5}\s*(.*)" # Captures till the end
    }

    all_found = True # Flag to check if all sections were parsed
    for key, pattern in patterns.items():
        # Use re.DOTALL to make '.' match newlines, re.IGNORECASE for flexibility
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            # Group 1 contains the captured content between headers (or score/last section)
            parsed_data[key] = match.group(1).strip()
            # logging.debug(f"Successfully parsed section: '{key}'") # Optional debug logging
        else:
            logging.warning(f"Could not find or parse section using key: '{key}'")
            all_found = False
            # Keep default "Parsing Error" for this key

    # If not all sections were found, log the full response for easier debugging
    if not all_found:
         logging.warning("Parsing incomplete. Raw response was:\n<<<<<\n%s\n>>>>>", response_text)


    # Score validation (remains the same)
    score_val = parsed_data.get("Quality Score (1-10)")
    if score_val and score_val != "Parsing Error" and not score_val.startswith("Error:"):
        try:
            float(score_val)
        except ValueError:
            logging.warning(f"Parsed score '{score_val}' is not a valid number.")
            parsed_data["Quality Score (1-10)"] = "Parsing Error: Non-numeric value"

    return parsed_data

def append_result_to_csv(result_dict, lock, columns, filename):
    """Appends a single result dictionary to the CSV file, handling headers and thread safety."""
    if not result_dict:
        logging.warning("Attempted to append an empty result dictionary. Skipping.")
        return

    # Ensure the lock is acquired before accessing the file
    with lock:
        try:
            # Check if file exists to determine if header needs to be written
            file_exists = os.path.isfile(filename)

            # Convert the single dictionary result to a DataFrame row
            df_row = pd.DataFrame([result_dict])

            # Ensure the DataFrame row has the correct columns in the correct order
            # Adds missing columns as NaN, drops extra columns
            df_row = df_row.reindex(columns=columns)

            # Append to CSV
            df_row.to_csv(
                filename,
                mode='a',               # 'a' for append
                header=not file_exists, # Write header only if file doesn't exist
                index=False,
                encoding='utf-8-sig'    # Good for Excel compatibility
            )
            # logging.info(f"Appended result for {result_dict.get('URL', 'N/A')} to {filename}") # Optional detailed log
        except Exception as e:
            logging.error(f"Error writing result for {result_dict.get('URL', 'N/A')} to CSV file {filename}: {e}")


# --- Core Processing Function for Each Thread ---
def process_url(url_row, gemini_model):
    """Handles scraping, LLM analysis, and parsing for a single URL row."""
    url = url_row['urls'] # Correctly reads from 'urls' column
    thread_name = threading.current_thread().name
    logging.info(f"Processing URL: {url}")

    # --- Get Context from row ---
    is_ymyl_input = url_row.get('Is YMYL', 'No')
    is_ymyl = str(is_ymyl_input).strip().lower() in ['yes', 'true', '1', 'y']
    provided_topic = url_row.get('Page Topic') # Get topic from CSV if available

    # --- Initialize result_data (set topic later) ---
    result_data = {
        'Analysis Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'URL': url,
        'LLM Model Used': MODEL_NAME,
        'Is YMYL (Provided/Default)': 'Yes' if is_ymyl else 'No',
        'Page Topic (Extracted/Provided)': 'N/A', # Default placeholder
        'Scraping Status': 'Not Attempted',
        'Quality Score (1-10)': 'N/A', 'Overall Assessment': 'N/A',
        'Key Strengths and Weaknesses': 'N/A', 'Top Risk Factors': 'N/A',
        'NoIndex Recommendations': 'N/A'
    }

    final_page_topic = "N/A" # Default topic value

    try:
        # 1. Scrape Content
        scraped_text, extracted_title_or_error = scrape_text_content(url)

        if scraped_text:
            result_data['Scraping Status'] = 'Success'
            # --- Determine the final page topic to use ---
            if provided_topic:
                final_page_topic = provided_topic # Use CSV topic if provided
            elif extracted_title_or_error != "No Title Found" and extracted_title_or_error:
                final_page_topic = extracted_title_or_error # Use scraped title if valid
            else:
                final_page_topic = "Topic Not Found" # Fallback if no CSV topic and no scraped title
            result_data['Page Topic (Extracted/Provided)'] = final_page_topic # Store the determined topic
            # --- End topic determination ---
        else:
            result_data['Scraping Status'] = extracted_title_or_error
            result_data['Page Topic (Extracted/Provided)'] = "N/A - Scraping Failed" # Set topic on scraping failure
            logging.warning(f"Skipping analysis for {url} due to scraping failure.")
            return result_data # Return partial results with error

        # 2. Construct Prompt (Use the determined final_page_topic)
        prompt = construct_prompt(
            page_topic=final_page_topic,
            is_ymyl=is_ymyl,
            content_to_analyze=scraped_text
        )

        # 3. Get LLM Analysis (pass the shared model instance)
        llm_response = get_gemini_analysis(prompt, gemini_model)
        logging.info(f"Raw LLM Response for {url}:\n<<<<<\n{llm_response}\n>>>>>") # Good for debugging

        # 4. Parse LLM Response
        analysis_results = parse_llm_response(llm_response)

        # Merge analysis results into the main result dictionary
        result_data.update(analysis_results)

    except Exception as e:
        # Catch any unexpected errors during the process_url execution
        logging.error(f"Unexpected error processing {url}: {e}", exc_info=True)
        result_data['Overall Assessment'] = f"Processing Error: {e}" # Mark error in output
        result_data['Scraping Status'] = result_data.get('Scraping Status', 'Unknown due to error')
        result_data['Page Topic (Extracted/Provided)'] = result_data.get('Page Topic (Extracted/Provided)', 'N/A - Processing Error') # Keep existing topic if possible

    return result_data
# --- Main Execution ---
if __name__ == "__main__":
    if not API_KEY:
        logging.error("Google API key not found in .env file.")
        sys.exit("API Key Error: Exiting script.")

    # --- Read Input ---
    try:
        input_df = pd.read_csv(INPUT_CSV_FILE)
        # --- Ensure Input CSV has the correct URL column name ---
        url_column_name = 'urls' # Make sure this matches your CSV header EXACTLY
        if url_column_name not in input_df.columns:
            logging.error(f"Input CSV '{INPUT_CSV_FILE}' must contain a '{url_column_name}' column.")
            sys.exit(f"Input CSV Error: Missing '{url_column_name}' column.")
        logging.info(f"Read {len(input_df)} URLs from {INPUT_CSV_FILE}")
    except FileNotFoundError:
        logging.error(f"Input CSV file not found: {INPUT_CSV_FILE}")
        sys.exit("File Not Found Error: Exiting script.")
    except Exception as e:
        logging.error(f"Error reading input CSV {INPUT_CSV_FILE}: {e}")
        sys.exit("CSV Read Error: Exiting script.")

    # --- Configure Gemini Model ---
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(
            MODEL_NAME,
            safety_settings=SAFETY_SETTINGS
        )
        logging.info(f"Google Gemini client configured for model: {MODEL_NAME}")
    except Exception as e:
        logging.error(f"Failed to configure Google Gemini client: {e}")
        sys.exit("Gemini Configuration Error: Exiting script.")

    total_urls = len(input_df)
    start_time = time.time()
    processed_count = 0 # Initialize counter outside the loop

    logging.info(f"Starting analysis with {MAX_WORKERS} parallel workers...")
    logging.info(f"Results will be appended incrementally to: {OUTPUT_CSV_FILE}")

    # --- Use ThreadPoolExecutor ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks (pass row, model instance)
        future_to_url = {
            executor.submit(process_url, row, model): row[url_column_name]
            for index, row in input_df.iterrows()
        }

        # Process futures as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                # Get the result dictionary from the completed future
                result = future.result() # This is the result_data dictionary

                # --- Append the single result to the CSV ---
                if result: # Ensure result is not None if process_url can return None on error
                    append_result_to_csv(result, csv_write_lock, OUTPUT_COLUMNS, OUTPUT_CSV_FILE)
                else:
                    logging.warning(f"Received None result for URL {url}. Not appending to CSV.")

                processed_count += 1
                logging.info(f"Completed analysis and saved result for URL {processed_count}/{total_urls}: {url}")

            except Exception as exc:
                # Catch exceptions raised *within* the process_url function OR during future.result()
                logging.error(f"URL {url} generated an exception during processing: {exc}", exc_info=True)
                # --- Optionally write an error row to the CSV ---
                error_result = {
                    'Analysis Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'URL': url,
                    'Scraping Status': 'Failed in thread/future',
                    'Overall Assessment': f'Thread Execution Error: {exc}',
                    # Fill other essential columns with 'ERROR' or 'N/A'
                    'LLM Model Used': MODEL_NAME,
                    'Is YMYL (Provided/Default)': 'N/A',
                    'Page Topic (Extracted/Provided)': 'N/A',
                    'Quality Score (1-10)': 'ERROR',
                    'Key Strengths (Alignment with Guidelines)': 'ERROR',
                    'Key Weaknesses / Red Flags (Deviation from Guidelines)': 'ERROR',
                    'NoIndex Recommendation (to Improve Overall Site Quality)': 'ERROR'
                }
                append_result_to_csv(error_result, csv_write_lock, OUTPUT_COLUMNS, OUTPUT_CSV_FILE)
                # --- End optional error row ---
                processed_count += 1 # Still count it as processed (with error)

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"--- Batch analysis complete for {total_urls} URLs in {duration:.2f} seconds ---")
    logging.info(f"Results saved incrementally to {OUTPUT_CSV_FILE}")