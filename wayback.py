
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

def get_wayback_content(url, timestamp):
    wayback_url = f"http://web.archive.org/web/{timestamp}/{url}"
    response = requests.get(wayback_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the main content area (adjust selectors as needed)
        main_content = soup.find('main') or soup.find('body')
        if main_content:
            return ' '.join(p.get_text() for p in main_content.find_all('p'))
    return None


def parse_date(date_string):
    """Parse date string in various formats."""
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_string}")

def compare_wayback_content(url, date1_str, date2_str):
    date1 = parse_date(date1_str)
    date2 = parse_date(date2_str)

    timestamp1 = date1.strftime("%Y%m%d")
    timestamp2 = date2.strftime("%Y%m%d")

    content_a = get_wayback_content(url, timestamp1)
    time.sleep(1)  # Be nice to the Wayback Machine API
    content_b = get_wayback_content(url, timestamp2)

    if content_a is None or content_b is None:
        return "Failed to retrieve content for one or both dates."


    else:
        return content_a,content_b

