import requests
from datetime import datetime, timedelta
from xml.etree import ElementTree
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def get_article_count(year_month):
    url = f"https://www.tomsguide.com/sitemap-{year_month}.xml"
    print(f"Fetching {url}...")
    try:
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        root = ElementTree.fromstring(response.content)
        count = len(root.findall('.//{*}url'))
        print(f"Found {count} articles")
        return count
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return 0

def analyze_sitemaps():
    print("Starting analysis...")
    months = []
    current_date = datetime.now()
    
    for i in range(6):
        date = current_date - timedelta(days=30*i)
        months.append(date.strftime("%Y-%m"))
    
    print(f"Analyzing months: {months}")
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        counts = list(executor.map(get_article_count, months))
    
    df = pd.DataFrame({
        'Month': months,
        'Articles': counts
    })
    
    return df

if __name__ == "__main__":
    print("Script starting...")
    results = analyze_sitemaps()
    print("\nMonthly Article Counts:")
    print(results.to_string(index=False))