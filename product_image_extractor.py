import re
import csv
from bs4 import BeautifulSoup
import os
import io
import sys
import requests
from lxml import etree
import time
from urllib.parse import urlparse

def extract_with_direct_xpath(html_content, source_url=""):
    """
    Extract products and count their images using XPath
    
    Args:
        html_content (str): HTML content to parse
        source_url (str): URL this content was fetched from (for reporting)
    
    Returns:
        list: List of dictionaries with product name and image count
    """
    try:
        parser = etree.HTMLParser()
        tree = etree.parse(io.StringIO(html_content), parser)
        
        products = []
        
        # Find all product sections
        product_sections = tree.xpath("//div[contains(@class, 'c-bestListListicle_editorial')]")
        
        for section in product_sections:
            try:
                # Get product name
                product_name_elements = section.xpath(".//h4[contains(@class, 'c-bestListListicle_hed')]/text()")
                if not product_name_elements:
                    print(f"Warning: Found a product section without a name in {source_url}, skipping...")
                    continue
                    
                product_name = product_name_elements[0].strip()
                
                # Get all gallery images using the XPath provided - just count them
                img_elements = section.xpath(".//div[@class='c-cmsImage c-bestListGallery_image']//picture//img")
                image_count = len(img_elements)
                
                products.append({
                    'product_name': product_name,
                    'image_count': image_count,
                    'source_url': source_url
                })
            except Exception as e:
                print(f"Warning: Error processing a product section in {source_url}: {e}")
                continue
        
        return products
        
    except Exception as e:
        print(f"Error in extraction process for {source_url}: {e}")
        return []

def save_to_csv(all_products, output_file='product_images_results.csv'):
    """
    Saves all product and image count data to a CSV file
    
    Args:
        all_products (list): List of product dictionaries
        output_file (str): Output CSV filename
    """
    try:
        # Don't overwrite if appending
        file_exists = os.path.isfile(output_file)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['product_name', 'image_count', 'source_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for product in all_products:
                writer.writerow({
                    'product_name': product['product_name'],
                    'image_count': product['image_count'],
                    'source_url': product['source_url']
                })
        
        print(f"CSV file created: {output_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def fetch_html_from_url(url, request_count=None, rate_limit=0.1):
    """
    Fetch HTML content from a URL with rate limiting
    
    Args:
        url (str): URL to fetch
        request_count (int, optional): Current request number (for reporting)
        rate_limit (float, optional): Minimum time between requests in seconds
    
    Returns:
        str: HTML content
    """
    try:
        # Rate limiting - sleep to ensure we respect rate limits
        time.sleep(rate_limit)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        count_str = f"[{request_count}] " if request_count is not None else ""
        print(f"{count_str}Fetching: {url}")
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def process_html_content(html_content, source_url=""):
    """
    Process HTML content to extract products and image counts
    
    Args:
        html_content (str): HTML content
        source_url (str): URL this content came from
    
    Returns:
        list: Extracted products and image counts
    """
    # Extract products and image counts
    products = extract_with_direct_xpath(html_content, source_url)
    
    if not products:
        print(f"No products were found in the HTML content from {source_url}.")
        return []
    
    # Print summary
    print(f"Found {len(products)} products on {source_url}:")
    for product in products:
        print(f"- {product['product_name']}: {product['image_count']} images")
    
    return products

def process_url(url, request_count=None, rate_limit=0.1):
    """
    Process a single URL
    
    Args:
        url (str): URL to process
        request_count (int, optional): Current request number
        rate_limit (float, optional): Rate limiting in seconds
        
    Returns:
        list: Products found on the page
    """
    # Fetch HTML
    html_content = fetch_html_from_url(url, request_count, rate_limit)
    
    if not html_content:
        print(f"Failed to fetch content from URL: {url}")
        return []
        
    # Process the content
    products = process_html_content(html_content, url)
    return products

def process_url_list(urls, rate_limit=0.1):
    """
    Process a list of URLs with rate limiting
    
    Args:
        urls (list): List of URLs to process
        rate_limit (float): Minimum time between requests in seconds
        
    Returns:
        list: All products found
    """
    all_products = []
    
    for i, url in enumerate(urls):
        try:
            # Process the URL
            products = process_url(url, i+1, rate_limit)
            
            # Add products to the master list
            all_products.extend(products)
            
            # Report progress
            print(f"Processed URL {i+1}/{len(urls)}: {len(products)} products found")
            
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
    
    return all_products

def read_urls_from_csv(csv_file):
    """
    Read URLs from a CSV file
    
    Args:
        csv_file (str): Path to CSV file
        
    Returns:
        list: List of URLs
    """
    urls = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip().startswith(('http://', 'https://')):
                    urls.append(row[0].strip())
    except Exception as e:
        print(f"Error reading URL CSV: {e}")
    
    return urls

def main():
    """
    Main function to handle inputs and process URLs
    """
    try:
        if len(sys.argv) < 2:
            print("Usage: python script.py [urls.csv|single_url]")
            return
        
        input_path = sys.argv[1]
        
        # Set rate limit - 10 URLs per second max = 0.1 seconds between requests
        rate_limit = 0.1  # seconds between requests
        
        # Check if input is a CSV file, a URL, or a local HTML file
        if input_path.endswith('.csv'):
            print(f"Reading URLs from CSV file: {input_path}")
            urls = read_urls_from_csv(input_path)
            
            if not urls:
                print("No valid URLs found in CSV file.")
                return
                
            print(f"Found {len(urls)} URLs to process")
            all_products = process_url_list(urls, rate_limit)
            
        elif input_path.startswith(('http://', 'https://')):
            print(f"Processing single URL: {input_path}")
            all_products = process_url(input_path)
            
        elif os.path.exists(input_path):
            print(f"Reading content from local HTML file: {input_path}")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding if UTF-8 fails
                with open(input_path, 'r', encoding='latin-1') as f:
                    html_content = f.read()
                    print("Note: File was read using latin-1 encoding")
                    
            all_products = process_html_content(html_content, f"local_file:{input_path}")
            
        else:
            print(f"Error: '{input_path}' is not a valid CSV file, URL, or HTML file.")
            return
        
        # Save all products to a single CSV file
        if all_products:
            save_to_csv(all_products)
            print(f"Successfully processed {len(all_products)} total products")
        else:
            print("No products were found in any of the URLs.")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()