import pandas as pd
import requests
from bs4 import BeautifulSoup
import argparse


def scrape_mattress_data(urls):
    """Scrape mattress data from the provided URLs and return as a DataFrame."""
    all_mattresses = {}
    all_attributes = set()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all mattress sections
            mattress_sections = soup.select("h3.c-bestListListicle_subhed")
            #print(f'These are the sections {mattress_sections}')
            
            for section in mattress_sections:
                mattress_name = section.text.strip()
                #print(f'This is the mattress name {mattress_name}')
                if not mattress_name:
                    continue
                
                # Find the parent container that holds this mattress's data
                parent_container = section.find_parent("div", class_="c-bestListListicle_item")
                print(f'This is the parent container {parent_container}')
                if not parent_container:
                    continue
                
                # Get all attribute sections for this mattress
                attribute_sections = parent_container.select("div.c-bestListProductRail_section")
                print(attribute_sections)
                
                mattress_data = {}
                
                for attr_section in attribute_sections:
                    try:
                        # Get attribute name
                        attr_name_elem = attr_section.select_one("span")
                        if not attr_name_elem:
                            continue
                        attr_name = attr_name_elem.text.strip()
                        
                        # Get attribute value
                        attr_value_elem = attr_section.select_one("span.c-bestListProductRail_value")
                        if not attr_value_elem:
                            continue
                        attr_value = attr_value_elem.text.strip()
                        
                        # Store the attribute and its value
                        mattress_data[attr_name] = attr_value
                        all_attributes.add(attr_name)
                    except Exception as e:
                        print(f"Error extracting attribute for {mattress_name}: {e}")
                
                # Add this mattress's data to our collection
                all_mattresses[mattress_name] = mattress_data
        
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
    
    # Create a pandas DataFrame
    df = pd.DataFrame(index=sorted(list(all_attributes)))
    
    # Fill in the DataFrame with our data
    for mattress_name, attributes in all_mattresses.items():
        df[mattress_name] = pd.Series(attributes)
    
    return df


def save_data(df, output_file):
    """Save the DataFrame to CSV and Excel files."""
    # Save as CSV
    df.to_csv(output_file + '.csv')
    
    # Save as Excel
    df.to_excel(output_file + '.xlsx')
    
    print(f"Data saved to {output_file}.csv and {output_file}.xlsx")


def main():
    parser = argparse.ArgumentParser(description='Scrape mattress review data from specified URLs')
    parser.add_argument('--urls', nargs='+', required=True, help='List of URLs to scrape')
    parser.add_argument('--output', default='mattress_comparison', help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    print(f"Starting scrape of {len(args.urls)} URLs...")
    df = scrape_mattress_data(args.urls)
    save_data(df, args.output)
    
    # Print preview of the data
    print("\nPreview of scraped data:")
    print(df.head())
    print(f"\nTotal mattresses scraped: {df.shape[1]}")
    print(f"Total attributes found: {df.shape[0]}")


if __name__ == "__main__":
    main()