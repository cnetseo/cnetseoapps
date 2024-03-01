import streamlit as st
import pandas as pd
import json
from datetime import datetime, timezone
import requests, zipfile, io
import base64
import time


api_key = st.secrets["dataforseoapikey"]["api_key"]


# Function to extract timestamps from JSON data

def extract_timestamps(items, exclude_domains):
    timestamps = []
    count = 0  
    for item in items:
        if item.get('type') == 'organic' and item.get('timestamp') is not None and item.get('domain') not in exclude_domains:  # Change here
            timestamps.append(datetime.strptime(item.get('timestamp').replace(" ", ""), '%Y-%m-%d%H:%M:%S%z'))
            count += 1  
            if count == 10:  
                break  
    return timestamps



def categorize_dates(dates):
    if len(dates) < 3:  # Check if there are less than 3 dates
        return "NA", "NA"  # Return "NA" for both ideal and minimum refresh cadence

    now = datetime.now(timezone.utc)  # Use timezone-aware datetime for current date
    differences = sorted([(now - date).days for date in dates])  # Sort differences in ascending order

    # Consider only the 5 smallest differences
    smallest_differences = differences[:5]

    average_smallest_difference = sum(smallest_differences) / len(smallest_differences)
    average_difference = sum(differences) / len(differences)

    def categorize(average):
        if average < 4.5:
            return 'daily'
        elif 4.5 <= average < 10.5:
            return 'weekly'
        elif 10.5 <= average < 18:
            return 'biweekly'
        elif 18 <= average < 51:
            return 'monthly'
        elif 51 <= average < 76:
            return 'bimonthly'
        elif 76 <= average < 116:
            return 'quarterly'
        elif 116 <= average < 200:
            return 'semi annually'
        else:
            return 'yearly'

    return categorize(average_smallest_difference), categorize(average_difference)


def getSERPInfo(keyword,exclude_domain):
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    payload = f"[{{\"keyword\":\"{keyword}\", \"location_code\":2826, \"language_code\":\"en\", \"device\":\"desktop\", \"os\":\"windows\", \"depth\":100}}]"
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    data_list = []
    if response_data["status_code"] == 20000:
        for result in response_data["tasks"][0]["result"]:
            print(result['items'])
            timestamps = extract_timestamps(result['items'],exclude_domain)
            smallest_cat, avg_cat = categorize_dates(timestamps)
            csv_data = {
                'keyword': keyword,
                'Ideal Refresh Cadence': smallest_cat,
                'Minimum Refresh Cadence': avg_cat
            }
            data_list.append(csv_data)
    else:
        print("error. Code: %d Message: %s" % (response_data["status_code"], response_data["status_message"]))
    return data_list

def main():
    st.title('RV Refresh Script')
    st.write('Please upload a CSV file with a list of singular keywords in the first column. This script will process the keywords and generate an output CSV file.')

    exclude_domains = st.text_input('Enter the domains to exclude, separated by commas', '')  
    exclude_domains = [domain.strip() for domain in exclude_domains.split(',')]  # Split the input into a list of domains

    # rest of the code...
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        first_column = data.columns[0]  # Get the name of the first column
        df = pd.DataFrame(columns=['keyword', 'Ideal Refresh Cadence', 'Minimum Refresh Cadence'])
        progress_text = st.empty()
        my_bar = st.progress(0)
        for i, keyword in enumerate(data[first_column]): 
            print(keyword) 
            result_list = getSERPInfo(keyword, exclude_domains) 
            print(result_list) 
            df = df.append(result_list, ignore_index=True)
           

            percent_complete = (i + 1) / len(data[first_column])  # Calculate the percentage of keywords processed
            my_bar.progress(percent_complete)  # Update the progress bar
            

        st.write(df)  # Move this line out of the loop
        progress_text.text('Operation complete.')

        # Prepare CSV data for download
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Some bytes handling
        href = f'<a href="data:file/csv;base64,{b64}" download="refresh_cadence.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)  # Provide download link

if __name__ == "__main__":
    main()