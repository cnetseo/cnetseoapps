import streamlit as st
import pandas as pd
import json
from datetime import datetime, timezone
import requests, zipfile, io
import base64

exclude_domain = "www.cnet.com"
api_key = st.secrets("dataforseoapikey")

# Function to extract timestamps from JSON data

def extract_timestamps(items):
    timestamps = []
    count = 0  # Initialize a counter
    for item in items:
        # Check if the item is not from the excluded domain
        if item.get('type') == 'organic' and item.get('timestamp') is not None and item.get('domain') != exclude_domain:
            timestamps.append(datetime.strptime(item.get('timestamp').replace(" ", ""), '%Y-%m-%d%H:%M:%S%z'))
            count += 1  # Increment the counter
            if count == 10:  # If we've extracted 10 timestamps
                break  # Break the loop
    return timestamps



def categorize_dates(dates):
    now = datetime.now(timezone.utc)  # Use timezone-aware datetime for current date
    differences = sorted([(now - date).days for date in dates])  # Sort differences in ascending order

    # Consider only the 5 smallest differences
    smallest_differences = differences[:5]

    average_smallest_difference = sum(smallest_differences) / len(smallest_differences)
    print(average_smallest_difference)
    average_difference = sum(differences) / len(differences)
    print(average_difference)

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

# ... (your import statements and function defs stay the same)

def getSERPInfo(keyword):
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
            timestamps = extract_timestamps(result['items'])
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
    st.title('Upload CSV')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        first_column = data.columns[0]  # Get the name of the first column
        df = pd.DataFrame(columns=['keyword', 'Ideal Refresh Cadence', 'Minimum Refresh Cadence'])
        for keyword in data[first_column]:  # Use the first column name here
            result_list = getSERPInfo(keyword)
            df = df.append(result_list, ignore_index=True)

        st.write(df)
        
        # Prepare CSV data for download
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Some bytes handling
        href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)  # Provide download link

if __name__ == "__main__":
    main()