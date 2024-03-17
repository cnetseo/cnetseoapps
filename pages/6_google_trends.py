import pandas as pd
import streamlit as st
import requests
import json
import time


serpapikey =  st.secrets['serpapi']["SERPAPIKEY"]

# Function to fetch google trends data
def fetch_google_trends_data(keywords, lookback_period):
    url = 'https://serpapi.com/search.json?'
    headers = []
    data = []

    for keyword in keywords:
        params = {
                "engine": "google_trends",
                "q": keyword,
                "data_type": "TIMESERIES",
                "date": lookback_period,
                "geo": "US",
                "api_key": serpapikey
        }

        try:
            response = requests.get(url, params=params)
            json_response = response.json()
            interest_data = json_response['interest_over_time']['timeline_data']
            row_data = {"Keyword": keyword}

            for data_point in interest_data:
                date = pd.to_datetime(data_point['date'])
                month_year = date.strftime('%Y-%m')
                value = data_point['value']

                if month_year not in row_data.keys():
                    row_data[month_year] = { 'sum': 0, 'count': 0 }

                row_data[month_year]['sum'] += value
                row_data[month_year]['count'] += 1

                if month_year not in headers:
                    headers.append(month_year)

            for month_year in row_data.keys():
                if month_year != "Keyword":
                    row_data[month_year] = row_data[month_year]['sum'] / row_data[month_year]['count']

            data.append(row_data)
            time.sleep(1)

        except Exception as e:
            st.write(f"Error processing keyword: {keyword} - Exception: {str(e)}")

    return headers, data

def main():
    # Streamlit app
    st.title("Google Trends Data Fetcher")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        lookback_period = st.selectbox("Select lookback period", ("now 7-d","today 3-m", "today 12-m"), index=0)

        if st.button("Fetch Google Trends Data"):
            keywords = df.iloc[:, 0].tolist()
            headers, data = fetch_google_trends_data(keywords, lookback_period)

            headers.sort()
            df = pd.DataFrame(data)
            df = df.set_index("Keyword")
            df = df[headers]

            st.dataframe(df)

if __name__ == "__main__":
    main()
