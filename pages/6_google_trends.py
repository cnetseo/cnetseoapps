import pandas as pd
import streamlit as st
import requests
import json
import time

serpapikey =  st.secrets['serpapi']["SERPAPIKEY"]

def next_month(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months[(months.index(month) + 1) % 12]

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
            print(interest_data)
            row_data = {"Keyword": keyword}
            #print(row_data)

            for data_point in interest_data:
                date_format = '%Y-%m'
                # Extract value based on the lookback period
                if lookback_period == 'today 3-m':
                    value = data_point['values'][0]['extracted_value']
                    date = pd.to_datetime(data_point['date'])
                    month_year = date.strftime(date_format)
                    date_list = [month_year]

                elif lookback_period == 'today 12-m':
                    value = data_point['values'][0]['extracted_value']
                    #print(value)
                    
                    # For 'today 12-m', we have a date range. We'll split the value between the two months.
                    date_str = data_point['date'].replace('\u2009', ' ')
                    start_date_str, end_date_str = date_str.split(' – ')
                    print(f"this is the {start_date_str} and the {end_date_str}")
                    start_month, start_day = start_date_str.split()
                    end_day, end_year = end_date_str.split(', ')
                    start_date = pd.to_datetime(f'{start_month} {start_day}, {end_year.strip()}')
                    end_date = pd.to_datetime(f'{start_month if int(end_day) > int(start_day) else next_month(start_month)} {end_day}, {end_year.strip()}')
                    
                    date_list = list(pd.date_range(start_date, end_date, freq='M').strftime(date_format))
                    if end_date.strftime(date_format) not in date_list:
                        date_list.append(end_date.strftime(date_format))
                else:
                    value = data_point['value']
                    date = pd.to_datetime(data_point['date'])
                    month_year = date.strftime(date_format)
                    date_list = [month_year]

                for month_year in date_list:
                    if month_year not in row_data.keys():
                        row_data[month_year] = { 'sum': 0, 'count': 0 }

                    row_data[month_year]['sum'] += value/len(date_list)
                    row_data[month_year]['count'] += 1

                    if month_year not in headers:
                        headers.append(month_year)

            for month_year in row_data.keys():
                if month_year != "Keyword":
                    row_data[month_year] = row_data[month_year]['sum'] / row_data[month_year]['count']

            data.append(row_data)
            time.sleep(1)

        except Exception as e:
            print(f"Error processing keyword: {keyword} - Exception: {str(e)}")
    print(headers,data)
    return headers, data

import base64

def main():
    # Streamlit app
    st.title("Google Trends Data Fetcher")

    option = st.radio("Choose how to input keywords", ("Upload CSV", "Enter manually"))
    keywords = []

    if option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            keywords = df.iloc[:, 0].tolist()
    else:
        keywords_input = st.text_input("Enter keywords separated by commas")
        if keywords_input:
            keywords = [keyword.strip() for keyword in keywords_input.split(",")]

    if keywords:
        lookback_period = st.selectbox("Select lookback period", ("now 7-d","today 3-m", "today 12-m"), index=0)

        if st.button("Fetch Google Trends Data"):
            headers, data = fetch_google_trends_data(keywords, lookback_period)

            headers.sort()
            df = pd.DataFrame(data)
            df = df.set_index("Keyword")
            df = df[headers]

            # Add a final column that averages all the row values
            df['Average'] = df.mean(axis=1)

            st.dataframe(df)

            # Convert dataframe to CSV and allow user to download
            csv = df.to_csv(index=True)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()