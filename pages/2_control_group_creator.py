import streamlit as st
import pandas as pd
import numpy as np
from scikit-learn import NearestNeighbors


# Select the columns to use for comparison
columns = ['Total Clicks', 'Average Position', 'Click Through Rate (CTR)']

def load_data(file):
    df = pd.read_csv(file)
    return df

def find_control_group(df, test_group_ids):
    # Normalize the data
    df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()

    # Get the test group
    test_group = df[df['Content ID'].isin(test_group_ids)]

    # Get the rest of the data
    control_group_pool = df[~df['Content ID'].isin(test_group_ids)]

    # Fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(control_group_pool[columns])

    # Find the nearest neighbors for each test group member
    distances, indices = nbrs.kneighbors(test_group[columns])

    # Get the unique indices
    unique_indices = np.unique(indices)

    # Get the control group
    control_group = control_group_pool.iloc[unique_indices]
    return control_group['Content ID'].head(3)

def main():
    st.set_page_config(page_title="Control Group Creator", page_icon="📈")
    st.markdown("Control Group Creator")
    st.sidebar.header("Control Group Creator")
    st.title('RV Control Group Script')
    st.write('Please upload a CSV file with a list of singular keywords in the first column. This script will process the keywords and generate an output CSV file.')

    file = st.file_uploader("Upload CSV", type=['csv'])

    if file is not None:
        df = load_data(file)
        test_group_ids = st.text_input("Enter Test Group IDs (comma-separated):")
        test_group_ids = [id.strip() for id in test_group_ids.split(",")]
        
        if st.button("Find Control Group"):
            result = find_control_group(df, test_group_ids)
            st.write(result)

if __name__ == "__main__":
    main()