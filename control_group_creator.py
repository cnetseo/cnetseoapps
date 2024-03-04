import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the data
df = pd.read_csv('page_selection.csv')

# Select the columns to use for comparison
columns = ['Total Clicks', 'Average Position', 'Click Through Rate (CTR)']

# Normalize the data
df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()

# Get the test group
test_group_ids = ['5257a123-31ac-47e7-82da-d978fbefb4b3',
                  '5e89a80e-8cac-4b81-9ce4-c4763b1b95d3',
                  '0fc5ee29-c687-48ea-a89f-818fe670043d']

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

# Print only the first 3 content ids
print(control_group['Content ID'].head(3))