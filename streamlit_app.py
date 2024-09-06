import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Check the page query parameter

def kmeans(data):
   
        # Filter numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check if there are any numeric columns
        if len(numeric_columns) > 0:
            st.write("Select the features to use for clustering:")
            
            # Multi-select box for choosing the features
            selected_features = st.multiselect("Features", numeric_columns, default=numeric_columns[:5])
            
            if selected_features:
                data = data[selected_features].dropna()
                
                # Normalize the data between 1 and 10
                data_normalized = ((data - data.min()) / (data.max() - data.min())) * 9 + 1
                
                # Slider for selecting number of clusters
                k = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=3)
                
                # K-means clustering functions
                def random_centroids(data, k):
                    centroids = []
                    for i in range(k):
                        centroid = data.apply(lambda x: float(x.sample()))
                        centroids.append(centroid)
                    return pd.concat(centroids, axis=1)
    
                def get_labels(data, centroids):
                    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
                    return distances.idxmin(axis=1)
    
                def new_centroids(data, labels, k):
                    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    
                def plot_clusters(data, labels, centroids):
                    pca = PCA(n_components=2)
                    data_2d = pca.fit_transform(data)
                    centroids_2d = pca.transform(centroids.T)
                    plt.figure()
                    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
                    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color='red', marker='X')
                    return plt
    
                # Run K-means clustering
                max_iterations = 100
                centroids = random_centroids(data_normalized, k)
                old_centroids = pd.DataFrame()
    
                images = []
                all_labels = None
    
                iteration = 1
                while iteration <= max_iterations and not centroids.equals(old_centroids):
                    old_centroids = centroids
                    labels = get_labels(data_normalized, centroids)
                    all_labels = labels  # Save the final labels after the last iteration
                    centroids = new_centroids(data_normalized, labels, k)
                    st.write(f"Iteration {iteration}")
                    fig = plot_clusters(data_normalized, labels, centroids)
                    st.pyplot(fig)
                    iteration += 1
                
               
                
                
                # Show final clustered data with cluster labels
                if all_labels is not None:
                    st.subheader('Clustered Data with Labels')
                    clustered_data = data.copy()
                    clustered_data['Cluster'] = all_labels.values
                    st.write(clustered_data)
            else:
                st.write("Please select at least one feature to proceed with clustering.")
        else:
            st.write("The dataset doesn't have any numeric columns for clustering.")

def fuzzycmeans(data):
   # Filter numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check if there are any numeric columns
        if len(numeric_columns) > 0:
            st.write("Select the features to use for clustering:")
            
            # Multi-select box for choosing the features
            selected_features = st.multiselect("Features", numeric_columns, default=numeric_columns[:5])
            
            if selected_features:
               # Normalize the data
               data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1
               
               # Function to initialize random fuzzy membership matrix
               def initialize_membership_matrix(n_samples, k):
                   membership_matrix = np.random.rand(n_samples, k)
                   membership_matrix = membership_matrix / membership_matrix.sum(axis=1, keepdims=True)
                   return membership_matrix
               
               # Function to compute centroids based on membership matrix
               def compute_centroids(data, membership_matrix, m):
                   um = membership_matrix ** m
                   centroids = (um.T @ data) / um.sum(axis=0)[:, None]
                   return centroids
               
               # Function to update the membership matrix
               def update_membership_matrix(data, centroids, m):
                   distances = np.zeros((data.shape[0], centroids.shape[0]))
                   
                   for i, centroid in enumerate(centroids):
                       distances[:, i] = np.linalg.norm(data - centroid, axis=1)
                   
                   # Avoid division by zero
                   distances = np.fmax(distances, np.finfo(np.float64).eps)
                   
                   new_membership_matrix = np.zeros_like(distances)
                   for i in range(centroids.shape[0]):
                       denominator = (distances[:, i][:, None] / distances) ** (2 / (m - 1))
                       new_membership_matrix[:, i] = 1 / denominator.sum(axis=1)
                   
                   return new_membership_matrix
               
               # Function to run Fuzzy C-Means clustering
               def fuzzy_c_means(data, k, m, max_iterations):
                   n_samples = data.shape[0]
                   membership_matrix = initialize_membership_matrix(n_samples, k)
                   for iteration in range(max_iterations):
                       centroids = compute_centroids(data, membership_matrix, m)
                       new_membership_matrix = update_membership_matrix(data, centroids, m)
                       
                       # Stop if membership matrix converges
                       if np.linalg.norm(new_membership_matrix - membership_matrix) < 1e-5:
                           break
                       membership_matrix = new_membership_matrix
                   
                   return centroids, membership_matrix
               
               # Streamlit interface
               st.title("Fuzzy C-Means Clustering")
               
               # Streamlit UI inputs
               k = st.slider("Number of Clusters (k)", 2, 10, 3)
               m = st.slider("Fuzziness Parameter (m)", 1.1, 2.5, 2.0, step=0.1)
               max_iterations = st.slider("Max Iterations", 50, 300, 100)
               
               # Run Fuzzy C-Means clustering
               centroids, membership_matrix = fuzzy_c_means(data.to_numpy(), k, m, max_iterations)
               
               # Assign each player to the cluster with the highest membership value
               labels = np.argmax(membership_matrix, axis=1)
               
               # Plot the clusters using PCA for dimensionality reduction
               pca = PCA(n_components=2)
               data_2d = pca.fit_transform(data)
               centroids_2d = pca.transform(centroids)
               
               fig, ax = plt.subplots()
               scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
               ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='x', color='red', s=100, label='Centroids')
               ax.set_title("Fuzzy C-Means Clustering")
               ax.legend()
               
               st.pyplot(fig)
        else:
            st.write("The dataset doesn't have any numeric columns for clustering.")
    
query_params = st.experimental_get_query_params()

# If the user navigates to the next page
if query_params.get("page") == ["next"]:
    st.title("Welcome to the Data clustering secton!")
    st.header("Perform K-means and Fuzzy C means Clustering")
    # Allow user to upload their data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file as a dataframe
        user_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(user_data)

        # Ask user which clustering method they would like to use
        clustering_method = st.radio(
            "Which clustering method would you like to use?",
            ("K-Means", "Fuzzy C-Means")
        )

        # Based on the selection, show additional instructions or options
        if clustering_method == "K-Means":
            st.write("You selected K-Means clustering.")
            kmeans(user_data)
        elif clustering_method == "Fuzzy C-Means":
            st.write("You selected Fuzzy C-Means clustering.")
            # Additional Fuzzy C-Means settings can be added here
else:
# Display the main page (data generation page)
    # Title of the app
    st.title("Custom Data Generator")
    
    # Ask user to define number of columns
    num_columns = st.number_input("Enter the number of columns", min_value=1, value=3)
    
    # Initialize an empty dictionary to store column details
    columns_data = {}
    
    # Loop through each column and ask for name and range
    for i in range(num_columns):
        col_name = st.text_input(f"Enter name for column {i + 1}", value=f"Column_{i + 1}")
        col_min = st.number_input(f"Enter minimum value for {col_name}", value=0)
        col_max = st.number_input(f"Enter maximum value for {col_name}", value=100)
        
        # Store the min and max values for each column
        columns_data[col_name] = (col_min, col_max)
    
    # Slider to define the number of rows
    num_rows = st.slider("Select number of rows", min_value=100, max_value=30000, value=300)
    
    # Generate the data based on user input
    def generate_data(columns_data, num_rows):
        data = {}
        for col, (min_val, max_val) in columns_data.items():
            data[col] = np.random.randint(min_val, max_val + 1, num_rows)
        return pd.DataFrame(data)
    
    # Create the dataframe
    if st.button("Generate Data"):
        df = generate_data(columns_data, num_rows)
        st.write(f"Generated Data with {num_rows} rows:")
        st.dataframe(df)
    
        # Option to download the dataframe
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='generated_data.csv', mime='text/csv')
    # Insert a button to navigate to the next page
    st.subheader('Perform clustering on the data')
    if st.button("Next Page"):
        st.experimental_set_query_params(page="next")
