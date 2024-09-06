import streamlit as st
import pandas as pd
import numpy as np
# Check the page query parameter

def kmeans(uploaded_file):
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")
    
        # Display the first few rows of the dataset
        st.write(data.head())
    
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
    else:
        st.write("Please upload a CSV file to proceed.")
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
            kmeans(uploaded_file)
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
