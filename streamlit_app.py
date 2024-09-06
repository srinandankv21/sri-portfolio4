import streamlit as st
import pandas as pd
import numpy as np
# Check the page query parameter
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
            # Additional K-Means settings (like number of clusters) can be added here
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
