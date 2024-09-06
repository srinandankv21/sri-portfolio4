import streamlit as st
import pandas as pd
import numpy as np
# Check the page query parameter
query_params = st.experimental_get_query_params()

# If the user navigates to the next page
if query_params.get("page") == ["next"]:
    st.title("Welcome to the Data clustering secton!")
    st.header("Perform K-means and Fuzzy C means Clustering")
else
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
