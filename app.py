import streamlit as st
from sqlalchemy import create_engine, inspect
import sqlite3
import os
import pandas as pd

def connect_to_database(file_path):
    try:
        engine = create_engine(f"sqlite:///{file_path}")
        connection = engine.connect()
        inspector = inspect(connection)
        return connection, inspector.get_table_names()
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None, []

def get_table_data(table_name, conn):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    return df

# User Upload
uploaded_file = st.file_uploader("Upload SQLite Database", type=["db", "sqlite"])

# Connect to Database
if uploaded_file:
    # Save the uploaded file temporarily
    with st.spinner("Uploading and connecting to the database..."):
        file_path = f"temp_database_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        connection, table_names = connect_to_database(file_path)

        # Remove the temporary file after use
        os.remove(file_path)

        if connection:
            st.success("Connected to the database!")
            st.write(f"Tables in the database: {table_names}")

            # Sidebar for database schema viewer
            st.sidebar.markdown("## Database Schema Viewer")

            # Display the selected table
            selected_table = st.sidebar.selectbox("Select a Table", table_names)
            
            if selected_table:
                df = get_table_data(selected_table, connection)
                st.sidebar.text(f"Data for table '{selected_table}':")
                st.sidebar.dataframe(df)
            
            # Close the connection when done
            connection.close()

else:
    st.warning("Please upload an SQLite database file.")

# Continue with the rest of your chat-based interface code...
# (The code below this point remains unchanged from your original code)
# ...
