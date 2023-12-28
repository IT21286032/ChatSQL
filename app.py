import streamlit as st
from sqlalchemy import create_engine, inspect, text
from typing import Dict, Any
import openai
import pandas as pd
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import os

from llama_index import (
    ServiceContext,
    SQLDatabase,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.llms import OpenAI
from llama_index.indices.struct_store import NLSQLTableQueryEngine

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@contextmanager
def sqlite_connect(db_bytes):
    fp = Path(str(uuid4()))
    fp.write_bytes(db_bytes.getvalue())
    conn = sqlite3.connect(str(fp))

    try:
        yield conn
    finally:
        conn.close()
        fp.unlink()

def get_table_data(table_name, conn):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    return df


def load_to_llm(uploaded_file):
    if uploaded_file:
        # Read the content of the uploaded file
        file_content = uploaded_file.read()

        # Cache the SQLDatabase and ServiceContext
        @st.cache(allow_output_mutation=True)
        def create_sql_database_and_service_context(file_content):
            engine = create_engine(f"sqlite:///:memory:")  # Use an in-memory database
            sql_database = SQLDatabase(engine)
            llm2 = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
            service_context = ServiceContext.from_defaults(llm=llm2, embed_model="local")
            return sql_database, service_context, engine

        # Cache the inspector and connection separately
        @st.cache(allow_output_mutation=True)
        def create_inspector_and_connection(file_content):
            engine = create_engine(f"sqlite:///:memory:")  # Use an in-memory database
            inspector = inspect(engine)
            conn = sqlite3.connect(":memory:")  # Use an in-memory database
            return inspector, conn

        sql_database, service_context, engine = create_sql_database_and_service_context(file_content)
        inspector, conn = create_inspector_and_connection(file_content)

        return sql_database, service_context, engine, inspector, conn
    else:
        return None, None, None, None, None

class StreamlitChatPack(BaseLlamaPack):
    def __init__(
        self,
        page: str = "Natural Language to SQL Query",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.page = page

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        import streamlit as st

        st.set_page_config(
            page_title=f"{self.page}",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello. Ask me anything related to the database."}
            ]

        selected_table = None

        st.title(f"{self.page}💬")
        st.info(
            f"Explore Snowflake views with this AI-powered app. Pose any question and receive exact SQL queries.",
            icon="ℹ️",
        )

        uploaded_file = st.file_uploader("Upload your SQLite database file", type=["db", "sqlite"])
        conn = None
        sql_database = None  # Initialize sql_database outside the if block

        if uploaded_file:
            file_content = uploaded_file.read()
            
            with sqlite3.connect(":memory:") as temp_conn:
                sql_database, service_context, engine, inspector, _ = load_db_llm(file_content)
                conn = temp_conn

            # Sidebar for database schema viewer
            st.sidebar.markdown("## Database Schema Viewer")

            # Use the 'inspector' variable here instead of creating a new one
            if inspector:
                # Get list of tables in the database
                table_names = inspector.get_table_names()

                # Sidebar selection for tables
                selected_table = st.sidebar.selectbox("Select a Table", table_names)

        # Display the selected table
        if selected_table:
            df = get_table_data(selected_table, conn)
            st.sidebar.text(f"Data for table '{selected_table}':")
            st.sidebar.dataframe(df)

        # Close the connection
        if conn is not None:
            conn.close()

        # Sidebar Intro
        st.sidebar.markdown('## App Created By')
        st.sidebar.markdown("""
            Kajeevan Jeyachandran: 
            [Linkedin](https://www.linkedin.com/in/kajeevanjeyachandran/ """)

        st.sidebar.markdown('## Other Projects')
        st.sidebar.markdown("""
        """)

        st.sidebar.markdown('## Disclaimer')
        st.sidebar.markdown("""This application is for demonstration purposes only and may not cover all aspects of real-world data complexities. Please use it as a guide and not as a definitive source for decision-making.""")

        if "query_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["query_engine"] = NLSQLTableQueryEngine(
                sql_database=sql_database,
                synthesize_response=True,
                service_context=service_context
            )

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input(
            "Enter your natural language query about the database"
        ):  # Prompt for user input and save to chat history
            with st.chat_message("user"):
                st.write(prompt)
            add_to_message_history("user", prompt)

        # If the last message is not from the assistant, generate a new response
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.spinner():
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query("User Question:"+prompt+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    add_to_message_history("assistant", sql_query)
if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()

    
