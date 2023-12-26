import streamlit as st
from sqlalchemy import create_engine, inspect, text
from typing import Dict, Any

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.llms import OpenAI
import openai
import os
import pandas as pd

from llama_index.llms.palm import PaLM

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import sqlite3

from llama_index import SQLDatabase, ServiceContext
from llama_index.indices.struct_store import NLSQLTableQueryEngine

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


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

        st.title(
            f"{self.page}💬"
        )
        st.info(
            f"Explore Snowflake views with this AI-powered app. Pose any question and receive exact SQL queries.",
            icon="ℹ️",
        )

        uploaded_file = st.file_uploader("Upload your SQLite database file", type=["db", "sqlite"])

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(message)
        def get_table_data(table_name, conn):
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            return df
        @st.cache(allow_output_mutation=True)
        def load_db_llm(uploaded_file):
            sql_database = None
            service_context = None
            engine = None
        
            if uploaded_file:
                engine = create_engine(f"sqlite:///{uploaded_file}")
                sql_database = SQLDatabase(engine)  # Include all tables
        
            llm2 = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
            service_context = ServiceContext.from_defaults(llm=llm2, embed_model="local")
        
            return sql_database, service_context, engine


            


        # Sidebar for database schema viewer
        st.sidebar.markdown("## Database Schema Viewer")

        # Create an inspector object
        inspector = inspect(engine)

        # Get list of tables in the database
        table_names = inspector.get_table_names()

        # Sidebar selection for tables
        selected_table = st.sidebar.selectbox("Select a Table", table_names)

        
        conn = sqlite3.connect(uploaded_file)
    
        # Display the selected table
        if selected_table:
            df = get_table_data(selected_table, conn)
            st.sidebar.text(f"Data for table '{selected_table}':")
            st.sidebar.dataframe(df)
    
        # Close the connection
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

        # If last message is not from assistant, generate a new response
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.spinner():
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query("User Question:"+prompt+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    # st.write(response.response)
                    add_to_message_history("assistant", sql_query)

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
