

import streamlit as st
from sqlalchemy import create_engine, inspect, text
from typing import Dict, Any

from sqlalchemy import create_engine, inspect
import pandas as pd
import sqlite3

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

import openai
import os
import pandas as pd
import sqlite3

from llama_index import (
    SQLDatabase,
    ServiceContext,
    VectorStoreIndex,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.indices.struct_store import NLSQLTableQueryEngine

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

class StreamlitChatPack:
    def __init__(self, run_from_main=False):
        self.run_from_main = run_from_main

    def run(self):
        st.set_page_config(
            page_title="Natural Language to SQL Query",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello. Ask me anything related to the database."}
            ]

        st.title("Natural Language to SQL Query 💬")
        st.info(
            "Explore Snowflake views with this AI-powered app. Pose any question and receive exact SQL queries.",
            icon="ℹ️",
        )

        self.sidebar()
        self.main()
    def get_tables_from_db(self, uploaded_file):
        if uploaded_file:
        tables = self.get_tables_from_db(uploaded_file)
        selected_table = st.sidebar.selectbox("Select a Table", tables)

        if selected_table:
            db_file_path = uploaded_file.name
            engine = create_engine(f"sqlite:///{db_file_path}")
            conn = engine.connect()
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            st.sidebar.text(f"Data for table '{selected_table}':")
            st.sidebar.dataframe(df)
            conn.close()


    def sidebar(self):
        st.sidebar.header("Navigation")
        uploaded_file = st.sidebar.file_uploader("Upload a SQLite database", type=["db", "sqlite"])
        
        if uploaded_file:
            # Display tables in the sidebar if a file is uploaded
            tables = self.get_tables_from_db(uploaded_file)
            st.sidebar.subheader("Tables in the Database")
            st.sidebar.write(tables)

    def main(self):
        uploaded_file = st.file_uploader("Upload your SQLite database file", type=["db", "sqlite"])

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(message)

        def get_table_data(table_name, conn):
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            return df

        @st.cache_resource
        def load_db_llm(database_file_path):
            if uploaded_file:
                engine = create_engine(f"sqlite:///{database_file_path}")
            else:
                engine = create_engine("sqlite:///ecommerce_platform1.db")

            sql_database = SQLDatabase(engine)
            llm2 = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
            service_context = ServiceContext.from_defaults(llm=llm2, embed_model="local")

            return sql_database, service_context, engine

        if uploaded_file is not None:
            db_file_path = uploaded_file.name
        else:
            db_file_path = "ecommerce_platform1.db"

        sql_database, service_context, engine = load_db_llm(db_file_path)

        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        selected_table = st.sidebar.selectbox("Select a Table", table_names)

        db_file = db_file_path
        conn = sqlite3.connect(db_file)

        if selected_table:
            df = get_table_data(selected_table, conn)
            st.sidebar.text(f"Data for table '{selected_table}':")
            st.sidebar.dataframe(df)

        conn.close()

        st.sidebar.markdown('## Database File Information')
        st.sidebar.text(f"Uploaded Database File: {uploaded_file.name}" if uploaded_file else "No file uploaded")

        st.sidebar.markdown('## App Created By')
        st.sidebar.markdown("""
        Harshad Suryawanshi: 
        [Linkedin](https://www.linkedin.com/in/harshadsuryawanshi/), [Medium](https://harshadsuryawanshi.medium.com/), [Twitter](https://twitter.com/HarshadSurya1c)
        """)

        #st.sidebar.markdown('## Other Projects')
       # st.sidebar.markdown("""
       # - [Pokemon Go! Inspired AInimal GO! - Multimodal RAG App](https://www.linkedin.com/posts/harshadsuryawanshi_llamaindex-ai-deeplearning-activity-7134632983495327744-M7yy)
       # - [Building My Own GPT4-V with PaLM and Kosmos](https://lnkd.in/dawgKZBP)
       # - [AI Equity Research Analyst](https://ai-eqty-rsrch-anlyst.streamlit.app/)
       # - [Recasting "The Office" Scene](https://blackmirroroffice.streamlit.app/)
       # - [Story Generator](https://appstorycombined-agaf9j4ceit.streamlit.app/)
       # """)

        st.sidebar.markdown('## Disclaimer')
        st.sidebar.markdown("""
        This application is for demonstration purposes only and may not cover all aspects of real-world data complexities. Please use it as a guide and not as a definitive source for decision-making.
        """)

        if "query_engine" not in st.session_state:
            st.session_state["query_engine"] = NLSQLTableQueryEngine(
                sql_database=sql_database,
                synthesize_response=True,
                service_context=service_context
            )

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Enter your natural language query about the database"):
            with st.chat_message("user"):
                st.write(prompt)
            add_to_message_history("user", prompt)

        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.spinner():
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query("User Question:" + prompt + ". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    add_to_message_history("assistant", sql_query)

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
