import os
import csv
import streamlit as st
import langchain
import pandas as pd
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

openai_api_key = st.secrets["openai"]["openai_api_key"]

def analyzeOutput(df, request):
    # Create engine for an in-memory SQLite database
    engine = create_engine('sqlite:///:memory:')
    df.to_sql("keywords", engine, index=False)

    db = SQLDatabase(engine=engine)

    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    answer = agent_executor.invoke({"input": request})
    return answer


def main():
    # Streamlit code
    st.title('Chat with CSV')

    # Load OpenAI API Key from secrets
    

    # User input
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    input_request = st.text_input("Enter your request")

    # Execute and display the output when the 'Analyze' button is clicked
    if st.button('Analyze'):
        if uploaded_file is not None and input_request:
            df = pd.read_csv(uploaded_file)
            output = analyzeOutput(df, input_request)
            st.write(output)
        else:
            st.write('Please upload a CSV file and enter a request.')

if __name__ == "__main__":
    main()