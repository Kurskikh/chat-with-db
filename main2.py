import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

DB_URI = "sqlite:///database5.db"

# SQL query template
SQL_TEMPLATE = """
Исходя из схемы таблицы ниже, напишите SQL запрос, который ответит на вопрос пользователя.
{schema}

Question: {question}
SQL Query:
"""

# Response template in natural language
RESPONSE_TEMPLATE = """
Исходя из схемы таблицы ниже, вопроса, SQL запроса и SQL ответа, напишите ответ на естественном языке:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

# Creating templates
sql_prompt = ChatPromptTemplate.from_template(SQL_TEMPLATE)
response_prompt = ChatPromptTemplate.from_template(RESPONSE_TEMPLATE)

# Database connection
db = SQLDatabase.from_uri(DB_URI)

def get_schema(_):
    """Get table schema from the database."""
    return db.get_table_info()

def run_query(query):
    """Execute SQL query on the database."""
    return db.run(query)

# SQL query chain
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | ChatOpenAI().bind(stop="\nSQL Result:")
    | StrOutputParser()
)

# Complete chain for response
full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"])
    )
    | response_prompt
    | ChatOpenAI()
    | StrOutputParser()
)

def display_chat_history():
    """Display chat history."""
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

def main():
    """Main function to run the application."""
    st.title('SQL Query Helper')

    # Initialize or access the session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am your SQL assistant. How can I help you today?")]

    # User input and update chat history
    user_query = st.chat_input("Type your SQL query question here...")
    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Process user query and generate response
        try:
            response = full_chain.invoke({"question": user_query})
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.session_state.chat_history.append(AIMessage(content=error_message))

    # Always display chat history after processing the user query
    display_chat_history()

if __name__ == "__main__":
    main()