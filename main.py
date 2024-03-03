import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Шаблон для SQL запроса
sql_template = """
Исходя из схемы таблицы ниже, напишите SQL запрос, который ответит на вопрос пользователя.
{schema}

Question: {question}
SQL Query:
"""

# Шаблон для ответа на естественном языке
response_template = """
Исходя из схемы таблицы ниже, вопроса, SQL запроса и SQL ответа, напишите ответ на естественном языке:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

# Create templates
sql_prompt = ChatPromptTemplate.from_template(sql_template)
response_prompt = ChatPromptTemplate.from_template(response_template)

# Database connection
db_uri = "sqlite:///database5.db"
db = SQLDatabase.from_uri(db_uri)

# Function to get table schema
def get_schema(_):
    return db.get_table_info()

# Function to execute SQL query
def run_query(query):
    return db.run(query)

# Создание цепочки для SQL запроса
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | ChatOpenAI().bind(stop="\nSQL Result:")
    | StrOutputParser()
)

# Создание полной цепочки для ответа
full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"])
    )
    | response_prompt
    | ChatOpenAI()
    | StrOutputParser()
)

# Функция для запуска цепочки с вводом из командной строки
def main():
    import sys
    question = " ".join(sys.argv[1:])  # Получение вопроса из аргументов командной строки
    result = full_chain.invoke({"question": question})
    print(result)

if __name__ == "__main__":
    while True:  # Бесконечный цикл для постоянного запроса вопросов
        try:
            # Запрос ввода вопроса от пользователя
            question = input("Введите вопрос (или 'exit' для выхода): ")
            if question.lower() == 'exit':  # Проверка на команду выхода
                print("Выход из программы.")
                break  # Выход из цикла, если пользователь ввел 'exit'
            # Вызов цепочки для получения и вывода ответа
            result = full_chain.invoke({"question": question})
            print("Результат:", result)
        except Exception as e:
            print("Произошла ошибка:", e)
            # Вы можете решить, хотите ли вы продолжить или выйти из цикла в случае ошибки