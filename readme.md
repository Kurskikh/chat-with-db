# Chat with a MySQL\SQL Database Using Python and LangChain

## Описание
Проект "Chat with a MySQL Database Using Python and LangChain" представляет собой пример кода, который позволяет пользователям взаимодействовать со своей базой данных MySQL напрямую через интерфейс ChatGPT. Это решение идеально подходит для быстрого получения информации из базы данных, выполнения запросов и анализа данных с помощью удобного текстового интерфейса.

## Предварительные требования
Прежде чем приступить к установке и запуску проекта, убедитесь, что у вас установлен Python 3.6 или выше. Также необходимо иметь установленный MySQL сервер и доступ к базе данных.

## Установка
Чтобы начать работу с проектом, выполните следующие шаги:

1. Клонируйте репозиторий или скачайте исходный код проекта на ваш локальный компьютер.
2. Создайте виртуальное окружение для изоляции зависимостей проекта:
    ```
    python -m venv venv
    ```
3. Активируйте виртуальное окружение:
    - Для Windows:
        ```
        venv\Scripts\activate
        ```
    - Для macOS и Linux:
        ```
        source venv/bin/activate
        ```
4. Установите необходимые зависимости:
    ```
    pip install -r requirements.txt
    ```

## Запуск
После установки всех зависимостей запустите приложение с помощью следующей команды:
    ```
    streamlit run main2.py
    ```
После запуска приложения откроется веб-интерфейс, через который вы сможете отправлять запросы к вашей базе данных MySQL, используя естественный язык.

## Использование

После запуска веб-приложения пользователь встречается с интерфейсом, где он может вводить свои вопросы, связанные с SQL-запросами, в текстовое поле. Вот шаги, как это работает:

1. Введите ваш вопрос на естественном языке, который связан с вашей базой данных, например: "Какое среднее количество заказов за месяц?".

2. Приложение использует LangChain для формирования SQL-запроса из вашего вопроса. Это достигается через взаимодействие с ChatGPT от OpenAI, который преобразует вопрос пользователя в SQL-запрос.

3. Затем приложение выполняет сформированный SQL-запрос на вашей базе данных.

4. После получения результатов запроса, приложение снова использует ChatGPT для формулирования ответа на естественном языке, основываясь на данных, полученных из базы данных.

5. Ответ отображается в интерфейсе веб-приложения, позволяя пользователю увидеть как сам SQL-запрос, так и ответ на его изначальный вопрос в понятной форме.

6. Весь диалог сохраняется в истории чата, что позволяет пользователю отслеживать свои запросы и полученные ответы.

### Пример использования

После того как вы зададите вопрос через интерфейс, например: "Каково общее количество пользователей в БД?", приложение автоматически сгенерирует SQL-запрос, выполнит его на сервере базы данных и предоставит вам ответ, например: "В базе данных 150 пользователей".

Этот процесс делает взаимодействие с базой данных более интуитивно понятным и доступным для пользователей без глубоких знаний SQL.

## Техническая реализация

Проект использует следующие ключевые компоненты для работы:

- `streamlit` для создания веб-интерфейса.
- `langchain_core` и `langchain_openai` для интеграции с OpenAI и выполнения запросов в естественном языке.
- `SQLDatabase` из `langchain_community.utilities` для взаимодействия с базой данных.
- `dotenv` для управления конфигурацией и безопасным хранением параметров подключения к базе данных.

Обратите внимание, что для работы приложения необходимо наличие файла `.env` в корне проекта, содержащего строку подключения к вашей базе данных в переменной `OPENAI_API_KEY="ключи OpenAI"`.

