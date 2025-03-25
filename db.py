import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    conn = pyodbc.connect(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={os.getenv("AZURE_SQL_SERVER")};'
        f'DATABASE={os.getenv("AZURE_SQL_DATABASE")};'
        f'UID={os.getenv("AZURE_SQL_USER")};'
        f'PWD={os.getenv("AZURE_SQL_PASSWORD")}'
    )
    return conn
