import sqlite3
import os

def init_folder_db():
    conn = sqlite3.connect("storage.db")
    cursor = conn.cursor()

    # Folders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS folders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE
    )
    """)

    # Files table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        folder_path TEXT,
        FOREIGN KEY (folder_path) REFERENCES folders(path)
    )
    """)

    # Ensure root folder exists
    cursor.execute("INSERT OR IGNORE INTO folders (path) VALUES ('root')")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_folder_db()
    print("✅ storage.db initialized with folder and file tables.")
