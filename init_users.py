import sqlite3
import hashlib

# Connect to SQLite (creates file if it doesn't exist)
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
)
""")

# Add a test user
email = "test@example.com"
password = "test123"
password_hash = hashlib.sha256(password.encode()).hexdigest()

cursor.execute("INSERT OR IGNORE INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))

conn.commit()
conn.close()

print("✅ users.db created with default user.")
