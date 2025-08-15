# src/core/database.py
import sqlite3

def init_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        correction_id TEXT NOT NULL,
        action TEXT NOT NULL,
        original_text TEXT NOT NULL,
        corrected_text TEXT NOT NULL,
        suggested_text TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()