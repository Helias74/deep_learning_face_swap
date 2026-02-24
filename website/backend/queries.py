import sqlite3

DB_PATH = "faceswap.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crée les tables si elles n'existent pas."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS renders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            result_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


# ── USERS ──

def sql_insert_user(email: str, password: str, name: str):
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO users (email, password, name) VALUES (?, ?, ?)",
        (email, password, name)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def sql_get_user_by_email(email: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row


def sql_get_user_by_id(user_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row

def sql_get_password_by_email(user_email: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (user_email,)).fetchone()
    conn.close()
    return row
