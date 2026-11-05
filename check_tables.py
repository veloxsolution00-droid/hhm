from app.storage.database import connect

conn = connect()
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t[0] for t in tables])