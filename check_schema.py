from app.storage.database import connect

conn = connect()

tables = ['documents', 'ingestion_jobs', 'deletion_operations', 'recovery_events']
for t in tables:
    print(f"\n=== {t} ===")
    try:
        schema = conn.execute(f"PRAGMA table_info({t})").fetchall()
        for s in schema:
            print(f"  {s[1]}: {s[2]} (notnull={s[3]})")
    except Exception as e:
        print(f"  Error: {e}")