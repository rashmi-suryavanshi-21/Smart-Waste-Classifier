import sqlite3
import pandas as pd

# queries.sql ko ';' se todkar har query alag chalao
text = open('database/queries.sql').read()
queries = [q.strip() for q in text.split(';') if q.strip()]

conn = sqlite3.connect('database/waste.db')
for i, q in enumerate(queries, start=1):
    print(f'\n--- Query {i} ---')
    print(pd.read_sql_query(q, conn).to_string(index=False))
conn.close()
