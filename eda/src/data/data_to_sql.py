import pandas as pd
import sqlite3 as sql3

# Load Data
df = pd.read_csv('../../data/processed/df_processed.csv')

# SQLite Database
connection = sql3.connect('../../data/database.db')
cursor = connection.cursor()
cursor.execute("""CREATE TABLE rain_in_australia("")""")

# Transform to sql
df.to_sql(name='rain_in_australia', con=connection, if_exists='replace')
connection.close()