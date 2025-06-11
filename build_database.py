# build_database.py
import pandas as pd
from core.vector_db import vector_db
from config import FACTS_CSV_PATH

def main():
    """
    Reads the trusted facts from the CSV and builds the FAISS vector database.
    In a real system, this script would first fetch and process data from sources like the PIB RSS feed.
    """
    print("Starting database build process...")
    try:
        facts_df = pd.read_csv(FACTS_CSV_PATH)
        if "statement" not in facts_df.columns:
            raise ValueError("CSV must contain a 'statement' column.")
        
        statements = facts_df["statement"].dropna().tolist()
        vector_db.build_and_save(statements)
        
    except FileNotFoundError:
        print(f"Error: The fact file was not found at {FACTS_CSV_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()