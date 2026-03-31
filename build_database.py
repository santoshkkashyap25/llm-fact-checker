# build_database.py
import pandas as pd
from core.vector_db import vector_db
from core.data_scraper import data_scraper
from config import FACTS_CSV_PATH, SCRAPE_ENABLED
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Builds the vector database from scraped or existing facts.
    """
    print("=" * 60)
    print("Starting Database Build Process")
    print("=" * 60)
    
    try:
        # Step 1: Check if we should scrape new data
        if SCRAPE_ENABLED:
            print("\n[1/3] Scraping fresh data from sources...")
            facts_df = data_scraper.scrape_all_sources()
            data_scraper.save_to_csv(facts_df)
            print(f"[OK] Scraped {len(facts_df)} facts")
        else:
            print("\n[1/3] Loading existing data from CSV...")
            if not FACTS_CSV_PATH.exists():
                print("[!] CSV not found. Generating sample data...")
                facts_df = pd.DataFrame(data_scraper.scrape_sample_facts())
                data_scraper.save_to_csv(facts_df)
            else:
                facts_df = pd.read_csv(FACTS_CSV_PATH)
            print(f"[OK] Loaded {len(facts_df)} facts")
        
        # Step 2: Validate data
        print("\n[2/3] Validating data...")
        if "statement" not in facts_df.columns:
            raise ValueError("CSV must contain a 'statement' column.")
        
        statements = facts_df["statement"].dropna().tolist()
        if len(statements) == 0:
            raise ValueError("No valid statements found in data.")
        
        print(f"[OK] Validated {len(statements)} statements")
        
        # Step 3: Build vector index
        print("\n[3/3] Building FAISS vector index...")
        vector_db.build_and_save(statements, metadata=facts_df.to_dict('records'))
        
        print("\n" + "=" * 60)
        print("[OK] Database build completed successfully!")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  - Total facts indexed: {len(statements)}")
        print(f"  - Index location: {vector_db.index_path}")
        print(f"  - Ready for queries!\n")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n[ERROR] Error: {e}")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n[ERROR] Error: {e}")
    except Exception as e:
        logger.exception("Unexpected error during build")
        print(f"\n[ERROR] Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()