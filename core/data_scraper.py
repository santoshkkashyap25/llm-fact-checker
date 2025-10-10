# core/data_scraper.py
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging
from config import PIB_RSS_URL, FACTS_CSV_PATH

logger = logging.getLogger(__name__)

class DataScraper:
    """Scrapes verified facts from trusted sources"""
    
    def __init__(self):
        self.sources = {
            'pib': PIB_RSS_URL,
        }
        self.scraped_facts = []
    
    def scrape_pib_rss(self) -> List[Dict[str, str]]:
        """Scrape PIB RSS feed for government announcements"""
        try:
            feed = feedparser.parse(PIB_RSS_URL)
            facts = []
            
            for entry in feed.entries[:50]:  # Limit to recent 50
                # Extract clean statement from title and summary
                statement = self._clean_text(entry.get('title', ''))
                if len(statement.split()) > 5:  # Only substantial statements
                    facts.append({
                        'statement': statement,
                        'source': 'PIB India',
                        'url': entry.get('link', ''),
                        'date': entry.get('published', datetime.now().isoformat()),
                        'category': 'government_announcement'
                    })
            
            logger.info(f"Scraped {len(facts)} facts from PIB RSS")
            return facts
            
        except Exception as e:
            logger.error(f"Error scraping PIB RSS: {e}")
            return []
    
    def scrape_sample_facts(self) -> List[Dict[str, str]]:
        """Generate sample facts for testing (fallback)"""
        sample_facts = [
            {
                'statement': 'India achieved 241 GW peak power demand on June 9, 2025 with zero shortage',
                'source': 'PIB India',
                'url': 'https://pib.gov.in/sample',
                'date': '2025-06-10',
                'category': 'energy'
            },
            {
                'statement': 'The Ayushman Bharat scheme provides health coverage of up to Rs 5 lakh per family per year',
                'source': 'PIB India',
                'url': 'https://pib.gov.in/sample',
                'date': '2025-01-15',
                'category': 'healthcare'
            },
            {
                'statement': 'IREDA was granted Navratna status by the Government of India',
                'source': 'PIB India',
                'url': 'https://pib.gov.in/sample',
                'date': '2025-03-20',
                'category': 'business'
            },
            {
                'statement': 'India has 28 states and 8 union territories as of 2024',
                'source': 'Government of India',
                'url': 'https://india.gov.in',
                'date': '2024-01-01',
                'category': 'geography'
            },
            {
                'statement': 'The Digital India initiative was launched in 2015',
                'source': 'PIB India',
                'url': 'https://pib.gov.in/sample',
                'date': '2015-07-01',
                'category': 'technology'
            }
        ]
        return sample_facts
    
    def _clean_text(self, text: str) -> str:
        """Clean scraped text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove common prefixes
        prefixes = ['Press Release:', 'PIB:', 'Government of India:']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text
    
    def scrape_all_sources(self) -> pd.DataFrame:
        """Scrape all configured sources and return DataFrame"""
        all_facts = []
        
        # Try PIB RSS
        pib_facts = self.scrape_pib_rss()
        all_facts.extend(pib_facts)
        
        # Add sample facts if scraping failed
        if len(all_facts) == 0:
            logger.warning("No facts scraped, using sample data")
            all_facts = self.scrape_sample_facts()
        
        df = pd.DataFrame(all_facts)
        return df
    
    def save_to_csv(self, df: pd.DataFrame):
        """Save scraped facts to CSV"""
        df.to_csv(FACTS_CSV_PATH, index=False)
        logger.info(f"Saved {len(df)} facts to {FACTS_CSV_PATH}")

data_scraper = DataScraper()
