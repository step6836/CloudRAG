"""
investor_scraper.py
-------------------
Automated scraper for earnings call transcripts from investor relations pages.
Handles both HTML and PDF formats. Updates rolling 4-quarter window.
"""

import requests
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
from pathlib import Path
from datetime import datetime
import time
import re
import sys
from typing import Dict, List, Optional

# Simple console logging (no file logs)
class SimpleLogger:
    @staticmethod
    def info(msg): 
        print(f"INFO: {msg}")
    
    @staticmethod
    def error(msg): 
        print(f"ERROR: {msg}", file=sys.stderr)

logger = SimpleLogger()

class TranscriptScraper:
    """Scrapes earnings call transcripts from company investor relations pages."""
    
    # Company-specific IR page URLs
    COMPANY_URLS = {
        "salesforce": "https://investor.salesforce.com/financials/default.aspx",
        "microsoft": "https://www.microsoft.com/en-us/investor/earnings",
        "amazon": "https://ir.aboutamazon.com/quarterly-results/default.aspx",
        "adobe": "https://www.adobe.com/investor-relations.html",
        "snowflake": "https://investors.snowflake.com/financials/quarterly-results/default.aspx",
        "oracle": "https://investor.oracle.com/financial-reporting/quarterly-results/default.aspx",
        "workday": "https://investor.workday.com/financial-information/quarterly-results/default.aspx"
    }
    
    def __init__(self, output_dir: str = None, pdf_dir: str = None):
        # Default to data folder if no paths provided
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = str(project_root / "data" / "transcripts")
        if pdf_dir is None:
            project_root = Path(__file__).parent.parent
            pdf_dir = str(project_root / "data" / "raw_pdfs")
            
        self.output_dir = Path(output_dir)
        self.pdf_dir = Path(pdf_dir)
        # Create pdf dir but not output dir (will create per-company folders)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # User agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_all_companies(self, force_update: bool = False):
        """Scrape transcripts for all companies."""
        logger.info("Starting scraper for all companies...")
        
        results = {}
        for company, url in self.COMPANY_URLS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping {company.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                transcripts = self.scrape_company(company, url, force_update)
                results[company] = {
                    "success": True,
                    "transcripts": len(transcripts),
                    "files": transcripts
                }
                logger.info(f"SUCCESS: {company}: {len(transcripts)} transcripts scraped")
                
                # Be polite.
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"{company}: Failed - {str(e)}")
                results[company] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def scrape_company(self, company: str, base_url: str, force_update: bool = False) -> List[str]:
        """
        Scrape transcripts for a specific company.
        Returns list of saved file paths.
        """
        logger.info(f"Fetching IR page: {base_url}")
        
        try:
            response = requests.get(base_url, headers=self.headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch IR page: {e}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links that might be transcripts
        transcript_links = self._find_transcript_links(soup, base_url)
        logger.info(f"Found {len(transcript_links)} potential transcript links")
        
        saved_files = []
        for i, link_info in enumerate(transcript_links[:4], 1):  # Get last 4 quarters
            url = link_info['url']
            quarter = link_info['quarter']
            
            # Skip if quarter is unknown
            if quarter.lower() == 'unknown':
                logger.info(f"Skipping - couldn't extract quarter info from link")
                continue
            
            logger.info(f"Downloading {quarter}: {url}")
            
            try:
                if url.endswith('.pdf'):
                    text = self._download_and_extract_pdf(url, company, quarter)
                else:
                    text = self._scrape_html_transcript(url)
                
                if text:
                    # Create company subfolder
                    company_folder = self.output_dir / company.lower()
                    company_folder.mkdir(parents=True, exist_ok=True)
                    
                    clean_quarter = quarter.lower().replace(' ', '_')
                    clean_quarter = re.sub(r'fy20(\d{2})', r'fy\1', clean_quarter)
                    clean_quarter = re.sub(r'(q\d)fy20(\d{2})', r'\1_fy\2', clean_quarter)
                    
                    filename = f"{company.lower()}_{clean_quarter}.txt"
                    filepath = company_folder / filename
                    
                    if filepath.exists() and not force_update:
                        logger.info(f"Skipping {quarter} (already exists)")
                        continue
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    saved_files.append(str(filepath))
                    logger.info(f"SUCCESS: Saved: {filename}")
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to process {quarter}: {e}")
                continue
        
        return saved_files
    
    def _find_transcript_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """
        Find transcript links on IR page.
        This is company-specific and may need customization.
        """
        links = []
        
        keywords = ['transcript', 'earnings', 'call', 'quarterly', 'results']
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text().lower()
            
            # Check if link is relevant
            if any(keyword in text or keyword in href.lower() for keyword in keywords):
                # Make absolute URL
                if href.startswith('http'):
                    full_url = href
                else:
                    full_url = requests.compat.urljoin(base_url, href)
                
                quarter = self._extract_quarter_info(text + ' ' + href)
                
                links.append({
                    'url': full_url,
                    'text': text,
                    'quarter': quarter or 'Unknown'
                })
        
        # Sort by quarter (most recent first)
        links.sort(key=lambda x: x['quarter'], reverse=True)
        
        return links
    
    def _extract_quarter_info(self, text: str) -> Optional[str]:
        """Extract quarter information from text."""
        # Look for patterns like "Q1 2026", "Q4 FY25", "q3 fy2025" etc.
        patterns = [
            r'Q[1-4]\s*(?:FY)?\s*20\d{2}',
            r'Q[1-4]\s*(?:Fiscal)?\s*20\d{2}',
            r'(First|Second|Third|Fourth)\s*Quarter\s*20\d{2}',
            r'q[1-4]\s*fy\s*20\d{2}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _download_and_extract_pdf(self, url: str, company: str, quarter: str) -> str:
        """Download PDF and extract text."""
        logger.info(f"  Extracting PDF: {url}")
        
        # Download PDF
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        # Save PDF
        pdf_filename = f"{company}_{quarter.lower().replace(' ', '_')}.pdf"
        pdf_path = self.pdf_dir / pdf_filename
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                text = self._clean_text(text)
                return text
                
        except Exception as e:
            logger.error(f"pdfplumber failed, trying PyPDF2: {e}")
            
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    text = self._clean_text(text)
                    return text
            except Exception as e2:
                logger.error(f"PyPDF2 also failed: {e2}")
                return ""
    
    def _scrape_html_transcript(self, url: str) -> str:
        """Scrape transcript from HTML page."""
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'Page \d+', '', text)
        
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        
        return text.strip()


def main():
    """Run the scraper."""
    scraper = TranscriptScraper()
    
    # Scrape all companies
    results = scraper.scrape_all_companies(force_update=False)
    
    # Print summary
    print("\n" + "="*60)
    print("SCRAPING SUMMARY")
    print("="*60)
    for company, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        count = result.get('transcripts', 0)
        print(f"{status}: {company.upper()}: {count} transcripts")
    print("="*60)
    
    # Note which companies are blocked
    blocked = [c for c, r in results.items() if not r['success'] and '403' in str(r.get('error', ''))]
    if blocked:
        print(f"\nNOTE: These companies blocked the scraper (403 Forbidden):")
        for c in blocked:
            print(f"  - {c.upper()}")
        print("Use manual download or try different user agents/proxies")


if __name__ == "__main__":
    main()