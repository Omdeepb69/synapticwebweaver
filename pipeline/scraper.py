import time
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import json
from datetime import datetime
from urllib.robotparser import RobotFileParser

from config.settings import (
    USER_AGENT, SCRAPING_DELAY, MAX_PAGES_PER_DOMAIN,
    RAW_DATA_DIR, ROBOTS_TXT_RESPECT
)
from utils.logger import logger
from utils.helpers import get_domain, normalize_text, generate_document_id, is_valid_url

class WebScraper:
    def __init__(self, use_selenium=False):
        """
        Initialize the web scraper.
        
        Args:
            use_selenium (bool): Whether to use Selenium for JavaScript-rendered pages
        """
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.use_selenium = use_selenium
        self.driver = None
        self.robot_parsers = {}  # Cache for robot.txt parsers
        
        if use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Set up Selenium WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"user-agent={USER_AGENT}")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
    
    def _can_fetch(self, url):
        """Check if the URL can be fetched according to robots.txt rules."""
        if not ROBOTS_TXT_RESPECT:
            return True
            
        domain = get_domain(url)
        
        # Get or create robot parser for this domain
        if domain not in self.robot_parsers:
            robots_url = f"https://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
                self.robot_parsers[domain] = parser
            except Exception as e:
                logger.warning(f"Could not read robots.txt for {domain}: {e}")
                return True
        
        return self.robot_parsers[domain].can_fetch(USER_AGENT, url)
    
    def fetch_page(self, url):
        """
        Fetch a web page and return its content.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            tuple: (html_content, status_code) or (None, error_code) on failure
        """
        if not is_valid_url(url):
            logger.error(f"Invalid URL: {url}")
            return None, 400
            
        if not self._can_fetch(url):
            logger.info(f"URL not allowed by robots.txt: {url}")
            return None, 403
            
        try:
            if self.use_selenium:
                self.driver.get(url)
                time.sleep(SCRAPING_DELAY)  # Wait for JavaScript to render
                html_content = self.driver.page_source
                return html_content, 200
            else:
                response = self.session.get(url, timeout=10)
                time.sleep(SCRAPING_DELAY)  # Respect websites by adding delay
                return response.text, response.status_code
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None, 500
    
    def extract_text_from_html(self, html_content):
        """
        Extract clean text from HTML content.
        
        Args:
            html_content (str): HTML content
            
        Returns:
            str: Extracted text
        """
        if not html_content:
            return ""
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style", "header", "footer", "nav"]):
            element.decompose()
        
        # Get text and normalize
        text = soup.get_text(separator=' ')
        return normalize_text(text)
    
    def extract_links(self, url, html_content):
        """
        Extract all links from a web page.
        
        Args:
            url (str): URL of the page
            html_content (str): HTML content
            
        Returns:
            list: List of extracted full URLs
        """
        if not html_content:
            return []
            
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            absolute_link = urljoin(url, link)
            
            # Filter out non-http(s) links
            if absolute_link.startswith(('http://', 'https://')):
                links.append(absolute_link)
                
        return links
    
    def crawl(self, start_url, max_pages=None, domain_restrict=True):
        """
        Crawl from a starting URL and collect web content.
        
        Args:
            start_url (str): URL to start crawling from
            max_pages (int): Maximum number of pages to crawl
            domain_restrict (bool): Whether to restrict crawling to the starting domain
            
        Returns:
            list: List of documents (dict with URL, text, metadata)
        """
        if not max_pages:
            max_pages = MAX_PAGES_PER_DOMAIN
            
        start_domain = get_domain(start_url)
        to_visit = [start_url]
        visited = set()
        domain_pages_count = {start_domain: 0}
        documents = []
        
        while to_visit and len(documents) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
                
            visited.add(url)
            domain = get_domain(url)
            
            # Domain restriction check
            if domain_restrict and domain != start_domain:
                continue
                
            # Domain page limit check
            if domain_pages_count.get(domain, 0) >= MAX_PAGES_PER_DOMAIN:
                continue
                
            logger.info(f"Crawling: {url}")
            
            html_content, status_code = self.fetch_page(url)
            if not html_content or status_code != 200:
                continue
                
            # Extract text
            text = self.extract_text_from_html(html_content)
            if not text or len(text.split()) < 50:  # Skip pages with little content
                continue
                
            # Save the document
            doc_id = generate_document_id(url, text)
            document = {
                'id': doc_id,
                'url': url,
                'domain': domain,
                'text': text,
                'crawl_date': datetime.now().isoformat(),
                'length': len(text)
            }
            documents.append(document)
            
            # Save raw data
            self._save_raw_data(doc_id, url, html_content, text)
            
            domain_pages_count[domain] = domain_pages_count.get(domain, 0) + 1
            
            # Extract links for further crawling
            links = self.extract_links(url, html_content)
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)
        
        return documents
    
    def _save_raw_data(self, doc_id, url, html_content, extracted_text):
        """Save raw HTML and extracted text to disk."""
        domain = get_domain(url)
        domain_dir = os.path.join(RAW_DATA_DIR, domain.replace('.', '_'))
        os.makedirs(domain_dir, exist_ok=True)
        
        # Save HTML
        html_path = os.path.join(domain_dir, f"{doc_id}_raw.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save metadata and text
        meta_path = os.path.join(domain_dir, f"{doc_id}_meta.json")
        meta_data = {
            'id': doc_id,
            'url': url,
            'crawl_date': datetime.now().isoformat(),
            'domain': domain
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)
        
        # Save extracted text
        text_path = os.path.join(domain_dir, f"{doc_id}_text.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
    
    def close(self):
        """Close resources."""
        if self.driver:
            self.driver.quit()