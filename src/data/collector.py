"""
Aqarmap Egypt Real Estate Scraper & Data Generator

Two modes:
1. Live scraping from aqarmap.com.eg (requires working selectors)
2. Sample data generation for testing/demo
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
import re
from urllib.parse import urljoin
from datetime import datetime


class AqarmapScraper:
    """
    Scraper for Aqarmap.com.eg
    
    Note: Website structures change. If scraping fails, use generate_sample_data().
    """
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        self.base_url = "https://aqarmap.com.eg"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def get_property_urls_from_search(self, search_url, max_pages=3):
        """Extract property URLs from search results"""
        property_urls = []
        
        for page in range(1, max_pages + 1):
            page_url = f"{search_url}?page={page}"
            print(f"   Fetching page {page}: {page_url}")
            
            try:
                response = self.session.get(page_url, timeout=15)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                all_links = soup.find_all('a', href=True)
                
                for link in all_links:
                    href = link.get('href')
                    if href and ('/for-sale/' in href or '/ad/' in href or '/property/' in href):
                        if href.startswith('/'):
                            full_url = urljoin(self.base_url, href)
                        else:
                            full_url = href
                        full_url = full_url.split('?')[0]
                        if full_url not in property_urls and self.base_url in full_url:
                            property_urls.append(full_url)
                
                time.sleep(2)
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        return list(set(property_urls))
    
    def scrape_property(self, url):
        """Scrape detailed property info from URL"""
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            page_text = soup.get_text()
            
            title = self._extract_title(soup)
            price = self._extract_price(soup, page_text)
            if price == 0:
                return None
            
            return {
                'url': url,
                'title': title,
                'price': price,
                'description': self._extract_description(soup),
                'location': self._extract_location(soup, page_text),
                'bedrooms': self._extract_bedrooms(page_text),
                'bathrooms': self._extract_bathrooms(page_text),
                'area_sqm': self._extract_area(page_text),
                'property_type': self._detect_property_type(title, page_text),
                'furnishing': self._detect_furnishing(page_text),
                'images': [],  # Skipping images for speed
                'source': 'aqarmap',
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    def _extract_title(self, soup):
        title_tag = soup.find('h1')
        if title_tag:
            return title_tag.text.strip()
        elif soup.find('title'):
            return soup.find('title').text.strip()
        return "Unknown"
    
    def _extract_price(self, soup, page_text):
        # Try selectors
        for selector in ['span[class*="price"]', 'div[class*="price"]', '.price']:
            elem = soup.select_one(selector)
            if elem:
                match = re.search(r'[\d,]+', elem.text)
                if match:
                    return int(match.group().replace(',', ''))
        # Regex fallback
        match = re.search(r'([\d,]+)\s*(?:EGP|جنيه|LE)', page_text)
        if match:
            return int(match.group(1).replace(',', ''))
        return 0
    
    def _extract_description(self, soup):
        for selector in ['div[class*="description"]', 'div[class*="details"]', '.description']:
            elem = soup.select_one(selector)
            if elem and len(elem.text.strip()) > 50:
                return elem.text.strip()[:2000]
        return ""
    
    def _extract_location(self, soup, page_text):
        for selector in ['span[class*="location"]', 'div[class*="location"]', '.location']:
            elem = soup.select_one(selector)
            if elem:
                return elem.text.strip()[:100]
        match = re.search(r'location[:\s]+([A-Za-z\s]+)', page_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:100]
        return "Cairo"
    
    def _extract_bedrooms(self, text):
        match = re.search(r'(\d+)\s*(?:bed|bedroom|br|غرفة)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 2
    
    def _extract_bathrooms(self, text):
        match = re.search(r'(\d+)\s*(?:bath|bathroom|ba|حمام)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 1
    
    def _extract_area(self, text):
        match = re.search(r'(\d+)\s*(?:m²|sqm|مساحة)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 100
    
    def _detect_property_type(self, title, text):
        combined = (title + " " + text).lower()
        if 'villa' in combined or 'فيلا' in combined:
            return 'Villa'
        elif 'townhouse' in combined or 'تاون' in combined:
            return 'Townhouse'
        elif 'studio' in combined or 'ستوديو' in combined:
            return 'Studio'
        elif 'duplex' in combined or 'دوبلكس' in combined:
            return 'Duplex'
        return 'Apartment'
    
    def _detect_furnishing(self, text):
        text = text.lower()
        if 'furnished' in text or 'مفروش' in text:
            return 'Furnished'
        elif 'semi' in text or 'نصف' in text:
            return 'Semi-furnished'
        return 'Unfurnished'
    
    def scrape_all(self, search_urls, max_pages_per_search=1, max_properties=50):
        """Scrape multiple search URLs"""
        all_properties = []
        all_urls = []
        
        print("\nCollecting URLs...")
        for search_url in search_urls:
            urls = self.get_property_urls_from_search(search_url, max_pages_per_search)
            all_urls.extend(urls)
        
        all_urls = list(set(all_urls))[:max_properties]
        print(f"Found {len(all_urls)} unique URLs")
        
        print("\nScraping properties...")
        for idx, url in enumerate(all_urls):
            print(f"[{idx+1}/{len(all_urls)}] {url[:70]}")
            data = self.scrape_property(url)
            if data:
                all_properties.append(data)
                print(f"   OK: {data['price']:,.0f} EGP, {data['area_sqm']}sqm")
            time.sleep(1)
        
        return all_properties
    
    def save_data(self, properties, filename=None):
        """Save to JSON"""
        if filename is None:
            filename = f"aqarmap_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(properties, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(properties)} to {filepath}")
        return filepath


def generate_sample_data(n=1000, output_dir='data/raw', save=True):
    """
    Generate realistic synthetic property data for development/testing.
    
    Args:
        n: Number of properties to generate
        output_dir: Output directory
        save: Whether to save to file
    
    Returns:
        filepath if save=True, else list of dicts
    """
    import random
    
    property_types = ['Apartment', 'Villa', 'Townhouse', 'Studio', 'Duplex']
    furnishing_options = ['Unfurnished', 'Semi-furnished', 'Furnished']
    locations = ['Cairo', 'Alexandria', 'Giza', 'New Cairo', 'Sheikh Zayed', '6th October']
    
    listings = []
    for i in range(n):
        ptype = random.choice(property_types)
        
        # Area based on type
        if ptype == 'Studio':
            area = random.randint(30, 60)
            bedrooms = 0
        elif ptype == 'Apartment':
            area = random.randint(60, 200)
            bedrooms = random.randint(1, 4)
        elif ptype == 'Villa':
            area = random.randint(150, 400)
            bedrooms = random.randint(3, 6)
        elif ptype == 'Townhouse':
            area = random.randint(120, 300)
            bedrooms = random.randint(2, 5)
        else:  # Duplex
            area = random.randint(100, 250)
            bedrooms = random.randint(2, 4)
        
        bathrooms = random.randint(max(1, bedrooms-2), bedrooms+1) if bedrooms > 0 else 1
        location = random.choice(locations)
        furnishing = random.choice(furnishing_options)
        
        # Base price per sqm
        base_ppsm = 8000 + random.randint(-2000, 3000)
        
        # Calculate base price
        price = base_ppsm * area
        price += bedrooms * 150000
        price += bathrooms * 100000
        
        # Multipliers
        if ptype == 'Villa':
            price *= 1.6
        elif ptype == 'Duplex':
            price *= 1.3
        elif ptype == 'Studio':
            price *= 0.85
            
        if furnishing == 'Furnished':
            price *= 1.15
        elif furnishing == 'Semi-furnished':
            price *= 1.08
            
        # Location premium/discount
        if location in ['New Cairo', 'Sheikh Zayed']:
            price *= 1.2
        elif location == 'Alexandria':
            price *= 1.1
        elif location == '6th October':
            price *= 0.9
            
        # Random variation
        price = int(price * random.uniform(0.85, 1.15))
        
        listings.append({
            'title': f"{ptype} in {location}",
            'price': price,
            'area_sqm': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_type': ptype,
            'furnishing': furnishing,
            'location': location,
            'description': f"Beautiful {ptype.lower()} in {location}. {bedrooms} bedrooms, {bathrooms} bathrooms, {area} sqm. {furnishing}.",
            'url': f"https://example.com/property/{i}",
            'scraped_at': datetime.now().isoformat(),
            'source': 'sample'
        })
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'sample_{n}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(listings, f, ensure_ascii=False, indent=2)
        print(f"Generated {n} samples -> {filepath}")
        return filepath
    
    return listings


if __name__ == "__main__":
    # Quick demo
    print("Generating sample data...")
    generate_sample_data(n=100)
    
    # To scrape live:
    # scraper = AqarmapScraper()
    # urls = ["https://aqarmap.com.eg/en/for-sale/apartment/cairo/"]
    # properties = scraper.scrape_all(urls, max_pages=1, max_properties=20)
    # scraper.save_data(properties)
