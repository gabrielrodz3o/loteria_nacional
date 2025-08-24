"""Web scraper for Dominican lottery results."""

import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Selenium imports (fallback)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LotteryResult:
    """Data class for lottery result."""
    fecha: date
    tipo_loteria: str
    primer_lugar: int
    segundo_lugar: int
    tercer_lugar: int
    fuente: str


class LotteryScraper:
    """Main scraper class for Dominican lottery results."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.scraping_user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Selenium driver (lazy initialization)
        self.driver = None
        
        # Target URLs
        self.urls = {
            'loteria_nacional': 'https://www.loterianacional.com.do/',
            'gana_mas': 'https://www.lndd.com.do/gana-mas/',
            'backup_url': 'https://www.lndd.com.do/resultados/',
        }
    
    def _init_selenium_driver(self) -> webdriver.Chrome:
        """Initialize Selenium WebDriver."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available. Install with: pip install selenium webdriver-manager")
        
        if not settings.selenium_enabled:
            raise ValueError("Selenium is disabled in settings")
        
        try:
            chrome_options = Options()
            
            if settings.selenium_headless:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={settings.scraping_user_agent}')
            
            # Install and setup ChromeDriver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            logger.info("Selenium WebDriver initialized successfully")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium driver: {e}")
            raise
    
    def _get_selenium_driver(self) -> webdriver.Chrome:
        """Get or create Selenium driver."""
        if self.driver is None:
            self.driver = self._init_selenium_driver()
        return self.driver
    
    def _close_selenium_driver(self):
        """Close Selenium driver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("Selenium driver closed")
            except Exception as e:
                logger.warning(f"Error closing Selenium driver: {e}")
    
    def _extract_numbers_from_text(self, text: str) -> List[int]:
        """Extract lottery numbers from text."""
        # Look for patterns like "23-45-67", "23 45 67", "23, 45, 67"
        patterns = [
            r'\b(\d{1,2})[-\s,]+(\d{1,2})[-\s,]+(\d{1,2})\b',
            r'\b(\d{2})\s+(\d{2})\s+(\d{2})\b',
            r'(\d{2})-(\d{2})-(\d{2})',
            r'(\d{2}),\s*(\d{2}),\s*(\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                numbers = [int(match.group(i)) for i in range(1, 4)]
                # Validate numbers are in range
                if all(0 <= num <= 99 for num in numbers):
                    return numbers
        
        # Fallback: look for any 2-digit numbers
        numbers = re.findall(r'\b\d{2}\b', text)
        if len(numbers) >= 3:
            try:
                result = [int(num) for num in numbers[:3]]
                if all(0 <= num <= 99 for num in result):
                    return result
            except ValueError:
                pass
        
        return []
    
    def _parse_date_from_text(self, text: str) -> Optional[date]:
        """Parse date from text."""
        # Common Spanish date patterns
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
            r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})',
        ]
        
        # Spanish month names
        months = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if 'de' in pattern:
                        # Spanish format: "15 de enero de 2024"
                        day = int(match.group(1))
                        month_name = match.group(2).lower()
                        year = int(match.group(3))
                        month = months.get(month_name)
                        if month:
                            return date(year, month, day)
                    else:
                        # Numeric format
                        g1, g2, g3 = match.groups()
                        if len(g1) == 4:  # YYYY-MM-DD
                            return date(int(g1), int(g2), int(g3))
                        elif len(g3) == 4:  # DD-MM-YYYY or MM-DD-YYYY
                            # Assume DD-MM-YYYY for Dominican context
                            return date(int(g3), int(g2), int(g1))
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def scrape_loteria_nacional(self) -> Optional[LotteryResult]:
        """Scrape Lotería Nacional results."""
        try:
            logger.info("Scraping Lotería Nacional...")
            
            # Try requests first
            response = self.session.get(self.urls['loteria_nacional'], timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for result elements (adapt selectors based on actual website)
            result_selectors = [
                '.resultado-sorteo',
                '.numeros-ganadores',
                '.winning-numbers',
                '[class*="result"]',
                '[class*="numero"]'
            ]
            
            numbers = []
            result_date = date.today()  # Default to today
            
            for selector in result_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    
                    # Try to extract numbers
                    found_numbers = self._extract_numbers_from_text(text)
                    if found_numbers and len(found_numbers) == 3:
                        numbers = found_numbers
                        break
                    
                    # Try to extract date
                    found_date = self._parse_date_from_text(text)
                    if found_date:
                        result_date = found_date
                
                if numbers:
                    break
            
            # If no numbers found with requests, try Selenium
            if not numbers and settings.selenium_enabled:
                numbers, result_date = self._scrape_with_selenium(
                    self.urls['loteria_nacional'], 'Lotería Nacional'
                )
            
            if numbers and len(numbers) == 3:
                return LotteryResult(
                    fecha=result_date,
                    tipo_loteria='Lotería Nacional',
                    primer_lugar=numbers[0],
                    segundo_lugar=numbers[1],
                    tercer_lugar=numbers[2],
                    fuente=self.urls['loteria_nacional']
                )
            
            logger.warning("No valid numbers found for Lotería Nacional")
            return None
            
        except Exception as e:
            logger.error(f"Error scraping Lotería Nacional: {e}")
            return None
    
    def scrape_gana_mas(self) -> Optional[LotteryResult]:
        """Scrape Gana Más results."""
        try:
            logger.info("Scraping Gana Más...")
            
            response = self.session.get(self.urls['gana_mas'], timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for Gana Más specific elements
            gana_mas_selectors = [
                '.gana-mas-resultado',
                '.sorteo-gana-mas',
                '[class*="gana"]',
                '.resultado-230',  # 2:30 PM draw
            ]
            
            numbers = []
            result_date = date.today()
            
            for selector in gana_mas_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    
                    found_numbers = self._extract_numbers_from_text(text)
                    if found_numbers and len(found_numbers) == 3:
                        numbers = found_numbers
                        break
                    
                    found_date = self._parse_date_from_text(text)
                    if found_date:
                        result_date = found_date
                
                if numbers:
                    break
            
            # Selenium fallback
            if not numbers and settings.selenium_enabled:
                numbers, result_date = self._scrape_with_selenium(
                    self.urls['gana_mas'], 'Gana Más'
                )
            
            if numbers and len(numbers) == 3:
                return LotteryResult(
                    fecha=result_date,
                    tipo_loteria='Gana Más',
                    primer_lugar=numbers[0],
                    segundo_lugar=numbers[1],
                    tercer_lugar=numbers[2],
                    fuente=self.urls['gana_mas']
                )
            
            logger.warning("No valid numbers found for Gana Más")
            return None
            
        except Exception as e:
            logger.error(f"Error scraping Gana Más: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str, lottery_type: str) -> Tuple[List[int], date]:
        """Scrape using Selenium as fallback."""
        try:
            logger.info(f"Using Selenium for {lottery_type}")
            
            driver = self._get_selenium_driver()
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Give additional time for dynamic content
            time.sleep(3)
            
            # Get page source and parse
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Look for numbers in various elements
            all_text = soup.get_text()
            numbers = self._extract_numbers_from_text(all_text)
            
            # Try to find date
            result_date = self._parse_date_from_text(all_text) or date.today()
            
            return numbers[:3] if len(numbers) >= 3 else [], result_date
            
        except Exception as e:
            logger.error(f"Selenium scraping failed for {lottery_type}: {e}")
            return [], date.today()
    
    def scrape_backup_source(self) -> List[LotteryResult]:
        """Scrape from backup source that might have both lottery types."""
        try:
            logger.info("Scraping backup source...")
            
            response = self.session.get(self.urls['backup_url'], timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            # Look for multiple result containers
            result_containers = soup.find_all(['div', 'section', 'article'], 
                                            class_=re.compile(r'result|sorteo|loteria', re.I))
            
            for container in result_containers:
                text = container.get_text(strip=True)
                
                # Determine lottery type
                lottery_type = None
                if 'gana' in text.lower() and 'más' in text.lower():
                    lottery_type = 'Gana Más'
                elif 'nacional' in text.lower():
                    lottery_type = 'Lotería Nacional'
                
                if lottery_type:
                    numbers = self._extract_numbers_from_text(text)
                    result_date = self._parse_date_from_text(text) or date.today()
                    
                    if numbers and len(numbers) == 3:
                        results.append(LotteryResult(
                            fecha=result_date,
                            tipo_loteria=lottery_type,
                            primer_lugar=numbers[0],
                            segundo_lugar=numbers[1],
                            tercer_lugar=numbers[2],
                            fuente=self.urls['backup_url']
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error scraping backup source: {e}")
            return []
    
    def scrape_all_results(self, target_date: Optional[date] = None) -> List[LotteryResult]:
        """Scrape all available lottery results."""
        target_date = target_date or date.today()
        logger.info(f"Scraping all lottery results for {target_date}")
        
        results = []
        
        # Try individual scrapers
        loteria_result = self.scrape_loteria_nacional()
        if loteria_result and loteria_result.fecha == target_date:
            results.append(loteria_result)
        
        gana_mas_result = self.scrape_gana_mas()
        if gana_mas_result and gana_mas_result.fecha == target_date:
            results.append(gana_mas_result)
        
        # Try backup source if we don't have both results
        if len(results) < 2:
            backup_results = self.scrape_backup_source()
            for result in backup_results:
                if result.fecha == target_date:
                    # Check if we already have this lottery type
                    existing_types = [r.tipo_loteria for r in results]
                    if result.tipo_loteria not in existing_types:
                        results.append(result)
        
        logger.info(f"Found {len(results)} lottery results for {target_date}")
        return results
    
    def validate_result(self, result: LotteryResult) -> bool:
        """Validate a lottery result."""
        try:
            # Check date is reasonable
            if result.fecha > date.today() or result.fecha < date(2020, 1, 1):
                return False
            
            # Check numbers are in valid range
            numbers = [result.primer_lugar, result.segundo_lugar, result.tercer_lugar]
            if not all(0 <= num <= 99 for num in numbers):
                return False
            
            # Check lottery type is valid
            if result.tipo_loteria not in ['Lotería Nacional', 'Gana Más']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating result: {e}")
            return False
    
    def scrape_historical_results(self, start_date: date, end_date: date) -> List[LotteryResult]:
        """Scrape historical results for a date range."""
        logger.info(f"Scraping historical results from {start_date} to {end_date}")
        
        all_results = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                # Some websites have date-specific URLs
                # This is a simplified implementation
                daily_results = self.scrape_all_results(current_date)
                all_results.extend(daily_results)
                
                # Rate limiting
                time.sleep(2)
                
                current_date = date.fromordinal(current_date.toordinal() + 1)
                
            except Exception as e:
                logger.error(f"Error scraping historical data for {current_date}: {e}")
                current_date = date.fromordinal(current_date.toordinal() + 1)
                continue
        
        logger.info(f"Scraped {len(all_results)} historical results")
        return all_results
    
    def test_scraping(self) -> Dict[str, Any]:
        """Test scraping functionality."""
        logger.info("Testing scraping functionality...")
        
        test_results = {
            'timestamp': datetime.now(),
            'loteria_nacional': {'status': 'failed', 'result': None, 'error': None},
            'gana_mas': {'status': 'failed', 'result': None, 'error': None},
            'backup_source': {'status': 'failed', 'results': [], 'error': None},
            'selenium_available': SELENIUM_AVAILABLE,
            'selenium_enabled': settings.selenium_enabled
        }
        
        # Test Lotería Nacional
        try:
            result = self.scrape_loteria_nacional()
            if result:
                test_results['loteria_nacional'] = {
                    'status': 'success',
                    'result': result.__dict__,
                    'error': None
                }
            else:
                test_results['loteria_nacional']['status'] = 'no_data'
        except Exception as e:
            test_results['loteria_nacional']['error'] = str(e)
        
        # Test Gana Más
        try:
            result = self.scrape_gana_mas()
            if result:
                test_results['gana_mas'] = {
                    'status': 'success',
                    'result': result.__dict__,
                    'error': None
                }
            else:
                test_results['gana_mas']['status'] = 'no_data'
        except Exception as e:
            test_results['gana_mas']['error'] = str(e)
        
        # Test backup source
        try:
            results = self.scrape_backup_source()
            if results:
                test_results['backup_source'] = {
                    'status': 'success',
                    'results': [r.__dict__ for r in results],
                    'error': None
                }
            else:
                test_results['backup_source']['status'] = 'no_data'
        except Exception as e:
            test_results['backup_source']['error'] = str(e)
        
        return test_results
    
    def __del__(self):
        """Cleanup on destruction."""
        self._close_selenium_driver()
        if hasattr(self, 'session'):
            self.session.close()


# Global scraper instance
lottery_scraper = LotteryScraper()