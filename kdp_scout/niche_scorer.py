"""Niche opportunity scorecard for Amazon KDP keywords.

For a given keyword, searches Amazon for the top results and computes
a composite opportunity score based on:
  - Average BSR of top 10 results (higher = less competitive)
  - Average review count of top 10 (lower = easier to compete)
  - Estimated monthly revenue (validates demand exists)
  - Number of results with BSR > 100,000 (underserved indicator)

Also includes a category threshold finder that identifies categories
where the #20 book has a beatable BSR for your projected launch velocity.
"""

import logging
import re
import time

from bs4 import BeautifulSoup

from kdp_scout.http_client import fetch, get_browser_headers
from kdp_scout.rate_limiter import registry as rate_registry
from kdp_scout.config import Config
from kdp_scout.collectors.bsr_model import estimate_daily_sales, estimate_monthly_revenue

logger = logging.getLogger(__name__)

SEARCH_URL = 'https://www.amazon.com/s'

# Sponsored result markers
SPONSORED_MARKERS = [
    'AdHolder', 'sp-sponsored-result', 'puis-sponsored-label',
    's-sponsored-label', 'a-spacing-micro s-sponsored-label',
]


def score_niche(keyword, department='kindle', top_n=10):
    """Score a keyword niche by analyzing top Amazon search results.

    Searches Amazon for the keyword, scrapes the top N organic results,
    and computes an opportunity score.

    Args:
        keyword: Search keyword to analyze.
        department: Amazon department ('kindle' or 'books').
        top_n: Number of top results to analyze (default 10).

    Returns:
        Dict with:
            - keyword: the search keyword
            - opportunity_score: 0-100 composite score
            - results: list of scraped result dicts
            - metrics: aggregated metrics dict
            - recommendation: human-readable recommendation
        Returns None if search fails.
    """
    rate_registry.get_limiter('niche_search', rate=Config.SEARCH_PROBE_RATE_LIMIT)
    rate_registry.acquire('niche_search')

    dept_param = 'digital-text' if department == 'kindle' else 'stripbooks'
    params = {'k': keyword, 'i': dept_param}

    try:
        response = fetch(SEARCH_URL, params=params, headers=get_browser_headers())
    except Exception as e:
        logger.error(f'Search failed for "{keyword}": {e}')
        return None

    if response.status_code != 200:
        logger.warning(f'Search returned {response.status_code} for "{keyword}"')
        return None

    html = response.text
    if _is_captcha(html):
        logger.warning(f'CAPTCHA detected for "{keyword}"')
        return None

    results = _parse_search_results(html, top_n)

    if not results:
        logger.warning(f'No results parsed for "{keyword}"')
        return None

    # Compute metrics
    bsr_values = [r['bsr'] for r in results if r['bsr'] is not None]
    review_counts = [r['review_count'] for r in results if r['review_count'] is not None]
    prices = [r['price'] for r in results if r['price'] is not None]

    avg_bsr = sum(bsr_values) / len(bsr_values) if bsr_values else None
    avg_reviews = sum(review_counts) / len(review_counts) if review_counts else None
    avg_price = sum(prices) / len(prices) if prices else None

    # Count underserved indicators
    high_bsr_count = sum(1 for b in bsr_values if b > 100_000)
    low_review_count = sum(1 for r in review_counts if r < 50)

    # Estimate demand via average revenue
    avg_daily_sales = None
    avg_monthly_rev = None
    if avg_bsr and avg_price:
        avg_daily_sales = estimate_daily_sales(avg_bsr)
        avg_monthly_rev = estimate_monthly_revenue(avg_bsr, avg_price)

    metrics = {
        'result_count': len(results),
        'results_with_bsr': len(bsr_values),
        'results_with_reviews': len(review_counts),
        'avg_bsr': round(avg_bsr) if avg_bsr else None,
        'avg_reviews': round(avg_reviews, 1) if avg_reviews else None,
        'avg_price': round(avg_price, 2) if avg_price else None,
        'avg_daily_sales': round(avg_daily_sales, 2) if avg_daily_sales else None,
        'avg_monthly_revenue': round(avg_monthly_rev, 2) if avg_monthly_rev else None,
        'high_bsr_count': high_bsr_count,
        'low_review_count': low_review_count,
    }

    # Compute composite opportunity score (0-100)
    opportunity_score = _compute_opportunity_score(metrics, len(results))

    # Generate recommendation
    recommendation = _generate_recommendation(opportunity_score, metrics)

    return {
        'keyword': keyword,
        'opportunity_score': opportunity_score,
        'results': results,
        'metrics': metrics,
        'recommendation': recommendation,
    }


def score_niches_batch(keywords, department='kindle', top_n=10,
                       progress_callback=None):
    """Score multiple keyword niches and rank them by opportunity.

    Args:
        keywords: List of keyword strings to analyze.
        department: Amazon department.
        top_n: Number of top results per keyword.
        progress_callback: Optional callable(completed, total, keyword).

    Returns:
        List of niche score dicts, sorted by opportunity_score descending.
    """
    results = []
    total = len(keywords)

    for i, keyword in enumerate(keywords):
        result = score_niche(keyword, department=department, top_n=top_n)
        if result:
            results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, keyword)

    results.sort(key=lambda x: x['opportunity_score'], reverse=True)
    return results


def find_beatable_categories(keyword, target_daily_sales=5, department='kindle'):
    """Find categories where the #20 book is beatable with projected launch velocity.

    Searches Amazon for the keyword, extracts category paths from results,
    and estimates what BSR position #20 would correspond to.

    Args:
        keyword: Search keyword to analyze.
        target_daily_sales: Your projected daily sales during launch.
        department: Amazon department.

    Returns:
        List of category dicts with:
            - category: category name/path
            - bsr_at_20: estimated BSR at position #20
            - daily_sales_at_20: estimated daily sales at position #20
            - beatable: bool (True if your target velocity would place you in top 20)
            - headroom: how many more daily sales you'd have vs the #20 book
    """
    niche_data = score_niche(keyword, department=department, top_n=20)
    if not niche_data or not niche_data['results']:
        return []

    # Collect categories from results
    category_bsr_map = {}
    for result in niche_data['results']:
        categories = result.get('categories', [])
        bsr = result.get('bsr')
        if not bsr:
            continue
        for cat in categories:
            if cat not in category_bsr_map:
                category_bsr_map[cat] = []
            category_bsr_map[cat].append(bsr)

    # For each category, estimate the BSR threshold at position #20
    category_results = []
    for cat_name, bsr_list in category_bsr_map.items():
        # Sort BSR values (lower = better selling)
        bsr_list.sort()

        # Estimate BSR at position 20 by extrapolation
        if len(bsr_list) >= 5:
            # Use the worst (highest) BSR as an approximation of position ~#N
            # where N is the count of results we have
            bsr_at_20 = max(bsr_list)
            # Scale up if we have fewer than 20 data points
            if len(bsr_list) < 20:
                scale_factor = 20 / len(bsr_list)
                bsr_at_20 = int(bsr_at_20 * scale_factor ** 0.5)
        else:
            # Too few results, use a rough estimate
            bsr_at_20 = max(bsr_list) * 3 if bsr_list else None

        if bsr_at_20 is None:
            continue

        daily_sales_at_20 = estimate_daily_sales(bsr_at_20)
        beatable = target_daily_sales >= daily_sales_at_20
        headroom = round(target_daily_sales - daily_sales_at_20, 2)

        category_results.append({
            'category': cat_name,
            'bsr_at_20': bsr_at_20,
            'daily_sales_at_20': round(daily_sales_at_20, 2),
            'beatable': beatable,
            'headroom': headroom,
            'sample_size': len(bsr_list),
        })

    # Sort: beatable first, then by headroom descending
    category_results.sort(
        key=lambda x: (-int(x['beatable']), -x['headroom'])
    )
    return category_results


def _compute_opportunity_score(metrics, result_count):
    """Compute composite opportunity score from niche metrics.

    Scoring logic:
    - High avg BSR = less competitive = higher score (40% weight)
    - Low avg reviews = easier to compete = higher score (30% weight)
    - Reasonable revenue = validated demand = higher score (20% weight)
    - High % of high-BSR results = underserved = higher score (10% weight)

    Args:
        metrics: Aggregated metrics dict from score_niche.
        result_count: Number of results analyzed.

    Returns:
        Float score 0-100.
    """
    score = 0.0

    # BSR competition signal (40% weight)
    # BSR > 200k avg = very low competition, < 10k = very high competition
    avg_bsr = metrics.get('avg_bsr')
    if avg_bsr:
        if avg_bsr >= 200_000:
            bsr_score = 1.0
        elif avg_bsr >= 100_000:
            bsr_score = 0.8
        elif avg_bsr >= 50_000:
            bsr_score = 0.6
        elif avg_bsr >= 20_000:
            bsr_score = 0.4
        elif avg_bsr >= 10_000:
            bsr_score = 0.2
        else:
            bsr_score = 0.1
        score += bsr_score * 40

    # Review barrier (30% weight)
    # < 20 avg reviews = easy entry, > 500 = very hard
    avg_reviews = metrics.get('avg_reviews')
    if avg_reviews is not None:
        if avg_reviews < 20:
            review_score = 1.0
        elif avg_reviews < 50:
            review_score = 0.8
        elif avg_reviews < 100:
            review_score = 0.6
        elif avg_reviews < 250:
            review_score = 0.4
        elif avg_reviews < 500:
            review_score = 0.2
        else:
            review_score = 0.1
        score += review_score * 30

    # Demand validation (20% weight)
    # Some revenue is needed to confirm demand exists
    avg_monthly = metrics.get('avg_monthly_revenue')
    if avg_monthly is not None:
        if avg_monthly >= 500:
            demand_score = 1.0
        elif avg_monthly >= 200:
            demand_score = 0.8
        elif avg_monthly >= 50:
            demand_score = 0.6
        elif avg_monthly >= 10:
            demand_score = 0.4
        elif avg_monthly > 0:
            demand_score = 0.2
        else:
            demand_score = 0.0
        score += demand_score * 20

    # Underserved ratio (10% weight)
    high_bsr_count = metrics.get('high_bsr_count', 0)
    if result_count > 0:
        underserved_ratio = high_bsr_count / result_count
        score += underserved_ratio * 10

    return round(min(100.0, max(0.0, score)), 1)


def _generate_recommendation(score, metrics):
    """Generate human-readable recommendation from score and metrics."""
    avg_bsr = metrics.get('avg_bsr')
    avg_reviews = metrics.get('avg_reviews')
    avg_monthly = metrics.get('avg_monthly_revenue')

    if score >= 70:
        label = 'STRONG OPPORTUNITY'
        detail = 'Low competition with validated demand. Prioritize this niche.'
    elif score >= 50:
        label = 'MODERATE OPPORTUNITY'
        detail = 'Competitive but winnable with good positioning.'
    elif score >= 30:
        label = 'CHALLENGING'
        detail = 'Established competition. Requires strong differentiation.'
    else:
        label = 'AVOID'
        detail = 'Highly competitive or insufficient demand.'

    parts = [f'{label}: {detail}']

    if avg_bsr and avg_bsr < 20_000:
        parts.append(f'Avg BSR {avg_bsr:,} indicates strong existing sales.')
    if avg_reviews and avg_reviews > 200:
        parts.append(f'Avg {avg_reviews:.0f} reviews creates a high entry barrier.')
    if avg_monthly and avg_monthly < 20:
        parts.append('Low estimated revenue — demand may be insufficient.')

    return ' '.join(parts)


def _parse_search_results(html, top_n=10):
    """Parse Amazon search results page for book data.

    Args:
        html: Raw HTML of search results page.
        top_n: Maximum number of organic results to return.

    Returns:
        List of result dicts with: asin, title, author, bsr, price,
        review_count, avg_rating, categories.
    """
    soup = BeautifulSoup(html, 'html.parser')
    result_divs = soup.find_all('div', attrs={'data-asin': True})

    results = []
    for div in result_divs:
        asin = div.get('data-asin', '').strip()
        if not asin:
            continue

        # Skip sponsored results
        if _is_sponsored(div):
            continue

        result = {
            'asin': asin,
            'title': _parse_result_title(div),
            'author': _parse_result_author(div),
            'price': _parse_result_price(div),
            'review_count': _parse_result_review_count(div),
            'avg_rating': _parse_result_rating(div),
            'bsr': None,  # BSR requires product page scrape
            'categories': [],
        }

        results.append(result)
        if len(results) >= top_n:
            break

    # For BSR, we'd need to scrape individual product pages.
    # Instead, use a heuristic: estimate BSR from search position
    # if we can't get actual BSR data.
    for i, result in enumerate(results):
        if result['bsr'] is None and result['review_count'] is not None:
            # Rough heuristic: search position correlates with BSR
            # Position 1 ~ BSR 1,000-5,000, Position 10 ~ BSR 50,000-200,000
            # This is a very rough estimate used when we can't scrape product pages
            position = i + 1
            result['estimated_bsr'] = True
            if result['review_count'] > 1000:
                result['bsr'] = position * 2000
            elif result['review_count'] > 100:
                result['bsr'] = position * 5000
            elif result['review_count'] > 10:
                result['bsr'] = position * 15000
            else:
                result['bsr'] = position * 50000

    return results


def _is_captcha(html):
    """Check if the page is a CAPTCHA."""
    markers = [
        'Enter the characters you see below',
        'Sorry, we just need to make sure you\'re not a robot',
        '/errors/validateCaptcha',
    ]
    html_lower = html.lower()
    return any(m.lower() in html_lower for m in markers)


def _is_sponsored(div):
    """Check if a result div is sponsored."""
    div_str = str(div)
    for marker in SPONSORED_MARKERS:
        if marker in div_str:
            return True
    if div.find(string=re.compile(r'\bSponsored\b', re.IGNORECASE)):
        return True
    return False


def _parse_result_title(div):
    """Extract title from a search result div."""
    for selector in ['h2 a span', 'h2 span']:
        el = div.select_one(selector)
        if el:
            return el.get_text(strip=True)
    return None


def _parse_result_author(div):
    """Extract author from a search result div."""
    # Look for "by Author Name" pattern
    author_div = div.select_one('.a-row .a-size-base+ .a-size-base')
    if author_div:
        return author_div.get_text(strip=True)
    # Alternate: look for author link
    author_link = div.select_one('.a-row a.a-size-base')
    if author_link:
        text = author_link.get_text(strip=True)
        if text and text.lower() != 'kindle edition':
            return text
    return None


def _parse_result_price(div):
    """Extract price from a search result div."""
    # Kindle price
    price_el = div.select_one('.a-price .a-offscreen')
    if price_el:
        match = re.search(r'\$([\d,.]+)', price_el.get_text())
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
    return None


def _parse_result_review_count(div):
    """Extract review count from a search result div."""
    # Review count is usually in an aria-label on the ratings link
    ratings_link = div.select_one('a[href*="customerReviews"]')
    if ratings_link:
        label = ratings_link.get('aria-label', '')
        match = re.search(r'([\d,]+)', label)
        if match:
            return int(match.group(1).replace(',', ''))

    # Alternate: span with review count
    review_span = div.select_one('.a-size-base.s-underline-text')
    if review_span:
        text = review_span.get_text(strip=True)
        match = re.search(r'([\d,]+)', text)
        if match:
            return int(match.group(1).replace(',', ''))

    return None


def _parse_result_rating(div):
    """Extract average rating from a search result div."""
    rating_el = div.select_one('.a-icon-star-small .a-icon-alt')
    if rating_el:
        match = re.search(r'([\d.]+)', rating_el.get_text())
        if match:
            return float(match.group(1))

    # Alternate: aria-label on rating element
    star_el = div.select_one('[aria-label*="out of 5 stars"]')
    if star_el:
        match = re.search(r'([\d.]+)', star_el.get('aria-label', ''))
        if match:
            return float(match.group(1))

    return None
