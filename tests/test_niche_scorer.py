"""Tests for niche_scorer module."""

import pytest
from unittest.mock import patch, MagicMock
from kdp_scout.niche_scorer import (
    _compute_opportunity_score,
    _generate_recommendation,
    _parse_result_title,
    _parse_result_price,
    _parse_result_review_count,
    _is_captcha,
    _is_sponsored,
    find_beatable_categories,
)


class TestComputeOpportunityScore:
    """Tests for _compute_opportunity_score."""

    def test_high_opportunity(self):
        """High BSR + low reviews + good revenue = high score."""
        metrics = {
            'avg_bsr': 250_000,
            'avg_reviews': 15,
            'avg_monthly_revenue': 600,
            'high_bsr_count': 8,
        }
        score = _compute_opportunity_score(metrics, result_count=10)
        assert score >= 70

    def test_low_opportunity(self):
        """Low BSR + high reviews = low score."""
        metrics = {
            'avg_bsr': 5_000,
            'avg_reviews': 800,
            'avg_monthly_revenue': 2000,
            'high_bsr_count': 0,
        }
        score = _compute_opportunity_score(metrics, result_count=10)
        assert score < 30

    def test_moderate_opportunity(self):
        """Mid-range metrics."""
        metrics = {
            'avg_bsr': 60_000,
            'avg_reviews': 80,
            'avg_monthly_revenue': 250,
            'high_bsr_count': 3,
        }
        score = _compute_opportunity_score(metrics, result_count=10)
        assert 30 <= score <= 70

    def test_no_data(self):
        """Missing metrics should still return a score."""
        metrics = {
            'avg_bsr': None,
            'avg_reviews': None,
            'avg_monthly_revenue': None,
            'high_bsr_count': 0,
        }
        score = _compute_opportunity_score(metrics, result_count=0)
        assert score == 0.0

    def test_score_capped_at_100(self):
        """Score should never exceed 100."""
        metrics = {
            'avg_bsr': 1_000_000,
            'avg_reviews': 1,
            'avg_monthly_revenue': 10_000,
            'high_bsr_count': 10,
        }
        score = _compute_opportunity_score(metrics, result_count=10)
        assert score <= 100.0

    def test_score_minimum_zero(self):
        """Score should never go below 0."""
        metrics = {
            'avg_bsr': 1,
            'avg_reviews': 10_000,
            'avg_monthly_revenue': 0,
            'high_bsr_count': 0,
        }
        score = _compute_opportunity_score(metrics, result_count=10)
        assert score >= 0.0

    def test_underserved_ratio_contribution(self):
        """High-BSR count should boost score."""
        base_metrics = {
            'avg_bsr': 100_000,
            'avg_reviews': 40,
            'avg_monthly_revenue': 200,
            'high_bsr_count': 0,
        }
        no_underserved = _compute_opportunity_score(base_metrics, result_count=10)

        base_metrics['high_bsr_count'] = 10
        all_underserved = _compute_opportunity_score(base_metrics, result_count=10)

        assert all_underserved > no_underserved


class TestGenerateRecommendation:
    """Tests for _generate_recommendation."""

    def test_strong_opportunity(self):
        rec = _generate_recommendation(75, {
            'avg_bsr': 200_000,
            'avg_reviews': 20,
            'avg_monthly_revenue': 500,
        })
        assert 'STRONG OPPORTUNITY' in rec

    def test_avoid(self):
        rec = _generate_recommendation(15, {
            'avg_bsr': 2_000,
            'avg_reviews': 1000,
            'avg_monthly_revenue': 5000,
        })
        assert 'AVOID' in rec

    def test_challenging(self):
        rec = _generate_recommendation(35, {
            'avg_bsr': 30_000,
            'avg_reviews': 150,
            'avg_monthly_revenue': 300,
        })
        assert 'CHALLENGING' in rec

    def test_low_revenue_warning(self):
        rec = _generate_recommendation(50, {
            'avg_bsr': 100_000,
            'avg_reviews': 30,
            'avg_monthly_revenue': 10,
        })
        assert 'insufficient' in rec.lower()


class TestCaptchaDetection:
    """Tests for CAPTCHA detection."""

    def test_detects_captcha(self):
        html = '<html>Enter the characters you see below</html>'
        assert _is_captcha(html) is True

    def test_normal_page(self):
        html = '<html><body>Normal search results</body></html>'
        assert _is_captcha(html) is False

    def test_robot_check(self):
        html = "Sorry, we just need to make sure you're not a robot"
        assert _is_captcha(html) is True


class TestIsSponsoredResult:
    """Tests for sponsored result detection."""

    def test_detects_sponsored_class(self):
        from bs4 import BeautifulSoup
        html = '<div data-asin="ABC" class="AdHolder">content</div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        assert _is_sponsored(div) is True

    def test_organic_result(self):
        from bs4 import BeautifulSoup
        html = '<div data-asin="ABC" class="s-result-item">content</div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        assert _is_sponsored(div) is False

    def test_detects_sponsored_text(self):
        from bs4 import BeautifulSoup
        html = '<div data-asin="ABC"><span>Sponsored</span></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        assert _is_sponsored(div) is True
