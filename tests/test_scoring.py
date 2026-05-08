"""Tests for keyword scoring algorithm."""

import math
import pytest
from unittest.mock import MagicMock, patch
from kdp_scout.keyword_engine import (
    KeywordScorer,
    DEFAULT_WEIGHTS,
    normalize_autocomplete,
    normalize_competition,
    normalize_bsr,
    normalize_impressions,
    normalize_orders,
    normalize_ctr,
    normalize_acos,
    normalize_search_volume,
    normalize_suggested_bid,
    normalize_own_ranking,
)


def make_keyword_row(**overrides):
    """Create a mock keyword row with metric fields."""
    defaults = {
        'id': 1,
        'keyword': 'test keyword',
        'autocomplete_position': None,
        'competition_count': None,
        'avg_bsr_top_results': None,
        'impressions': None,
        'clicks': None,
        'orders': None,
        'estimated_volume': None,
        'suggested_bid': None,
        'source': 'autocomplete',
        'first_seen': '2025-01-01',
        'category': None,
        'score': None,
        'snapshot_date': '2025-01-01',
    }
    defaults.update(overrides)
    return defaults


# ── Normalizer unit tests ────────────────────────────────────────────


class TestNormalizeAutocomplete:
    def test_position_1(self):
        assert normalize_autocomplete(1) == 1.0

    def test_position_5(self):
        assert normalize_autocomplete(5) == pytest.approx(0.6)

    def test_position_10(self):
        assert normalize_autocomplete(10) == pytest.approx(0.1)

    def test_position_11_is_zero(self):
        assert normalize_autocomplete(11) == 0.0

    def test_position_20_is_zero(self):
        assert normalize_autocomplete(20) == 0.0

    def test_none(self):
        assert normalize_autocomplete(None) == 0.0

    def test_zero(self):
        assert normalize_autocomplete(0) == 0.0

    def test_negative(self):
        assert normalize_autocomplete(-1) == 0.0


class TestNormalizeCompetition:
    def test_zero_competition(self):
        assert normalize_competition(0) == 1.0

    def test_50k_competition(self):
        assert normalize_competition(50000) == pytest.approx(0.5)

    def test_100k_competition(self):
        assert normalize_competition(100000) == pytest.approx(1.0 / 3.0)

    def test_very_high_competition(self):
        result = normalize_competition(10000000)
        assert 0.0 < result < 0.01

    def test_none(self):
        assert normalize_competition(None) == 0.0

    def test_negative(self):
        assert normalize_competition(-100) == 0.0


class TestNormalizeBsr:
    def test_bsr_1(self):
        assert normalize_bsr(1) == pytest.approx(1.0)

    def test_bsr_1000(self):
        # log10(1000) = 3, so 1 - 3/6 = 0.5
        assert normalize_bsr(1000) == pytest.approx(0.5)

    def test_bsr_1m(self):
        # log10(1000000) = 6, so 1 - 6/6 = 0.0
        assert normalize_bsr(1000000) == pytest.approx(0.0)

    def test_bsr_10m_clamps_to_zero(self):
        assert normalize_bsr(10000000) == 0.0

    def test_none(self):
        assert normalize_bsr(None) == 0.0

    def test_zero(self):
        assert normalize_bsr(0) == 0.0

    def test_negative(self):
        assert normalize_bsr(-100) == 0.0


class TestNormalizeImpressions:
    def test_100k_impressions(self):
        # log10(100000) = 5, so 5/5 = 1.0
        assert normalize_impressions(100000) == pytest.approx(1.0)

    def test_1000_impressions(self):
        # log10(1000) = 3, so 3/5 = 0.6
        assert normalize_impressions(1000) == pytest.approx(0.6)

    def test_1_impression(self):
        # log10(1) = 0, so 0/5 = 0.0
        assert normalize_impressions(1) == pytest.approx(0.0)

    def test_million_impressions_clamped(self):
        assert normalize_impressions(1000000) == 1.0

    def test_none(self):
        assert normalize_impressions(None) == 0.0

    def test_zero(self):
        assert normalize_impressions(0) == 0.0

    def test_negative(self):
        assert normalize_impressions(-5) == 0.0


class TestNormalizeOrders:
    def test_1000_orders(self):
        # log10(1000) = 3, so 3/3 = 1.0
        assert normalize_orders(1000) == pytest.approx(1.0)

    def test_10_orders(self):
        # log10(10) = 1, so 1/3 = ~0.333
        assert normalize_orders(10) == pytest.approx(1.0 / 3.0)

    def test_1_order(self):
        # log10(1) = 0, so 0/3 = 0.0
        assert normalize_orders(1) == pytest.approx(0.0)

    def test_5000_orders_clamped(self):
        assert normalize_orders(5000) == 1.0

    def test_none(self):
        assert normalize_orders(None) == 0.0

    def test_zero(self):
        assert normalize_orders(0) == 0.0

    def test_negative(self):
        assert normalize_orders(-1) == 0.0


class TestNormalizeCtr:
    def test_5_percent_ctr(self):
        # 50 clicks / 1000 impressions = 5% = 1.0
        assert normalize_ctr(50, 1000) == pytest.approx(1.0)

    def test_2_5_percent_ctr(self):
        # 25 clicks / 1000 impressions = 2.5% = 0.5
        assert normalize_ctr(25, 1000) == pytest.approx(0.5)

    def test_10_percent_ctr_clamped(self):
        assert normalize_ctr(100, 1000) == 1.0

    def test_zero_clicks(self):
        assert normalize_ctr(0, 1000) == 0.0

    def test_none_clicks(self):
        assert normalize_ctr(None, 1000) == 0.0

    def test_none_impressions(self):
        assert normalize_ctr(50, None) == 0.0

    def test_zero_impressions(self):
        assert normalize_ctr(50, 0) == 0.0

    def test_negative_clicks(self):
        assert normalize_ctr(-5, 1000) == 0.0

    def test_both_none(self):
        assert normalize_ctr(None, None) == 0.0


class TestNormalizeAcos:
    def test_zero_acos(self):
        # 0% ACOS = perfect profitability = 1.0
        assert normalize_acos(0.0) == 1.0

    def test_50_percent_acos(self):
        # 50% ACOS = 0.5
        assert normalize_acos(0.5) == pytest.approx(0.5)

    def test_100_percent_acos(self):
        # 100% ACOS = 0.0
        assert normalize_acos(1.0) == pytest.approx(0.0)

    def test_200_percent_acos_clamped(self):
        assert normalize_acos(2.0) == 0.0

    def test_none(self):
        assert normalize_acos(None) == 0.0

    def test_35_percent_acos(self):
        assert normalize_acos(0.35) == pytest.approx(0.65)


class TestNormalizeSearchVolume:
    def test_100k_volume(self):
        assert normalize_search_volume(100000) == pytest.approx(1.0)

    def test_1000_volume(self):
        assert normalize_search_volume(1000) == pytest.approx(0.6)

    def test_none(self):
        assert normalize_search_volume(None) == 0.0

    def test_zero(self):
        assert normalize_search_volume(0) == 0.0

    def test_negative(self):
        assert normalize_search_volume(-100) == 0.0

    def test_million_clamped(self):
        assert normalize_search_volume(1000000) == 1.0


class TestNormalizeSuggestedBid:
    def test_3_dollar_bid(self):
        assert normalize_suggested_bid(3.0) == pytest.approx(1.0)

    def test_1_50_bid(self):
        assert normalize_suggested_bid(1.5) == pytest.approx(0.5)

    def test_5_dollar_bid_clamped(self):
        assert normalize_suggested_bid(5.0) == 1.0

    def test_none(self):
        assert normalize_suggested_bid(None) == 0.0

    def test_zero(self):
        assert normalize_suggested_bid(0) == 0.0

    def test_negative(self):
        assert normalize_suggested_bid(-1.0) == 0.0


class TestNormalizeOwnRanking:
    def test_rank_1(self):
        assert normalize_own_ranking(1) == pytest.approx(1.0)

    def test_rank_25(self):
        assert normalize_own_ranking(25) == pytest.approx(25.0 / 49.0)

    def test_rank_50(self):
        assert normalize_own_ranking(50) == pytest.approx(0.0, abs=1e-10)

    def test_rank_100_clamped(self):
        assert normalize_own_ranking(100) == 0.0

    def test_none(self):
        assert normalize_own_ranking(None) == 0.0

    def test_zero(self):
        assert normalize_own_ranking(0) == 0.0

    def test_negative(self):
        assert normalize_own_ranking(-5) == 0.0


# ── Weights validation ───────────────────────────────────────────────


class TestWeights:
    def test_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_weights_positive(self):
        for name, weight in DEFAULT_WEIGHTS.items():
            assert weight > 0, f"Weight '{name}' should be positive"

    def test_expected_weight_keys(self):
        expected = {
            'autocomplete', 'competition', 'bsr_demand',
            'ads_impressions', 'ads_orders', 'ads_profitability',
            'search_volume', 'commercial_value', 'click_through_rate',
            'own_ranking', 'semantic_relevance',
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected


# ── KeywordScorer integration tests ──────────────────────────────────


class TestScoreKeyword:
    """Tests for the composite scoring via KeywordScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a KeywordScorer with mocked DB access."""
        with patch('kdp_scout.keyword_engine.init_db'):
            with patch('kdp_scout.keyword_engine.KeywordRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo_cls.return_value = mock_repo
                s = KeywordScorer()
                s._repo = mock_repo
                # Default: no ACOS data, no ranking data, no ads data
                mock_repo.get_ads_acos_for_keyword.return_value = None
                mock_repo.get_own_ranking_for_keyword.return_value = None
                mock_repo.get_ads_data_for_keyword.return_value = None
                return s

    def test_no_signals_returns_zero(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        assert scorer.score_keyword(1) == 0.0

    def test_nonexistent_keyword_returns_zero(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = None
        assert scorer.score_keyword(999) == 0.0

    def test_score_keyword_returns_float(self, scorer):
        """Backward compatibility: score_keyword returns a float."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        result = scorer.score_keyword(1)
        assert isinstance(result, float)

    def test_score_keyword_detailed_returns_dict(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        result = scorer.score_keyword_detailed(1)
        assert isinstance(result, dict)
        assert 'total' in result
        assert 'components' in result

    def test_detailed_has_all_components(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        result = scorer.score_keyword_detailed(1)
        expected_keys = set(DEFAULT_WEIGHTS.keys())
        assert set(result['components'].keys()) == expected_keys

    def test_each_component_has_required_fields(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        result = scorer.score_keyword_detailed(1)
        for name, comp in result['components'].items():
            assert 'score' in comp, f"Component '{name}' missing 'score'"
            assert 'weight' in comp, f"Component '{name}' missing 'weight'"
            assert 'weighted' in comp, f"Component '{name}' missing 'weighted'"
            assert 'raw' in comp, f"Component '{name}' missing 'raw'"
            assert 'description' in comp, f"Component '{name}' missing 'description'"

    # -- Score range tests --

    def test_score_always_in_range_no_data(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        score = scorer.score_keyword(1)
        assert 0.0 <= score <= 100.0

    def test_score_always_in_range_all_data(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
            competition_count=100,
            avg_bsr_top_results=10,
            impressions=100000,
            clicks=5000,
            orders=1000,
            estimated_volume=100000,
            suggested_bid=5.0,
        )
        scorer._repo.get_ads_acos_for_keyword.return_value = 0.0
        scorer._repo.get_own_ranking_for_keyword.return_value = 1
        score = scorer.score_keyword(1)
        assert 0.0 <= score <= 100.0

    def test_perfect_score_is_90(self, scorer):
        """All non-semantic signals at maximum should produce 90.

        The remaining 10% is from semantic_relevance which requires
        separate semantic analysis to populate.
        """
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
            competition_count=0,
            avg_bsr_top_results=1,
            impressions=100000,
            clicks=5000,  # 5% CTR
            orders=1000,
            estimated_volume=100000,
            suggested_bid=3.0,
        )
        scorer._repo.get_ads_acos_for_keyword.return_value = 0.0
        scorer._repo.get_own_ranking_for_keyword.return_value = 1
        score = scorer.score_keyword(1)
        assert score == pytest.approx(90.0, abs=0.1)

    # -- Autocomplete scoring in context --

    def test_autocomplete_only_position_1(self, scorer):
        """Position 1 autocomplete with default weight (0.15) = 15 points."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
        )
        score = scorer.score_keyword(1)
        assert score == pytest.approx(15.0, abs=0.1)

    def test_autocomplete_only_position_5(self, scorer):
        """Position 5 = 0.6 normalized * 0.15 weight * 100 = 9.0."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=5,
        )
        score = scorer.score_keyword(1)
        assert score == pytest.approx(9.0, abs=0.1)

    # -- Combined scoring tests --

    def test_autocomplete_and_orders(self, scorer):
        """Autocomplete pos 1 (15.0) + 1000 orders (15.0) = 30.0."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
            orders=1000,
        )
        score = scorer.score_keyword(1)
        assert score == pytest.approx(30.0, abs=0.1)

    def test_with_acos_data(self, scorer):
        """ACOS cross-reference contributes to score."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
        )
        scorer._repo.get_ads_acos_for_keyword.return_value = 0.35  # 35% ACOS
        score = scorer.score_keyword(1)
        # 15.0 from autocomplete + 6.5 from ACOS (0.65 * 0.10 * 100)
        assert score == pytest.approx(21.5, abs=0.1)

    def test_with_own_ranking(self, scorer):
        """Own ranking cross-reference contributes to score."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
        )
        scorer._repo.get_own_ranking_for_keyword.return_value = 1
        score = scorer.score_keyword(1)
        # 15.0 from autocomplete + 5.0 from ranking (1.0 * 0.05 * 100)
        assert score == pytest.approx(20.0, abs=0.1)

    # -- Edge cases --

    def test_keyword_with_only_ads_data(self, scorer):
        """Keyword with only impressions and orders."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=10000,
            orders=50,
            clicks=200,
        )
        score = scorer.score_keyword(1)
        assert score > 0.0
        assert score <= 100.0

    def test_keyword_with_only_autocomplete(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=3,
        )
        score = scorer.score_keyword(1)
        # 0.8 * 0.15 * 100 = 12.0
        assert score == pytest.approx(12.0, abs=0.1)

    def test_none_values_dont_crash(self, scorer):
        """All None values should produce 0 score without errors."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=None,
            competition_count=None,
            avg_bsr_top_results=None,
            impressions=None,
            clicks=None,
            orders=None,
            estimated_volume=None,
            suggested_bid=None,
        )
        scorer._repo.get_ads_acos_for_keyword.return_value = None
        scorer._repo.get_own_ranking_for_keyword.return_value = None
        score = scorer.score_keyword(1)
        assert score == 0.0

    def test_empty_result_for_none_keyword(self, scorer):
        scorer._repo.get_keyword_with_metrics.return_value = None
        result = scorer.score_keyword_detailed(999)
        assert result['total'] == 0.0
        assert len(result['components']) == len(DEFAULT_WEIGHTS)

    def test_detailed_weighted_sum_equals_total(self, scorer):
        """The sum of all weighted components should equal total."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=3,
            competition_count=25000,
            avg_bsr_top_results=50000,
            impressions=5000,
            clicks=100,
            orders=20,
            estimated_volume=10000,
            suggested_bid=1.50,
        )
        scorer._repo.get_ads_acos_for_keyword.return_value = 0.40
        scorer._repo.get_own_ranking_for_keyword.return_value = 5

        result = scorer.score_keyword_detailed(1)
        computed_sum = sum(c['weighted'] for c in result['components'].values())
        assert result['total'] == pytest.approx(round(computed_sum, 1), abs=0.1)


class TestScoreAllKeywords:
    """Tests for batch scoring."""

    @pytest.fixture
    def scorer(self):
        with patch('kdp_scout.keyword_engine.init_db'):
            with patch('kdp_scout.keyword_engine.KeywordRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo_cls.return_value = mock_repo
                s = KeywordScorer()
                s._repo = mock_repo
                mock_repo.get_ads_acos_for_keyword.return_value = None
                mock_repo.get_own_ranking_for_keyword.return_value = None
                mock_repo.get_ads_data_for_keyword.return_value = None
                return s

    def test_score_all_recalculate(self, scorer):
        scorer._repo.get_all_keyword_ids.return_value = [1, 2, 3]
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        count = scorer.score_all_keywords(recalculate=True)
        assert count == 3
        assert scorer._repo.update_score.call_count == 3

    def test_score_all_new_only(self, scorer):
        scorer._repo.get_unscored_keyword_ids.return_value = [4, 5]
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row()
        count = scorer.score_all_keywords(recalculate=False)
        assert count == 2


class TestCustomWeights:
    """Test that custom weights override defaults."""

    @pytest.fixture
    def scorer(self):
        custom_weights = {
            'autocomplete': 1.0,
            'competition': 0.0,
            'bsr_demand': 0.0,
            'ads_impressions': 0.0,
            'ads_orders': 0.0,
            'ads_profitability': 0.0,
            'search_volume': 0.0,
            'commercial_value': 0.0,
            'click_through_rate': 0.0,
            'own_ranking': 0.0,
        }
        with patch('kdp_scout.keyword_engine.init_db'):
            with patch('kdp_scout.keyword_engine.KeywordRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo_cls.return_value = mock_repo
                s = KeywordScorer(weights=custom_weights)
                s._repo = mock_repo
                mock_repo.get_ads_acos_for_keyword.return_value = None
                mock_repo.get_own_ranking_for_keyword.return_value = None
                mock_repo.get_ads_data_for_keyword.return_value = None
                return s

    def test_autocomplete_only_weight(self, scorer):
        """With all weight on autocomplete, pos 1 = 100."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=1,
            competition_count=100,
        )
        score = scorer.score_keyword(1)
        assert score == pytest.approx(100.0, abs=0.1)


# ── Ads data cross-reference fallback tests ──────────────────────────


class TestAdsDataFallback:
    """Tests for falling back to ads_search_terms when keyword_metrics
    lacks impressions/clicks/orders."""

    @pytest.fixture
    def scorer(self):
        """Create a KeywordScorer with mocked DB access."""
        with patch('kdp_scout.keyword_engine.init_db'):
            with patch('kdp_scout.keyword_engine.KeywordRepository') as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo_cls.return_value = mock_repo
                s = KeywordScorer()
                s._repo = mock_repo
                mock_repo.get_ads_acos_for_keyword.return_value = None
                mock_repo.get_own_ranking_for_keyword.return_value = None
                mock_repo.get_ads_data_for_keyword.return_value = None
                return s

    def test_fallback_to_ads_search_terms(self, scorer):
        """Ads data from ads_search_terms is used when keyword_metrics
        lacks impressions/clicks/orders."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=3,
            impressions=None,
            clicks=None,
            orders=None,
        )
        scorer._repo.get_ads_data_for_keyword.return_value = {
            'impressions': 2341,
            'clicks': 22,
            'orders': 5,
            'spend': 10.50,
            'sales': 25.00,
        }
        result = scorer.score_keyword_detailed(1)
        # ads_impressions should reflect 2341 impressions, not "No data"
        assert result['components']['ads_impressions']['raw'] == 2341
        assert result['components']['ads_impressions']['score'] > 0.0
        # ads_orders should reflect 5 orders
        assert result['components']['ads_orders']['raw'] == 5
        assert result['components']['ads_orders']['score'] > 0.0
        # CTR should be computed from fallback clicks/impressions
        assert result['components']['click_through_rate']['raw'] is not None
        assert result['components']['click_through_rate']['score'] > 0.0

    def test_keyword_metrics_takes_precedence(self, scorer):
        """keyword_metrics data takes precedence if both exist."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=500,
            clicks=10,
            orders=2,
        )
        scorer._repo.get_ads_data_for_keyword.return_value = {
            'impressions': 9999,
            'clicks': 999,
            'orders': 99,
            'spend': 50.0,
            'sales': 100.0,
        }
        result = scorer.score_keyword_detailed(1)
        # Should use keyword_metrics values, not ads_search_terms
        assert result['components']['ads_impressions']['raw'] == 500
        assert result['components']['ads_orders']['raw'] == 2

    def test_ctr_from_ads_fallback(self, scorer):
        """CTR is correctly computed from ads clicks/impressions."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=None,
            clicks=None,
            orders=None,
        )
        scorer._repo.get_ads_data_for_keyword.return_value = {
            'impressions': 1000,
            'clicks': 25,
            'orders': 3,
            'spend': 5.0,
            'sales': 15.0,
        }
        result = scorer.score_keyword_detailed(1)
        # CTR = 25/1000 = 2.5%, normalized = 2.5/5.0 = 0.5
        assert result['components']['click_through_rate']['raw'] == pytest.approx(0.025)
        assert result['components']['click_through_rate']['score'] == pytest.approx(0.5)

    def test_ads_only_keyword_scores_correctly(self, scorer):
        """Keywords with ads data but no autocomplete data score correctly."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            autocomplete_position=None,
            impressions=None,
            clicks=None,
            orders=None,
        )
        scorer._repo.get_ads_data_for_keyword.return_value = {
            'impressions': 10000,
            'clicks': 200,
            'orders': 50,
            'spend': 30.0,
            'sales': 100.0,
        }
        scorer._repo.get_ads_acos_for_keyword.return_value = 0.30  # 30%
        result = scorer.score_keyword_detailed(1)
        # Should score > 0 from ads signals alone
        assert result['total'] > 0.0
        # Autocomplete should be 0
        assert result['components']['autocomplete']['score'] == 0.0
        # Ads impressions should contribute
        assert result['components']['ads_impressions']['score'] > 0.0
        # Ads orders should contribute
        assert result['components']['ads_orders']['score'] > 0.0
        # ACOS should contribute
        assert result['components']['ads_profitability']['score'] > 0.0

    def test_aggregation_multiple_rows(self, scorer):
        """The aggregation handles multiple matching rows in ads_search_terms
        (sum impressions, sum clicks, etc.) -- verified via mock return value
        that represents pre-aggregated sums."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=None,
            clicks=None,
            orders=None,
        )
        # This simulates the DB returning summed values from multiple rows
        scorer._repo.get_ads_data_for_keyword.return_value = {
            'impressions': 5000,   # e.g., 2000 + 3000 from two report dates
            'clicks': 100,         # e.g., 40 + 60
            'orders': 10,          # e.g., 4 + 6
            'spend': 20.0,
            'sales': 50.0,
        }
        result = scorer.score_keyword_detailed(1)
        assert result['components']['ads_impressions']['raw'] == 5000
        assert result['components']['ads_orders']['raw'] == 10
        assert result['components']['click_through_rate']['raw'] == pytest.approx(
            100 / 5000
        )

    def test_no_fallback_when_metrics_have_data(self, scorer):
        """get_ads_data_for_keyword should NOT be called when keyword_metrics
        already has impressions/clicks/orders."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=100,
            clicks=5,
            orders=1,
        )
        scorer.score_keyword_detailed(1)
        scorer._repo.get_ads_data_for_keyword.assert_not_called()

    def test_fallback_called_when_all_none(self, scorer):
        """get_ads_data_for_keyword IS called when all ads fields are None."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=None,
            clicks=None,
            orders=None,
        )
        scorer.score_keyword_detailed(1)
        scorer._repo.get_ads_data_for_keyword.assert_called_once_with('test keyword')

    def test_fallback_called_when_all_zero(self, scorer):
        """get_ads_data_for_keyword IS called when all ads fields are 0."""
        scorer._repo.get_keyword_with_metrics.return_value = make_keyword_row(
            impressions=0,
            clicks=0,
            orders=0,
        )
        scorer.score_keyword_detailed(1)
        scorer._repo.get_ads_data_for_keyword.assert_called_once_with('test keyword')
