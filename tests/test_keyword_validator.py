"""Tests for keyword_validator module."""

import pytest
from kdp_scout.keyword_validator import (
    validate_backend_keywords,
    suggest_trope_keywords,
    optimize_slot_content,
    _extract_words,
)


class TestValidateBackendKeywords:
    """Tests for validate_backend_keywords."""

    def test_valid_slots(self):
        slots = ['historical fiction thriller', 'ancient civilizations']
        result = validate_backend_keywords(slots)
        assert result['valid'] is True
        assert len(result['slots']) == 2
        assert result['slots'][0]['is_valid'] is True
        assert result['slots'][1]['is_valid'] is True

    def test_empty_slots(self):
        result = validate_backend_keywords([])
        assert result['valid'] is True
        assert result['slots'] == []
        assert result['unique_words'] == 0

    def test_byte_limit_enforcement(self):
        # 500 bytes is a lot — create a string that exceeds it
        long_slot = 'a ' * 260  # 520 bytes
        result = validate_backend_keywords([long_slot])
        assert result['valid'] is False
        assert result['slots'][0]['is_valid'] is False
        assert any('exceeds' in w for w in result['warnings'])

    def test_title_redundancy_detection(self):
        slots = ['ancient plague mystery thriller']
        result = validate_backend_keywords(
            slots, title='The Ancient Plague'
        )
        # 'ancient' and 'plague' should be flagged as redundant
        assert any('title/subtitle' in w for w in result['warnings'])
        redundant = result['slots'][0]['redundant_with_title']
        assert 'ancient' in redundant
        assert 'plague' in redundant

    def test_cross_slot_duplication(self):
        slots = ['mystery thriller suspense', 'dark thriller fiction']
        result = validate_backend_keywords(slots)
        # 'thriller' appears in both slots
        assert any('Duplicate' in w for w in result['warnings'])

    def test_multi_byte_character_warning(self):
        slots = ['café résumé naïve']
        result = validate_backend_keywords(slots)
        assert any('multi-byte' in w for w in result['warnings'])

    def test_utilization_suggestion(self):
        slots = ['short']
        result = validate_backend_keywords(slots)
        assert any('keyword space' in s for s in result['suggestions'])

    def test_comma_suggestion(self):
        slots = ['mystery, thriller, suspense']
        result = validate_backend_keywords(slots)
        assert any('commas' in s.lower() for s in result['suggestions'])

    def test_quote_suggestion(self):
        slots = ['"best thriller" \'mystery\'']
        result = validate_backend_keywords(slots)
        assert any('quotes' in s.lower() for s in result['suggestions'])

    def test_max_seven_slots(self):
        slots = [f'keyword {i}' for i in range(10)]
        result = validate_backend_keywords(slots)
        # Should only process first 7
        assert len(result['slots']) == 7

    def test_byte_count_accuracy(self):
        slot = 'hello world'
        result = validate_backend_keywords([slot])
        assert result['slots'][0]['byte_count'] == len(slot.encode('utf-8'))
        assert result['slots'][0]['byte_count'] == 11


class TestSuggestTropeKeywords:
    """Tests for suggest_trope_keywords."""

    def test_romance_tropes(self):
        tropes = suggest_trope_keywords('romance')
        assert len(tropes) > 0
        assert any('enemies to lovers' in t for t in tropes)

    def test_thriller_tropes(self):
        tropes = suggest_trope_keywords('thriller')
        assert len(tropes) > 0
        assert any('unreliable narrator' in t for t in tropes)

    def test_historical_fiction_tropes(self):
        tropes = suggest_trope_keywords('historical fiction')
        assert len(tropes) > 0

    def test_unknown_genre_returns_empty(self):
        tropes = suggest_trope_keywords('underwater basket weaving')
        assert tropes == []

    def test_excludes_existing_keywords(self):
        existing = ['enemies to lovers slow burn romance']
        tropes = suggest_trope_keywords('romance', existing_keywords=existing)
        # 'enemies to lovers' and 'slow burn' should be excluded
        assert 'enemies to lovers' not in tropes
        assert 'slow burn' not in tropes

    def test_case_insensitive_genre(self):
        tropes_lower = suggest_trope_keywords('romance')
        tropes_upper = suggest_trope_keywords('Romance')
        assert tropes_lower == tropes_upper


class TestOptimizeSlotContent:
    """Tests for optimize_slot_content."""

    def test_removes_commas(self):
        result = optimize_slot_content('mystery, thriller, suspense')
        assert ',' not in result
        assert 'mystery' in result
        assert 'thriller' in result

    def test_removes_quotes(self):
        result = optimize_slot_content('"best" \'thriller\'')
        assert '"' not in result
        assert "'" not in result

    def test_removes_title_words(self):
        result = optimize_slot_content(
            'ancient plague mystery thriller',
            title='The Ancient Plague',
        )
        assert 'ancient' not in result.lower()
        assert 'plague' not in result.lower()
        assert 'mystery' in result.lower()
        assert 'thriller' in result.lower()

    def test_removes_duplicate_words(self):
        result = optimize_slot_content('thriller dark thriller suspense')
        words = result.lower().split()
        assert words.count('thriller') == 1

    def test_empty_input(self):
        assert optimize_slot_content('') == ''
        assert optimize_slot_content(None) == ''

    def test_normalizes_whitespace(self):
        result = optimize_slot_content('  mystery   thriller  ')
        assert result == 'mystery thriller'


class TestExtractWords:
    """Tests for _extract_words helper."""

    def test_basic(self):
        words = _extract_words('The Ancient Plague')
        assert 'ancient' in words
        assert 'plague' in words
        assert 'the' not in words  # stop word

    def test_empty(self):
        assert _extract_words('') == set()
        assert _extract_words(None) == set()

    def test_filters_short_words(self):
        words = _extract_words('I am a big fan')
        assert 'big' in words
        assert 'fan' in words
        assert 'am' not in words  # stop word
