"""Rufus-aware keyword byte validator for KDP backend keywords.

Amazon's Rufus AI uses semantic matching, not keyword matching.
This validator checks:
  - Byte count per keyword box (500-byte limit, not character limit)
  - Redundancy with title/subtitle words
  - Suggests semantic synonyms and trope keywords
"""

import logging
import re

logger = logging.getLogger(__name__)

KDP_BACKEND_BYTE_LIMIT = 500
KDP_SLOT_COUNT = 7


# Common trope keywords by genre for suggestion
TROPE_KEYWORDS = {
    'romance': [
        'enemies to lovers', 'slow burn', 'second chance romance',
        'grumpy sunshine', 'forbidden love', 'fake dating',
        'friends to lovers', 'forced proximity', 'one bed',
        'marriage of convenience', 'boss employee romance',
    ],
    'thriller': [
        'unreliable narrator', 'locked room mystery', 'police procedural',
        'psychological suspense', 'cat and mouse', 'race against time',
        'cold case', 'missing person', 'serial killer', 'conspiracy',
        'espionage', 'domestic thriller',
    ],
    'fantasy': [
        'chosen one', 'portal fantasy', 'dark academia',
        'found family', 'magic system', 'quest narrative',
        'court intrigue', 'dragon rider', 'sword and sorcery',
        'urban fantasy', 'fae romance',
    ],
    'sci-fi': [
        'first contact', 'space opera', 'dystopian', 'post apocalyptic',
        'time travel', 'artificial intelligence', 'cyberpunk',
        'colony ship', 'alien invasion', 'hard science fiction',
    ],
    'historical fiction': [
        'medieval', 'world war', 'ancient civilizations',
        'plague', 'empire', 'dynasty', 'historical mystery',
        'alternate history', 'biographical fiction',
        'ancient rome', 'ancient greece', 'victorian era',
    ],
    'mystery': [
        'cozy mystery', 'whodunit', 'amateur sleuth', 'cold case',
        'forensic mystery', 'small town mystery', 'locked room',
        'detective series', 'noir', 'private investigator',
    ],
}


def validate_backend_keywords(keyword_slots, title=None, subtitle=None):
    """Validate KDP backend keyword slots against Amazon's rules.

    Args:
        keyword_slots: List of up to 7 keyword strings (one per box).
        title: Optional book title to check for redundancy.
        subtitle: Optional subtitle to check for redundancy.

    Returns:
        Dict with:
            - valid: bool (True if all slots pass)
            - slots: list of per-slot analysis dicts
            - warnings: list of warning strings
            - suggestions: list of improvement suggestions
    """
    warnings = []
    suggestions = []
    slot_results = []

    # Build set of title/subtitle words for redundancy check
    title_words = set()
    if title:
        title_words.update(_extract_words(title))
    if subtitle:
        title_words.update(_extract_words(subtitle))

    all_words_used = set()
    all_valid = True

    for i, slot in enumerate(keyword_slots[:KDP_SLOT_COUNT]):
        slot = slot.strip() if slot else ''
        byte_count = len(slot.encode('utf-8'))
        is_valid = byte_count <= KDP_BACKEND_BYTE_LIMIT

        if not is_valid:
            all_valid = False
            warnings.append(
                f'Slot {i + 1}: {byte_count} bytes exceeds '
                f'{KDP_BACKEND_BYTE_LIMIT}-byte limit. '
                f'Amazon will IGNORE the entire slot.'
            )

        # Check for multi-byte characters that eat into the limit
        char_count = len(slot)
        multi_byte_chars = []
        for ch in slot:
            ch_bytes = len(ch.encode('utf-8'))
            if ch_bytes > 1:
                multi_byte_chars.append((ch, ch_bytes))

        if multi_byte_chars:
            warnings.append(
                f'Slot {i + 1}: Contains {len(multi_byte_chars)} multi-byte '
                f'character(s) that consume extra space: '
                f'{", ".join(f"{c} ({b} bytes)" for c, b in multi_byte_chars[:5])}'
            )

        # Check for redundancy with title
        slot_words = _extract_words(slot)
        redundant_words = slot_words & title_words
        if redundant_words:
            warnings.append(
                f'Slot {i + 1}: Words already in title/subtitle '
                f'(wasted space): {", ".join(sorted(redundant_words))}'
            )

        # Check for cross-slot word duplication
        duplicate_words = slot_words & all_words_used
        if duplicate_words:
            warnings.append(
                f'Slot {i + 1}: Duplicate words from other slots '
                f'(wasted space): {", ".join(sorted(duplicate_words))}'
            )

        all_words_used.update(slot_words)

        slot_results.append({
            'slot': i + 1,
            'content': slot,
            'byte_count': byte_count,
            'byte_limit': KDP_BACKEND_BYTE_LIMIT,
            'byte_pct': round(byte_count / KDP_BACKEND_BYTE_LIMIT * 100, 1),
            'char_count': char_count,
            'word_count': len(slot_words),
            'is_valid': is_valid,
            'redundant_with_title': sorted(redundant_words) if redundant_words else [],
            'multi_byte_chars': multi_byte_chars,
        })

    # Overall suggestions
    total_bytes_used = sum(s['byte_count'] for s in slot_results)
    total_capacity = KDP_BACKEND_BYTE_LIMIT * KDP_SLOT_COUNT
    utilization = total_bytes_used / total_capacity * 100 if total_capacity else 0

    if utilization < 50:
        suggestions.append(
            f'Only using {utilization:.0f}% of available keyword space. '
            f'Add more keywords to improve discoverability.'
        )

    # Check for common mistakes
    for slot in keyword_slots[:KDP_SLOT_COUNT]:
        if not slot:
            continue
        if ',' in slot:
            suggestions.append(
                'Avoid commas in backend keywords. '
                'Amazon treats spaces as separators — commas waste bytes.'
            )
            break

    for slot in keyword_slots[:KDP_SLOT_COUNT]:
        if not slot:
            continue
        if re.search(r'["\']', slot):
            suggestions.append(
                'Avoid quotes in backend keywords. '
                'They consume bytes and Amazon ignores them.'
            )
            break

    return {
        'valid': all_valid,
        'slots': slot_results,
        'warnings': warnings,
        'suggestions': suggestions,
        'total_bytes_used': total_bytes_used,
        'total_capacity': total_capacity,
        'utilization_pct': round(utilization, 1),
        'unique_words': len(all_words_used),
    }


def suggest_trope_keywords(genre, existing_keywords=None):
    """Suggest trope and semantic keywords for a genre.

    Amazon's Rufus AI matches queries like "enemies to lovers" even
    if those exact words aren't in the title. Backend keywords should
    include trope names and semantic synonyms.

    Args:
        genre: Genre string (e.g., 'romance', 'thriller').
        existing_keywords: Optional list of existing keyword strings
            to avoid suggesting duplicates.

    Returns:
        List of suggested keyword strings not already in use.
    """
    genre_lower = genre.lower().strip()
    existing_words = set()
    if existing_keywords:
        for kw in existing_keywords:
            existing_words.update(_extract_words(kw))

    suggestions = []

    # Find matching genre tropes
    for genre_key, tropes in TROPE_KEYWORDS.items():
        if genre_key in genre_lower or genre_lower in genre_key:
            for trope in tropes:
                trope_words = _extract_words(trope)
                # Only suggest if not all words are already covered
                if not trope_words.issubset(existing_words):
                    suggestions.append(trope)

    return suggestions


def optimize_slot_content(slot_content, title=None):
    """Optimize a single keyword slot by removing waste.

    Removes:
    - Words already in the title
    - Commas and quotes (waste bytes)
    - Duplicate words within the slot
    - Leading/trailing whitespace

    Args:
        slot_content: The keyword slot string.
        title: Optional book title for redundancy removal.

    Returns:
        Optimized slot string.
    """
    if not slot_content:
        return ''

    # Remove commas, quotes, and extra punctuation
    cleaned = re.sub(r'[,"\';:!?()]', ' ', slot_content)
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())

    # Remove title words
    if title:
        title_words = _extract_words(title)
        words = cleaned.split()
        words = [w for w in words if w.lower() not in title_words]
        cleaned = ' '.join(words)

    # Remove duplicate words (keep first occurrence)
    seen = set()
    unique_words = []
    for word in cleaned.split():
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    cleaned = ' '.join(unique_words)

    return cleaned.strip()


def _extract_words(text):
    """Extract lowercase words from text, excluding common stop words."""
    if not text:
        return set()
    # Split on non-alphanumeric characters
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter out very short stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'is', 'it', 'as', 'be',
        'am', 'are', 'was', 'were', 'do', 'does', 'did', 'has',
        'have', 'had', 'not', 'no', 'so', 'if', 'up', 'my', 'me',
    }
    return {w for w in words if w not in stop_words and len(w) > 1}
