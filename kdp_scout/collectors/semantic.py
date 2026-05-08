"""Semantic analysis collector for KDP Scout.

Uses Claude API to cluster keywords semantically and generate
natural language search phrases optimized for Amazon's A10 algorithm.
Results are cached in the semantic_clusters database table.
"""

import json
import logging

from kdp_scout.collectors.base import BaseCollector
from kdp_scout.config import Config
from kdp_scout.db import SemanticClusterRepository, init_db

logger = logging.getLogger(__name__)


class SemanticCollector(BaseCollector):
    """Collects semantic keyword clusters using Claude API.

    Takes a list of keywords, groups them into semantic clusters,
    and generates natural search phrases for each cluster.
    Results are cached in the database to avoid redundant API calls.
    """

    name = 'semantic'

    def __init__(self):
        """Initialize with database access."""
        init_db()
        self._repo = SemanticClusterRepository()

    def close(self):
        """Close database connection."""
        self._repo.close()

    def is_available(self):
        """Check if Claude API is configured.

        Returns:
            True if ANTHROPIC_API_KEY is set.
        """
        return bool(Config.ANTHROPIC_API_KEY)

    def collect(self, query, **kwargs):
        """Cluster keywords semantically and generate phrases.

        Args:
            query: Not used directly. Pass keywords via kwargs.
            **kwargs:
                keywords: List of keyword strings to cluster.
                book_title: Optional book title for context.
                book_genre: Optional book genre for context.
                use_cache: Whether to use cached results (default True).

        Returns:
            List of cluster dicts: [{'label': str, 'keywords': list,
                'relevance_score': float, 'phrases': list}]
        """
        keywords = kwargs.get('keywords', [])
        book_title = kwargs.get('book_title')
        book_genre = kwargs.get('book_genre')
        use_cache = kwargs.get('use_cache', True)

        if not keywords:
            return []

        book_context = _build_context_key(book_title, book_genre)

        # Check cache
        if use_cache and book_context:
            cached = self._repo.get_latest_clusters(
                book_context, max_age_hours=24,
            )
            if cached:
                logger.info(
                    f'Using {len(cached)} cached semantic clusters '
                    f'for "{book_context}"'
                )
                return [
                    {
                        'label': row['cluster_label'],
                        'keywords': json.loads(row['keywords']),
                        'relevance_score': row['relevance_score'],
                        'phrases': json.loads(row['generated_phrases'])
                        if row['generated_phrases'] else [],
                    }
                    for row in cached
                ]

        # Call Claude API for clustering
        clusters = self._cluster_via_claude(keywords, book_title, book_genre)

        # Cache results
        for cluster in clusters:
            self._repo.add_cluster(
                cluster_label=cluster['label'],
                keywords=json.dumps(cluster['keywords']),
                relevance_score=cluster['relevance_score'],
                generated_phrases=json.dumps(cluster['phrases']),
                book_context=book_context,
            )

        return clusters

    def _cluster_via_claude(self, keywords, book_title=None, book_genre=None):
        """Use Claude API to semantically cluster keywords.

        Args:
            keywords: List of keyword strings.
            book_title: Optional book title for context.
            book_genre: Optional genre for context.

        Returns:
            List of cluster dicts.
        """
        if not self.is_available():
            logger.warning(
                'ANTHROPIC_API_KEY not set. Cannot perform semantic analysis.'
            )
            return []

        try:
            import anthropic
        except ImportError:
            logger.warning('anthropic package not installed.')
            return []

        context_parts = []
        if book_title:
            context_parts.append(f'Book: {book_title}')
        if book_genre:
            context_parts.append(f'Genre: {book_genre}')
        context_str = (' | '.join(context_parts) + '\n\n') if context_parts else ''

        keyword_list = '\n'.join(f'- {kw}' for kw in keywords[:60])

        prompt = f"""{context_str}Analyze these Amazon book keywords and group them into semantic clusters:

{keyword_list}

For each cluster:
1. Give it a descriptive label
2. List which keywords belong to it
3. Rate its relevance to book discovery (0.0 to 1.0)
4. Generate 2-3 natural search phrases a reader would type (each under 50 characters)
5. NEVER include book titles or author names in the phrases

Return ONLY valid JSON:
{{
  "clusters": [
    {{
      "label": "cluster name",
      "keywords": ["kw1", "kw2"],
      "relevance": 0.85,
      "phrases": ["natural search phrase one", "natural search phrase two"]
    }}
  ]
}}"""

        try:
            client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=2000,
                messages=[{'role': 'user', 'content': prompt}],
            )

            content = response.content[0].text.strip()
            # Handle markdown code blocks
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

            data = json.loads(content)
            clusters = []

            for item in data.get('clusters', []):
                clusters.append({
                    'label': item.get('label', 'Unknown'),
                    'keywords': item.get('keywords', []),
                    'relevance_score': float(item.get('relevance', 0.5)),
                    'phrases': item.get('phrases', []),
                })

            logger.info(f'Semantic analysis: {len(clusters)} clusters found')
            return clusters

        except anthropic.AuthenticationError:
            logger.error('Invalid ANTHROPIC_API_KEY. Check your .env file.')
            return []
        except anthropic.RateLimitError:
            logger.error('Anthropic API rate limit exceeded. Try again later.')
            return []
        except anthropic.APIConnectionError:
            logger.error('Could not connect to Anthropic API.')
            return []
        except anthropic.APIError as e:
            logger.error(f'Anthropic API error: {e}')
            return []
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f'Failed to parse semantic analysis response: {e}')
            return []


def _build_context_key(book_title, book_genre):
    """Build a cache key from book context.

    Args:
        book_title: Book title string or None.
        book_genre: Book genre string or None.

    Returns:
        Context key string, or None if no context provided.
    """
    parts = []
    if book_title:
        parts.append(book_title.strip())
    if book_genre:
        parts.append(book_genre.strip())
    return ' | '.join(parts) if parts else None
