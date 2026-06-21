"""RSS-specific type definitions: feed providers, content selectors, and feed config.

Cross-stage types (``Entry``, ``Summary``, ``ContentType``, ``Label``,
``TopicType``, ``Filter``) live in ``digest_generator.core.types``. Genuine
infrastructure types (``DeviceType``, ``ModelConfig``) live in
``digest_generator.shared.transformers.types``.
"""

from dataclasses import dataclass
from enum import StrEnum

from digest_generator.core.types import ContentType

__all__ = [
    "BoilerplateMarker",
    "Feed",
    "SelectorType",
]


class BoilerplateMarker(StrEnum):
    """Common boilerplate phrases used to detect low-quality extracted content."""

    SUBSCRIBE = "subscribe"  # Newsletter signup form
    NEWSLETTER = "newsletter"  # Newsletter signup form
    COOKIE = "cookie"  # Cookie consent form
    COOKIES = "cookies"  # Cookie consent form
    SIGN_UP = "sign up"  # Signup form
    SIGN_IN = "sign in"  # Login form
    LOG_IN = "log in"  # Login form
    REGISTER = "register"  # Signup form
    PRIVACY_POLICY = "privacy policy"  # Privacy policy
    TERMS_OF_SERVICE = "terms of service"  # Terms of service
    ACCEPT_ALL = "accept all"  # Accept all cookies
    MANAGE_PREFERENCES = "manage preferences"  # Manage preferences
    UNSUBSCRIBE = "unsubscribe"  # Unsubscribe form


class SelectorType(StrEnum):
    """CSS selectors for article content extraction, tried in enum order."""

    ARTICLE = "article"  # HTML5 semantic
    ROLE_ARTICLE = '[role="article"]'  # ARIA role
    ITEMPROP_ARTICLE_BODY = '[itemprop="articleBody"]'  # Schema.org markup
    HTML_CONTENT = ".html-content"  # Spotify Engineering
    SINGLE_POST_CONTENT = ".single-post__content"  # Spotify Engineering (BEM)
    ENTRY_WORDPRESS_CONTENT = ".entry-content"  # WordPress default
    POST_CONTENT = ".post-content"  # Common blog pattern
    ARTICLE_CONTENT = ".article-content"  # Generic article
    POST_BODY = ".post-body"  # WordPress variant
    ARTICLE_BODY = ".article-body"  # Article variant
    BLOG_CONTENT = ".blog-content"  # AWS blogs
    BLOG_POST_CONTENT = ".blog-post-content"  # Tech blogs
    POST_BEM_CONTENT = ".post__content"  # BEM pattern
    ARTICLE_BEM_CONTENT = ".article__body"  # BEM pattern
    MAIN = "main"  # HTML5 semantic fallback


@dataclass
class Feed:
    """An RSS feed source: a stable name, a URL, and a content category.

    ``name`` is the stable identifier (used as the fetch cache filename and
    as each entry's ``origin``). ``content_type`` places the feed's articles
    in a digest section. Feeds are defined by the user in ``feeds.yaml`` and
    materialized by ``digest_generator.sources.rss.config``.
    """

    name: str
    url: str
    content_type: ContentType
