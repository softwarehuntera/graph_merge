"""Tools for calculating string similarity."""

from typing import Any

from Levenshtein import jaro_winkler, ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compare_multiple(str1: str, str2: str, match_function_args: dict[str, dict[str, Any]]) -> bool:
    """Apply multiple similarity functions and return True if any conditions are met.

    Args:
        str1 (str): First string.
        str2 (str): Second string.
        match_function_args (dict): Dictionary of similarity function(s) with respective arguments.

    Example Use:
        compare_multiple(
            str1="foo",
            str2="bar",
            match_function_args={
                "ratio_match": {"threshold": 0.7},
                "string_similarity_cosine": {"threshold": 0.7, "ngram_size": 2}
            }
        )
        >>> False
    Returns:
        bool: True if any provided simility functions are satisfied.
    """
    is_match = False

    for function, args in match_function_args.items():
        if SIMILARITY_FUNCTION_KEY[function](str1=str1, str2=str2, **args):
            is_match = True
            break

    return is_match


def exact_match(str1: str, str2: str) -> bool:
    """Check if str1 and str2 are exact matches.

    Args:
        str1 (str): First string.
        str2 (str): Second string.

    Returns:
        bool: True if strings are exact match.
    """
    return str1 == str2


def contains(str1: str, str2: str) -> bool:
    """Check if str1 and str2 are exact matches.

    Args:
        str1 (str): First string.
        str2 (str): Second string.

    Returns:
        bool: True if one string is contained in the other.
    """
    return (str1 in str2) or (str2 in str1)


def jaro_match(str1: str, str2: str, threshold: float = 0.6) -> bool:
    """Check if str1 and str2 jaro_winkler similarity exceed threshold.

    Args:
        str1 (str): First string.
        str2 (str): Second string.
        threshold (float, optional): Similarity threshold. Defaults to 0.6.

    Returns:
        bool: True if similarity exceeds threshold.
    """
    jaro_score = jaro_winkler(str1, str2)
    return jaro_score >= threshold


def ratio_match(str1: str, str2: str, threshold: float = 0.6) -> bool:
    """Check if str1 and str2 ratio similarity exceeds threshold.

    Args:
        str1 (str): First string.
        str2 (str): Second string.
        threshold (float, optional): Similarity threshold. Defaults to 0.6.

    Returns:
        bool: True if similarity exceeds threshold.
    """
    ratio_score = ratio(str1, str2)
    return ratio_score >= threshold


def string_similarity_cosine(str1: str, str2: str, ngram_size: int = 3, threshold: float = 0.6) -> bool:
    """Check if str1 and str2 ngram cosine similarity exceeds threshold.

    Args:
        str1 (str): First string.
        str2 (str): Second string.
        ngram_size (int, optional): Number of characters per element in vector. Defaults to 3.
        threshold (float, optional): Similarity threshold. Defaults to 0.6.

    Returns:
        bool: True if similarity exceeds threshold.
    """
    vectorizer = TfidfVectorizer(
        min_df=1,
        analyzer="char",
        ngram_range=(1, ngram_size),
    )
    right_matrix = vectorizer.fit_transform([str1])
    left_matrix = vectorizer.transform([str2])
    cos_sim = cosine_similarity(left_matrix, right_matrix)[0][0]
    return cos_sim >= threshold


SIMILARITY_FUNCTION_KEY = {
    "exact_match": exact_match,
    "contains": contains,
    "jaro_match": jaro_match,
    "ratio_match": ratio_match,
    "string_similarity_cosine": string_similarity_cosine,
}
