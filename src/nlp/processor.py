"""
Natural Language Processing for Real Estate Text
Provides: amenity extraction, sentiment, keywords, similarity
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class NLPProcessor:
    """Lightweight NLP for property descriptions"""
    
    def __init__(self):
        # Pre-compile patterns for speed
        self.amenity_patterns = {
            'Pool': r'\b(pool|swimming|piscina|سباحة)\b',
            'Gym': r'\b(gym|fitness|workout|نادي)\b',
            'Parking': r'\b(parking|garage|carport|موقف)\b',
            'Security': r'\b(security|cctv|guarded|حراسة)\b',
            'Garden': r'\b(garden|yard|landscape|حديقة)\b',
            'Balcony': r'\b(balcony|terrace|veranda|شرفة)\b',
            'Elevator': r'\b(elevator|lift|ascensor|مصعد)\b',
            'Concierge': r'\b(concierge|doorman|porter|بواب)\b',
            'Sea View': r'\b(sea view|ocean view|waterfront|مطل على البحر)\b',
            'City View': r'\b(city view|skyline|panoramic|بانوراما)\b',
            'Central A/C': r'\b(central ac|air conditioning|split ac|تكييف|مكيف)\b',
        }
        
        # Sentiment lexicon
        self.positive_words = [
            'beautiful', 'luxury', 'luxurious', 'spacious', 'modern', 'great', 'nice', 
            'perfect', 'amazing', 'excellent', 'stunning', 'gorgeous', 'premium',
            'high-end', 'elegant', 'bright', 'quiet', 'peaceful', 'prime',
            'مميز', 'فاخر', 'رائع', 'جميل', 'هادئ', 'مميز'
        ]
        self.negative_words = [
            'old', 'small', 'poor', 'need', 'needs', 'renovation', 'issue', 'problem',
            'damage', 'noisy', 'dark', 'dull', 'outdated', 'wear', 'tear',
            'تحتاج', 'قديم', 'متهالك', 'سيء', 'مشكلة', 'ضوضاء'
        ]
    
    def extract_amenities(self, text):
        """
        Extract mentioned amenities from text.
        
        Args:
            text: Property description string
            
        Returns:
            List of amenity names (capitalized)
        """
        if not text or not isinstance(text, str):
            return []
        
        found = []
        text_lower = text.lower()
        for amenity, pattern in self.amenity_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                found.append(amenity)
        return found
    
    def analyze_sentiment(self, text):
        """
        Simple sentiment analysis using lexicon.
        Returns score between -1 (negative) and +1 (positive).
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        pos_count = sum(1 for w in self.positive_words if w in text_lower)
        neg_count = sum(1 for w in self.negative_words if w in text_lower)
        total = pos_count + neg_count
        
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
    
    def extract_keywords(self, text, top_n=5):
        """
        Extract top keywords using simple frequency (stopwords removed).
        
        Args:
            text: Input string
            top_n: Number of keywords to return
            
        Returns:
            List of keyword strings
        """
        if not text or not isinstance(text, str):
            return []
        
        # Tokenize: keep words with 3+ letters
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Simple stopwords (extend as needed)
        stopwords = {
            'the', 'and', 'for', 'with', 'has', 'this', 'that', 'are', 'from', 'has',
            'have', 'but', 'not', 'you', 'your', 'all', 'can', 'will', 'our', 'and',
            'was', 'were', 'been', 'have', 'has', 'had', 'having', 'any', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'than', 'then', 'there',
            'these', 'they', 'this', 'through', 'to', 'too', 'under', 'up', 'very',
            'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
            'with', 'you', 'your', 'apartment', 'villa', 'house', 'property', 'home',
            'bedroom', 'bathroom', 'sqm', 'square', 'meter', 'floor', 'room',
        }
        
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        if not words:
            return []
        
        from collections import Counter
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]
    
    def compute_similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts using TF-IDF.
        
        Returns:
            Float between 0 (completely different) and 1 (identical)
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            vectors = vectorizer.fit_transform([text1, text2])
            sim = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(sim)
        except Exception:
            return 0.0
    
    def extract_features(self, text):
        """
        Extract all NLP features from text in one call.
        
        Returns:
            Dict with keys: amenities, sentiment, keywords
        """
        return {
            'amenities': self.extract_amenities(text),
            'sentiment': self.analyze_sentiment(text),
            'keywords': self.extract_keywords(text),
        }


# Convenience functions (module-level)
_processor = None

def get_processor():
    global _processor
    if _processor is None:
        _processor = NLPProcessor()
    return _processor

def extract_amenities(text):
    return get_processor().extract_amenities(text)

def analyze_sentiment(text):
    return get_processor().analyze_sentiment(text)

def extract_keywords(text, top_n=5):
    return get_processor().extract_keywords(text, top_n)

def compute_similarity(text1, text2):
    return get_processor().compute_similarity(text1, text2)
