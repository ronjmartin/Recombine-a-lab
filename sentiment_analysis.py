import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from typing import Dict, List, Any, Tuple
import math
import string
from collections import Counter

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

# Initialize NLTK components
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for analysis: tokenize, remove stopwords, and lemmatize
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: Preprocessed tokens
    """
    # Tokenize into words
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Perform sentiment analysis on text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, Any]: Sentiment analysis results
    """
    # Initialize result dictionary
    result = {
        'sentiment_score': 0,
        'overall_sentiment': 'Neutral',
        'emotions': {},
        'section_sentiments': []
    }
    
    if not text or len(text.strip()) == 0:
        return result
    
    # Overall sentiment analysis
    sentiment_scores = sia.polarity_scores(text)
    result['sentiment_score'] = sentiment_scores['compound']
    
    # Determine overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        result['overall_sentiment'] = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        result['overall_sentiment'] = 'Negative'
    else:
        result['overall_sentiment'] = 'Neutral'
    
    # Analyze emotions with more granularity
    result['emotions'] = {
        'Positive': sentiment_scores['pos'],
        'Negative': sentiment_scores['neg'],
        'Neutral': sentiment_scores['neu']
    }
    
    # Analyze sentiment by sections (paragraphs or chunks)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # If text is long, use paragraphs; otherwise, use sentences
    if len(paragraphs) <= 3:
        sections = sent_tokenize(text)
    else:
        sections = paragraphs
    
    # Process manageable number of sections
    max_sections = 10
    sections_to_process = sections[:max_sections] if len(sections) > max_sections else sections
    
    for section in sections_to_process:
        if len(section.strip()) > 0:
            section_score = sia.polarity_scores(section)['compound']
            result['section_sentiments'].append(section_score)
    
    return result

def extract_key_findings(text: str) -> Dict[str, Any]:
    """
    Extract key findings from text: topics, entities, and summary
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, Any]: Key findings including topics, entities, and summary
    """
    result = {
        'topics': [],
        'entities': [],
        'summary': ""
    }
    
    if not text or len(text.strip()) == 0:
        return result
    
    # Preprocess text
    tokens = preprocess_text(text)
    
    # Extract key topics/phrases (using frequency analysis)
    freq_dist = FreqDist(tokens)
    common_words = [word for word, freq in freq_dist.most_common(20)]
    
    # Extract bigrams for more context
    bigrams = list(nltk.bigrams(tokens))
    bigram_freq = FreqDist(bigrams)
    common_bigrams = [f"{b[0]} {b[1]}" for b, freq in bigram_freq.most_common(10)]
    
    # Combine unigrams and bigrams for topics
    result['topics'] = common_bigrams + [word for word in common_words if word not in ' '.join(common_bigrams)]
    result['topics'] = result['topics'][:15]  # Limit to top 15 topics
    
    # Extract named entities
    try:
        sentences = sent_tokenize(text)
        entities = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            named_entities = ne_chunk(tagged)
            
            current_entity = []
            current_type = None
            
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_type = chunk.label()
                    entity_text = ' '.join([word for word, tag in chunk.leaves()])
                    entities.append({
                        'text': entity_text,
                        'type': entity_type
                    })
        
        # Remove duplicates and sort by occurrence
        unique_entities = {}
        for entity in entities:
            key = f"{entity['text']}|{entity['type']}"
            if key not in unique_entities:
                unique_entities[key] = entity
        
        # Convert to list and limit
        result['entities'] = list(unique_entities.values())[:20]
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
    
    # Generate simple extractive summary
    try:
        sentences = sent_tokenize(text)
        
        # Score sentences based on term frequency
        word_frequencies = Counter(tokens)
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        
        # Normalize frequencies
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_tokens = word_tokenize(sentence.lower())
            sentence_tokens = [token for token in sentence_tokens if token.isalpha() and token not in stop_words]
            
            # Avoid very short sentences
            if len(sentence_tokens) < 3:
                continue
                
            score = sum(word_frequencies.get(word, 0) for word in sentence_tokens)
            # Normalize by sentence length
            score = score / (len(sentence_tokens) + 1)
            
            # Boost score for sentences at the beginning and end
            if i < len(sentences) * 0.2:  # First 20% of sentences
                score *= 1.2
            elif i > len(sentences) * 0.8:  # Last 20% of sentences
                score *= 1.1
                
            sentence_scores[sentence] = score
        
        # Select top sentences for summary
        summary_sentences_count = min(7, max(3, len(sentences) // 10))
        summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:summary_sentences_count]
        
        # Re-order summary sentences based on original order
        ordered_summary = [s[0] for s in summary_sentences]
        ordered_summary.sort(key=lambda s: sentences.index(s))
        
        result['summary'] = ' '.join(ordered_summary)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
    
    return result
