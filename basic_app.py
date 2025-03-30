import streamlit as st
import os
import time
import pandas as pd
import re
import io
import base64
import json
import sqlite3
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PDF Analysis Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database constants
DB_PATH = "pdf_knowledge_base.db"

# Create database tables if they don't exist
def initialize_db():
    """Initialize the knowledge base database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        text_content TEXT NOT NULL,
        tables TEXT,
        sentiment TEXT,
        key_findings TEXT,
        original_filename TEXT,
        upload_time TEXT,
        tags TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
initialize_db()

# PDF processing functions
def extract_text_basic(uploaded_file):
    """
    Basic text extraction function that returns file content as text
    
    Since we can't use the PDF extraction libraries, we'll return a message
    explaining the limitation but allow the user to continue with the workflow
    """
    return f"""
    This is a basic version of the PDF Analysis Dashboard.
    
    In this version, full PDF text extraction is unavailable because the required libraries 
    (PyPDF2, pdfplumber, camelot) couldn't be loaded.
    
    The filename you uploaded is: {uploaded_file.name}
    Size: {uploaded_file.size / 1024:.2f} KB
    
    To use the full functionality with proper PDF extraction,
    please check that all dependencies are correctly installed.
    """

def extract_sample_tables():
    """Return sample tables for demonstration purposes"""
    tables = []
    
    # Create a sample table
    table1 = pd.DataFrame({
        'Category': ['Category A', 'Category B', 'Category C', 'Category D'],
        'Value': [42, 58, 73, 91],
        'Percentage': ['25%', '35%', '20%', '20%']
    })
    
    # Create another sample table
    table2 = pd.DataFrame({
        'Date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
        'Revenue': [12500, 13750, 15000, 16250],
        'Expenses': [10000, 11000, 12000, 13000],
        'Profit': [2500, 2750, 3000, 3250]
    })
    
    tables.append(table1)
    tables.append(table2)
    
    return tables

# Sentiment analysis functions
def basic_sentiment_analysis(text):
    """
    Perform basic sentiment analysis on text using a rule-based approach
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, Any]: Basic sentiment analysis results
    """
    # Define positive and negative word lists
    positive_words = [
        'good', 'great', 'excellent', 'positive', 'nice', 'wonderful', 'amazing',
        'fantastic', 'happy', 'pleased', 'satisfied', 'enjoy', 'beneficial',
        'success', 'successful', 'recommend', 'improvement', 'improved', 'better',
        'best', 'quality', 'superior', 'outstanding', 'exceptional'
    ]
    
    negative_words = [
        'bad', 'terrible', 'poor', 'negative', 'awful', 'horrible', 'unpleasant',
        'disappointed', 'disappointing', 'fail', 'failure', 'worst', 'worse',
        'problem', 'issue', 'complaint', 'difficult', 'trouble', 'unfortunate',
        'unsatisfied', 'dissatisfied', 'inferior', 'frustrating', 'inadequate'
    ]
    
    # Convert text to lowercase for comparison
    text_lower = text.lower()
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', text_lower))
    negative_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', text_lower))
    
    # Calculate sentiment score (-1 to 1)
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
    else:
        sentiment_score = 0
    
    # Determine overall sentiment
    if sentiment_score > 0.05:
        overall_sentiment = 'Positive'
    elif sentiment_score < -0.05:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Neutral'
    
    # Split text into sections (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    section_sentiments = []
    
    # Calculate sentiment for each paragraph
    for paragraph in paragraphs[:5]:  # Limit to first 5 paragraphs
        if len(paragraph.strip()) > 0:
            # Count positive and negative words in paragraph
            para_lower = paragraph.lower()
            para_positive = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', para_lower))
            para_negative = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', para_lower))
            
            para_total = para_positive + para_negative
            if para_total > 0:
                para_score = (para_positive - para_negative) / para_total
            else:
                para_score = 0
                
            section_sentiments.append(para_score)
    
    # Create emotion dictionary
    emotions = {
        'Positive': positive_count / max(1, total_sentiment_words),
        'Negative': negative_count / max(1, total_sentiment_words),
        'Neutral': 1 - ((positive_count + negative_count) / max(1, len(text.split())))
    }
    
    return {
        'overall_sentiment': overall_sentiment,
        'sentiment_score': sentiment_score,
        'emotions': emotions,
        'section_sentiments': section_sentiments
    }

def extract_basic_key_findings(text):
    """
    Extract basic key findings without requiring NLTK
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, Any]: Basic findings
    """
    # Split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common English stop words
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                 "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                 "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
                 "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
                 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top words as topics
    topics = [word for word, count in sorted_words[:10]]
    
    # Extract potential entities (capitalized words)
    words_with_case = re.findall(r'\b[A-Z][a-z]+\b', text)
    entity_counts = {}
    
    for word in words_with_case:
        if word in entity_counts:
            entity_counts[word] += 1
        else:
            entity_counts[word] = 1
    
    # Convert to list of entities
    entities = [
        {'text': word, 'label': 'ENTITY', 'count': count}
        for word, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    ]
    
    # Create a simple summary
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Take first sentence and a couple more based on keyword density
    summary = sentences[0] if sentences else ""
    
    # Add 1-2 more sentences with the highest keyword density
    sentence_scores = []
    for i, sentence in enumerate(sentences[1:6]):  # Consider next 5 sentences
        score = sum(1 for word in topics[:5] if word.lower() in sentence.lower())
        sentence_scores.append((i+1, score))
    
    # Sort by score
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Add top 2 sentences to summary
    for idx, _ in sentence_scores[:2]:
        if idx < len(sentences):
            summary += " " + sentences[idx]
    
    return {
        'topics': topics,
        'entities': entities,
        'summary': summary
    }

# Visualization functions
def create_visualization(df, x_col, y_col, viz_type):
    """Create a simple visualization for tables"""
    # Since we're not using plotly, return a styled dataframe
    if viz_type == "Bar Chart":
        return df.style.bar(subset=[y_col], color='#5694f1')
    else:
        return df

def create_sentiment_gauge(sentiment_score):
    """Create a simple gauge display for sentiment"""
    # Convert from [-1, 1] to [0, 1]
    score_normalized = (sentiment_score + 1) / 2  
    
    st.write("Sentiment Gauge:")
    st.progress(score_normalized)
    
    # Determine color based on sentiment score
    if sentiment_score > 0.5:
        color = "green"
    elif sentiment_score > 0:
        color = "lightgreen"
    elif sentiment_score > -0.5:
        color = "orange"
    else:
        color = "red"
    
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{sentiment_score:.2f}</h1>", unsafe_allow_html=True)

def create_emotion_chart(emotion_df):
    """Create a simple chart for emotions"""
    st.bar_chart(emotion_df.set_index('Emotion')['Score'])

def create_section_sentiment_chart(section_df):
    """Create a chart for section sentiments"""
    st.line_chart(section_df.set_index('Section')['Score'])

def create_entity_chart(entities):
    """Create a simple entity visualization"""
    entity_df = pd.DataFrame(entities)
    st.bar_chart(entity_df.set_index('text')['count'])

# Knowledge base functions
def save_document(doc_data):
    """Save document data to the knowledge base"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Process table data for storage
        if 'tables' in doc_data and doc_data['tables']:
            # Convert DataFrames to serializable format
            serialized_tables = []
            for table in doc_data['tables']:
                serialized_tables.append(table.to_dict('records'))
            doc_data['tables'] = json.dumps(serialized_tables)
        else:
            doc_data['tables'] = '[]'
        
        # Process sentiment data
        if 'sentiment' in doc_data and doc_data['sentiment']:
            doc_data['sentiment'] = json.dumps(doc_data['sentiment'])
        else:
            doc_data['sentiment'] = '{}'
        
        # Process key findings
        if 'key_findings' in doc_data and doc_data['key_findings']:
            doc_data['key_findings'] = json.dumps(doc_data['key_findings'])
        else:
            doc_data['key_findings'] = '{}'
        
        # Process tags
        if 'tags' in doc_data and doc_data['tags']:
            doc_data['tags'] = json.dumps(doc_data['tags'])
        else:
            doc_data['tags'] = '[]'
        
        # Insert document into database
        cursor.execute('''
        INSERT INTO documents (
            title, description, text_content, tables, sentiment, 
            key_findings, original_filename, upload_time, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_data['title'], 
            doc_data.get('description', ''),
            doc_data['text_content'],
            doc_data['tables'],
            doc_data['sentiment'],
            doc_data['key_findings'],
            doc_data.get('original_filename', ''),
            doc_data.get('upload_time', time.strftime('%Y-%m-%d %H:%M:%S')),
            doc_data['tags']
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving document: {str(e)}")
        return False

def get_all_documents():
    """Get all documents from the knowledge base"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, title, description, original_filename, upload_time, tags
        FROM documents
        ORDER BY upload_time DESC
        ''')
        
        rows = cursor.fetchall()
        documents = []
        
        for row in rows:
            doc = dict(row)
            # Parse JSON stored tags
            try:
                doc['tags'] = json.loads(doc['tags'])
            except:
                doc['tags'] = []
            documents.append(doc)
        
        conn.close()
        return documents
    except Exception as e:
        st.error(f"Error getting documents: {str(e)}")
        return []

def get_document(doc_id):
    """Get a specific document from the knowledge base"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, title, description, text_content, tables, sentiment, 
               key_findings, original_filename, upload_time, tags
        FROM documents
        WHERE id = ?
        ''', (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            doc = dict(row)
            
            # Parse JSON fields
            try:
                doc['tables'] = json.loads(doc['tables'])
                # Convert tables back to DataFrames
                if isinstance(doc['tables'], list):
                    doc_tables = []
                    for table_data in doc['tables']:
                        doc_tables.append(pd.DataFrame(table_data))
                    doc['tables'] = doc_tables
            except:
                doc['tables'] = []
            
            try:
                doc['sentiment'] = json.loads(doc['sentiment'])
            except:
                doc['sentiment'] = {}
            
            try:
                doc['key_findings'] = json.loads(doc['key_findings'])
            except:
                doc['key_findings'] = {}
            
            try:
                doc['tags'] = json.loads(doc['tags'])
            except:
                doc['tags'] = []
            
            return doc
        else:
            return None
    except Exception as e:
        st.error(f"Error getting document {doc_id}: {str(e)}")
        return None

def search_documents(query):
    """Search for documents in the knowledge base"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use LIKE query for basic search
        cursor.execute('''
        SELECT id, title, description, text_content, original_filename, upload_time, tags
        FROM documents
        WHERE text_content LIKE ? OR title LIKE ? OR description LIKE ?
        ORDER BY upload_time DESC
        ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
        
        rows = cursor.fetchall()
        documents = []
        
        for row in rows:
            doc = dict(row)
            
            # Calculate simple relevance score (term frequency)
            query_lower = query.lower()
            text_lower = doc['text_content'].lower()
            title_lower = doc['title'].lower()
            
            title_matches = title_lower.count(query_lower) * 3  # Weight title matches more
            content_matches = text_lower.count(query_lower)
            
            # Normalize score by document length
            text_length = max(1, len(text_lower) / 1000)  # Per 1000 chars
            doc['relevance'] = (title_matches + content_matches) / text_length
            
            # Truncate text_content for display
            doc['text_content'] = doc['text_content'][:300] + "..."
            
            # Parse JSON stored tags
            try:
                doc['tags'] = json.loads(doc['tags'])
            except:
                doc['tags'] = []
            
            documents.append(doc)
        
        # Sort by relevance score
        documents.sort(key=lambda x: x['relevance'], reverse=True)
        
        conn.close()
        return documents
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def main():
    # Sidebar
    st.sidebar.title("PDF Analysis Dashboard")
    st.sidebar.markdown("---")
    
    # App mode selection
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Upload & Analyze PDF", "Knowledge Base", "About"]
    )
    
    if app_mode == "Upload & Analyze PDF":
        upload_and_analyze()
    elif app_mode == "Knowledge Base":
        knowledge_base_page()
    else:
        about_page()

def upload_and_analyze():
    st.title("Upload & Analyze PDF")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "FileName": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.write("### File Details")
        st.json(file_details)
        
        # Add a progress bar for analysis
        with st.spinner("Analyzing PDF..."):
            # Extract text from PDF (basic version)
            text_content = extract_text_basic(uploaded_file)
            
            # Extract sample tables
            tables = extract_sample_tables()
            
            # Analyze sentiment (basic version)
            sentiment = basic_sentiment_analysis(text_content)
            
            # Extract key findings (basic version)
            key_findings = extract_basic_key_findings(text_content)
        
        # Create tabs for different analysis results
        tabs = st.tabs(["Text Content", "Tables", "Sentiment Analysis", "Key Findings", "Save to Knowledge Base"])
        
        # Tab 1: Text Content
        with tabs[0]:
            st.markdown("### Extracted Text")
            st.text_area("", text_content, height=400)
            
            # Download button for text
            st.download_button(
                label="Download Text",
                data=text_content,
                file_name=f"{uploaded_file.name.split('.')[0]}_text.txt",
                mime="text/plain"
            )
        
        # Tab 2: Tables
        with tabs[1]:
            st.markdown("### Sample Tables")
            st.info("Note: These are sample tables for demonstration. In the full version, tables would be extracted from your PDF.")
            
            if tables:
                for i, table in enumerate(tables):
                    st.markdown(f"#### Table {i+1}")
                    st.dataframe(table)
                    
                    # Download button for each table
                    csv = table.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Table {i+1} as CSV",
                        data=csv,
                        file_name=f"{uploaded_file.name.split('.')[0]}_table_{i+1}.csv",
                        mime="text/csv"
                    )
                    
                    # Simple visualization
                    st.markdown("#### Visualization")
                    viz_type = st.selectbox(
                        f"Choose visualization for Table {i+1}",
                        ["Bar Chart", "Table View"],
                        key=f"viz_type_{i}"
                    )
                    
                    if len(table.columns) > 1:
                        x_col = st.selectbox(f"Select X-axis column for Table {i+1}", table.columns, key=f"x_col_{i}")
                        y_col = st.selectbox(f"Select Y-axis column for Table {i+1}", table.columns, key=f"y_col_{i}")
                        
                        try:
                            st.bar_chart(table.set_index(x_col)[y_col])
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                    else:
                        st.info("Table needs at least two columns for visualization")
            else:
                st.info("No tables found in the PDF")
        
        # Tab 3: Sentiment Analysis
        with tabs[2]:
            st.markdown("### Sentiment Analysis")
            
            # Display sentiment scores
            st.write(f"**Overall Sentiment:** {sentiment['overall_sentiment']}")
            st.write(f"**Sentiment Score:** {sentiment['sentiment_score']:.2f}")
            
            # Sentiment gauge
            create_sentiment_gauge(sentiment['sentiment_score'])
            
            # Display emotion breakdown
            st.markdown("#### Emotion Breakdown")
            emotion_df = pd.DataFrame({
                'Emotion': list(sentiment['emotions'].keys()),
                'Score': list(sentiment['emotions'].values())
            })
            emotion_df = emotion_df.sort_values('Score', ascending=False)
            
            # Bar chart for emotions
            create_emotion_chart(emotion_df)
            
            # Display detailed sentiment by section
            if 'section_sentiments' in sentiment and sentiment['section_sentiments']:
                st.markdown("#### Sentiment by Document Section")
                
                section_df = pd.DataFrame({
                    'Section': [f"Section {i+1}" for i in range(len(sentiment['section_sentiments']))],
                    'Score': sentiment['section_sentiments']
                })
                
                create_section_sentiment_chart(section_df)
        
        # Tab 4: Key Findings
        with tabs[3]:
            st.markdown("### Key Findings & Insights")
            
            # Display key phrases/topics
            st.markdown("#### Key Topics")
            if key_findings['topics']:
                for topic in key_findings['topics']:
                    st.markdown(f"- {topic}")
            else:
                st.info("No significant topics detected")
            
            # Display key entities
            st.markdown("#### Named Entities")
            if key_findings['entities']:
                entity_df = pd.DataFrame(key_findings['entities'])
                st.dataframe(entity_df)
                
                # Create entity visualization
                create_entity_chart(key_findings['entities'])
            else:
                st.info("No significant entities detected")
            
            # Display summary
            st.markdown("#### Summary")
            if key_findings['summary']:
                st.write(key_findings['summary'])
            else:
                st.info("Could not generate summary")
        
        # Tab 5: Save to Knowledge Base
        with tabs[4]:
            st.markdown("### Save to Knowledge Base")
            
            # Form for saving to knowledge base
            with st.form(key="save_to_kb_form"):
                title = st.text_input("Document Title", value=uploaded_file.name.split('.')[0])
                description = st.text_area("Description (optional)")
                tags = st.text_input("Tags (comma-separated)")
                
                # Submit button
                submit_button = st.form_submit_button(label="Save to Knowledge Base")
                
                if submit_button:
                    # Prepare document data
                    doc_data = {
                        'title': title,
                        'description': description,
                        'tags': [tag.strip() for tag in tags.split(',')] if tags else [],
                        'text_content': text_content,
                        'tables': tables,
                        'sentiment': sentiment,
                        'key_findings': key_findings,
                        'original_filename': uploaded_file.name,
                        'upload_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save to knowledge base
                    success = save_document(doc_data)
                    
                    if success:
                        st.success("Document successfully saved to Knowledge Base!")
                    else:
                        st.error("Failed to save document to Knowledge Base")

def knowledge_base_page():
    st.title("Knowledge Base")
    
    # Tabs for Browse and Search
    kb_tabs = st.tabs(["Browse Documents", "Search"])
    
    # Tab 1: Browse Documents
    with kb_tabs[0]:
        st.markdown("### Browse Documents")
        
        # Get all documents
        documents = get_all_documents()
        
        if documents:
            # Display documents in a dataframe
            doc_df = pd.DataFrame([
                {'id': doc['id'], 'title': doc['title'], 
                 'description': doc['description'], 'upload_time': doc['upload_time']} 
                for doc in documents
            ])
            st.dataframe(doc_df)
            
            # Select document to view
            selected_doc_id = st.selectbox(
                "Select document to view details",
                options=[doc['id'] for doc in documents],
                format_func=lambda x: next((doc['title'] for doc in documents if doc['id'] == x), '')
            )
            
            if selected_doc_id:
                display_document_details(selected_doc_id)
        else:
            st.info("No documents found in the Knowledge Base")
    
    # Tab 2: Search
    with kb_tabs[1]:
        st.markdown("### Search Documents")
        
        search_query = st.text_input("Enter search query")
        search_button = st.button("Search")
        
        if search_button and search_query:
            # Search documents
            search_results = search_documents(search_query)
            
            if search_results:
                st.success(f"Found {len(search_results)} matching documents")
                
                # Display search results
                search_df = pd.DataFrame([
                    {'id': doc['id'], 'title': doc['title'], 
                     'description': doc['description'], 'relevance': doc['relevance']} 
                    for doc in search_results
                ])
                st.dataframe(search_df)
                
                # Select document to view
                selected_doc_id = st.selectbox(
                    "Select document to view details",
                    options=[doc['id'] for doc in search_results],
                    format_func=lambda x: next((doc['title'] for doc in search_results if doc['id'] == x), '')
                )
                
                if selected_doc_id:
                    display_document_details(selected_doc_id)
            else:
                st.info("No matching documents found")

def display_document_details(doc_id):
    """Display details of a selected document"""
    document = get_document(doc_id)
    
    if document:
        st.markdown(f"## {document['title']}")
        st.markdown(f"**Upload Date:** {document['upload_time']}")
        
        if document['description']:
            st.markdown(f"**Description:** {document['description']}")
        
        if document['tags']:
            st.markdown(f"**Tags:** {', '.join(document['tags'])}")
        
        # Create tabs for document content
        doc_tabs = st.tabs(["Text Content", "Tables", "Sentiment", "Key Findings"])
        
        # Tab 1: Text Content
        with doc_tabs[0]:
            st.markdown("### Text Content")
            st.text_area("", document['text_content'], height=400)
            
            # Download button for text
            st.download_button(
                label="Download Text",
                data=document['text_content'],
                file_name=f"{document['title']}_text.txt",
                mime="text/plain"
            )
        
        # Tab 2: Tables
        with doc_tabs[1]:
            st.markdown("### Tables")
            
            if document['tables']:
                for i, table in enumerate(document['tables']):
                    try:
                        st.markdown(f"#### Table {i+1}")
                        st.dataframe(table)
                        
                        # Download button for each table
                        csv = table.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Table {i+1} as CSV",
                            data=csv,
                            file_name=f"{document['title']}_table_{i+1}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error displaying table {i+1}: {str(e)}")
            else:
                st.info("No tables found in this document")
        
        # Tab 3: Sentiment
        with doc_tabs[2]:
            st.markdown("### Sentiment Analysis")
            
            sentiment = document['sentiment']
            
            if sentiment:
                # Display sentiment scores
                st.write(f"**Overall Sentiment:** {sentiment['overall_sentiment']}")
                st.write(f"**Sentiment Score:** {sentiment['sentiment_score']:.2f}")
                
                # Sentiment gauge chart
                create_sentiment_gauge(sentiment['sentiment_score'])
                
                # Display emotion breakdown
                if 'emotions' in sentiment:
                    st.markdown("#### Emotion Breakdown")
                    emotion_df = pd.DataFrame({
                        'Emotion': list(sentiment['emotions'].keys()),
                        'Score': list(sentiment['emotions'].values())
                    })
                    emotion_df = emotion_df.sort_values('Score', ascending=False)
                    
                    # Bar chart for emotions
                    create_emotion_chart(emotion_df)
            else:
                st.info("No sentiment analysis available for this document")
        
        # Tab 4: Key Findings
        with doc_tabs[3]:
            st.markdown("### Key Findings & Insights")
            
            key_findings = document['key_findings']
            
            if key_findings:
                # Display key phrases/topics
                st.markdown("#### Key Topics")
                if key_findings['topics']:
                    for topic in key_findings['topics']:
                        st.markdown(f"- {topic}")
                else:
                    st.info("No significant topics detected")
                
                # Display key entities
                st.markdown("#### Named Entities")
                if key_findings['entities']:
                    entity_df = pd.DataFrame(key_findings['entities'])
                    st.dataframe(entity_df)
                    
                    # Create entity visualization
                    create_entity_chart(key_findings['entities'])
                else:
                    st.info("No significant entities detected")
                
                # Display summary
                st.markdown("#### Summary")
                if key_findings['summary']:
                    st.write(key_findings['summary'])
                else:
                    st.info("Could not generate summary")
            else:
                st.info("No key findings available for this document")

def about_page():
    st.title("About PDF Analysis Dashboard")
    
    st.markdown("""
    ## Overview
    
    This application provides tools for analyzing PDF documents. 
    You can upload PDFs, extract text, perform sentiment analysis, 
    identify key insights, and save everything to a knowledge base.
    
    ## Features
    
    - **PDF Upload**: Upload PDF files for analysis
    - **Text Extraction**: Basic text representation for PDFs
    - **Sentiment Analysis**: Analyze the emotional tone using a rule-based approach
    - **Key Findings**: Extract important topics, entities, and generate a summary
    - **Knowledge Base**: Save analyzed documents for future reference and searching
    - **Visualization**: Simple data visualization capabilities
    
    ## How to Use
    
    1. Navigate to the "Upload & Analyze PDF" section
    2. Upload a PDF document using the file uploader
    3. View the analysis results in the different tabs
    4. Save important documents to the knowledge base
    5. Use the knowledge base to browse or search for previously analyzed documents
    
    ## Note About This Version
    
    This is a basic version with limited PDF extraction capabilities. The full version 
    requires additional libraries (PyPDF2, pdfplumber, camelot) for complete PDF processing.
    """)
    
    # Display system information
    st.markdown("## System Information")
    st.write(f"Application Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Database Path: {DB_PATH}")

if __name__ == "__main__":
    main()