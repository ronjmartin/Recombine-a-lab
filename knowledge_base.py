import sqlite3
import json
import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Database constants
DB_PATH = "pdf_knowledge_base.db"

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

def save_document(doc_data: Dict[str, Any]) -> bool:
    """
    Save document data to the knowledge base
    
    Args:
        doc_data (Dict[str, Any]): Document data to save
        
    Returns:
        bool: True if successful, False otherwise
    """
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
        print(f"Error saving document: {str(e)}")
        return False

def get_all_documents() -> List[Dict[str, Any]]:
    """
    Get all documents from the knowledge base
    
    Returns:
        List[Dict[str, Any]]: List of document data
    """
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
        print(f"Error getting documents: {str(e)}")
        return []

def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific document from the knowledge base
    
    Args:
        doc_id (int): Document ID
        
    Returns:
        Optional[Dict[str, Any]]: Document data or None if not found
    """
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
        print(f"Error getting document {doc_id}: {str(e)}")
        return None

def delete_document(doc_id: int) -> bool:
    """
    Delete a document from the knowledge base
    
    Args:
        doc_id (int): Document ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting document {doc_id}: {str(e)}")
        return False

def search_documents(query: str) -> List[Dict[str, Any]]:
    """
    Search for documents in the knowledge base
    
    Args:
        query (str): Search query
        
    Returns:
        List[Dict[str, Any]]: List of matching documents
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Split query into terms for better matching
        terms = query.split()
        query_pattern = ' OR '.join([f"text_content LIKE '%{term}%'" for term in terms])
        
        cursor.execute(f'''
        SELECT id, title, description, text_content, original_filename, upload_time, tags
        FROM documents
        WHERE {query_pattern}
        ORDER BY upload_time DESC
        ''')
        
        rows = cursor.fetchall()
        documents = []
        
        for row in rows:
            doc = dict(row)
            
            # Calculate relevance score (simple term frequency)
            relevance = 0
            text_lower = doc['text_content'].lower()
            
            for term in terms:
                term_lower = term.lower()
                term_count = text_lower.count(term_lower)
                relevance += term_count
            
            # Normalize score by document length
            text_length = max(1, len(text_lower) / 1000)  # Per 1000 chars
            doc['relevance'] = relevance / text_length
            
            # Truncate text_content to reduce response size
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
        print(f"Error searching documents: {str(e)}")
        return []
