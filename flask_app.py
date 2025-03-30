from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import os
import time
import pandas as pd
from io import BytesIO
import json
import tempfile
import sqlite3

# Import custom modules
import pdf_extraction as pdf_ext
import sentiment_analysis as sa
import knowledge_base as kb

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Initialize knowledge base
kb.initialize_db()

@app.route('/')
def index():
    """Main page with options for different functionalities"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_and_analyze():
    """Handle PDF upload and analysis"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['pdf_file']
        
        # Check if the file is empty
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Check if the file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            flash('File must be a PDF')
            return redirect(request.url)
        
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        file.save(temp_file.name)
        temp_file.close()
        
        try:
            # Extract text from PDF
            text_content = pdf_ext.extract_text(temp_file.name)
            
            # Extract tables from PDF
            tables = pdf_ext.extract_tables(temp_file.name)
            
            # Analyze sentiment
            sentiment = sa.analyze_sentiment(text_content)
            
            # Extract key findings
            key_findings = sa.extract_key_findings(text_content)
            
            # File details
            file_details = {
                "FileName": file.filename,
                "FileType": file.content_type,
                "FileSize": f"{os.path.getsize(temp_file.name) / 1024:.2f} KB"
            }
            
            # Store analysis results in session for retrieval in other routes
            # Since we can't use sessions easily without setup, we'll store in a temporary DB
            analysis_id = int(time.time())
            store_analysis_results(analysis_id, {
                'text_content': text_content,
                'tables': tables,
                'sentiment': sentiment,
                'key_findings': key_findings,
                'file_details': file_details,
                'temp_file': temp_file.name,
                'filename': file.filename
            })
            
            return redirect(url_for('analysis_results', analysis_id=analysis_id))
            
        except Exception as e:
            flash(f'Error analyzing PDF: {str(e)}')
            return redirect(request.url)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    
    return render_template('upload.html')

@app.route('/analysis/<int:analysis_id>')
def analysis_results(analysis_id):
    """Display analysis results"""
    results = get_analysis_results(analysis_id)
    
    if not results:
        flash('Analysis results not found')
        return redirect(url_for('upload_and_analyze'))
    
    return render_template('analysis.html', 
                          analysis_id=analysis_id,
                          text_content=results['text_content'],
                          tables=results['tables'],
                          sentiment=results['sentiment'],
                          key_findings=results['key_findings'],
                          file_details=results['file_details'],
                          filename=results['filename'])

@app.route('/knowledge_base')
def knowledge_base_page():
    """Display the knowledge base page"""
    # Get all documents
    documents = kb.get_all_documents()
    return render_template('knowledge_base.html', documents=documents)

@app.route('/document/<int:doc_id>')
def document_details(doc_id):
    """Display details of a specific document"""
    document = kb.get_document(doc_id)
    
    if not document:
        flash('Document not found')
        return redirect(url_for('knowledge_base_page'))
    
    return render_template('document_details.html', document=document)

@app.route('/search')
def search_documents():
    """Search for documents in the knowledge base"""
    query = request.args.get('query', '')
    
    if not query:
        return redirect(url_for('knowledge_base_page'))
    
    results = kb.search_documents(query)
    return render_template('search_results.html', results=results, query=query)

@app.route('/save_to_kb/<int:analysis_id>', methods=['POST'])
def save_to_kb(analysis_id):
    """Save analysis results to knowledge base"""
    results = get_analysis_results(analysis_id)
    
    if not results:
        flash('Analysis results not found')
        return redirect(url_for('upload_and_analyze'))
    
    title = request.form.get('title', results['filename'].split('.')[0])
    description = request.form.get('description', '')
    tags = request.form.get('tags', '')
    
    # Prepare document data
    doc_data = {
        'title': title,
        'description': description,
        'tags': [tag.strip() for tag in tags.split(',')] if tags else [],
        'text_content': results['text_content'],
        'tables': results['tables'],
        'sentiment': results['sentiment'],
        'key_findings': results['key_findings'],
        'original_filename': results['filename'],
        'upload_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to knowledge base
    success = kb.save_document(doc_data)
    
    if success:
        flash('Document successfully saved to Knowledge Base!')
    else:
        flash('Failed to save document to Knowledge Base')
    
    return redirect(url_for('knowledge_base_page'))

@app.route('/delete_document/<int:doc_id>', methods=['POST'])
def delete_document(doc_id):
    """Delete a document from the knowledge base"""
    success = kb.delete_document(doc_id)
    
    if success:
        flash('Document successfully deleted')
    else:
        flash('Failed to delete document')
    
    return redirect(url_for('knowledge_base_page'))

@app.route('/about')
def about_page():
    """Display the about page"""
    return render_template('about.html')

# Helper functions for temporary data storage
def store_analysis_results(analysis_id, results):
    """Store analysis results in a temporary database"""
    conn = sqlite3.connect('temp_analysis.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            data TEXT,
            timestamp INTEGER
        )
    ''')
    
    # Convert tables to JSON-serializable format
    results_copy = results.copy()
    if 'tables' in results_copy:
        tables_json = []
        for table in results_copy['tables']:
            tables_json.append(table.to_dict())
        results_copy['tables'] = tables_json
    
    # Store results
    cursor.execute(
        'INSERT INTO analysis_results (id, data, timestamp) VALUES (?, ?, ?)',
        (analysis_id, json.dumps(results_copy), int(time.time()))
    )
    
    conn.commit()
    conn.close()

def get_analysis_results(analysis_id):
    """Retrieve analysis results from temporary database"""
    conn = sqlite3.connect('temp_analysis.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            data TEXT,
            timestamp INTEGER
        )
    ''')
    
    # Get results
    cursor.execute('SELECT data FROM analysis_results WHERE id = ?', (analysis_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        results = json.loads(result[0])
        
        # Convert table data back to pandas DataFrames
        if 'tables' in results:
            tables = []
            for table_dict in results['tables']:
                tables.append(pd.DataFrame.from_dict(table_dict))
            results['tables'] = tables
        
        return results
    
    return None

# Clean up old analysis results
def cleanup_old_results():
    """Remove analysis results older than 1 hour"""
    conn = sqlite3.connect('temp_analysis.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            data TEXT,
            timestamp INTEGER
        )
    ''')
    
    # Remove old results
    cursor.execute('DELETE FROM analysis_results WHERE timestamp < ?', (int(time.time()) - 3600,))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Clean up old analysis results
    cleanup_old_results()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)