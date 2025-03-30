import streamlit as st
import os
import time
import pandas as pd
from io import BytesIO
import base64
import sys

# Set page configuration
st.set_page_config(
    page_title="PDF Analysis Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock functions to simulate the behavior without external dependencies
def extract_text_mock(pdf_path):
    """Mock function to simulate text extraction"""
    return "This is a sample extracted text from the PDF. The actual extraction requires PyPDF2 and pdfplumber libraries."

def extract_tables_mock(pdf_path):
    """Mock function to simulate table extraction"""
    # Return a sample dataframe
    return [pd.DataFrame({
        'Column 1': ['Row 1', 'Row 2', 'Row 3'],
        'Column 2': [100, 200, 300],
        'Column 3': ['A', 'B', 'C']
    })]

def analyze_sentiment_mock(text):
    """Mock function to simulate sentiment analysis"""
    return {
        'overall_sentiment': 'Positive',
        'sentiment_score': 0.75,
        'emotions': {
            'Joy': 0.6,
            'Trust': 0.5,
            'Anticipation': 0.4,
            'Surprise': 0.3,
            'Fear': 0.2,
            'Sadness': 0.1
        },
        'section_sentiments': [0.8, 0.7, 0.6, 0.5]
    }

def extract_key_findings_mock(text):
    """Mock function to simulate key findings extraction"""
    return {
        'topics': ['Topic 1', 'Topic 2', 'Topic 3'],
        'entities': [
            {'text': 'Entity 1', 'label': 'PERSON', 'count': 5},
            {'text': 'Entity 2', 'label': 'ORG', 'count': 3},
            {'text': 'Entity 3', 'label': 'LOC', 'count': 2}
        ],
        'summary': 'This is a sample summary of the document.'
    }

# Simple database simulation
class KnowledgeBase:
    def __init__(self):
        self.documents = []
        self.next_id = 1
    
    def save_document(self, doc_data):
        doc_data['id'] = self.next_id
        self.next_id += 1
        self.documents.append(doc_data)
        return True
    
    def get_all_documents(self):
        return self.documents
    
    def get_document(self, doc_id):
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def search_documents(self, query):
        # Simple search by title or description
        results = []
        for doc in self.documents:
            if (query.lower() in doc['title'].lower() or 
                (doc['description'] and query.lower() in doc['description'].lower())):
                doc_copy = doc.copy()
                doc_copy['relevance'] = 0.95  # Mock relevance score
                results.append(doc_copy)
        return results

# Initialize knowledge base
kb = KnowledgeBase()

# Visualization functions
def create_visualization(df, x_col, y_col, viz_type):
    """Simple visualization function"""
    st.write(f"Visualization of {x_col} vs {y_col} using {viz_type}")
    return df.plot.bar(x=x_col, y=y_col, figsize=(10, 5))

def create_sentiment_gauge(sentiment_score):
    """Create a simple gauge display for sentiment"""
    import numpy as np
    
    # Create gauge chart using a simple progress bar
    score_normalized = (sentiment_score + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
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
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
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
            # Extract text from PDF (mock function)
            text_content = extract_text_mock("temp.pdf")
            
            # Extract tables from PDF (mock function)
            tables = extract_tables_mock("temp.pdf")
            
            # Analyze sentiment (mock function)
            sentiment = analyze_sentiment_mock(text_content)
            
            # Extract key findings (mock function)
            key_findings = extract_key_findings_mock(text_content)
        
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
            st.markdown("### Extracted Tables")
            
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
                    
                    # Visualize table
                    st.markdown("#### Visualization")
                    viz_type = st.selectbox(
                        f"Choose visualization for Table {i+1}",
                        ["Bar Chart", "Line Chart", "Scatter Plot"],
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
            
            # Sentiment gauge chart
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
                    success = kb.save_document(doc_data)
                    
                    if success:
                        st.success("Document successfully saved to Knowledge Base!")
                    else:
                        st.error("Failed to save document to Knowledge Base")
        
        # Clean up temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

def knowledge_base_page():
    st.title("Knowledge Base")
    
    # Tabs for Browse and Search
    kb_tabs = st.tabs(["Browse Documents", "Search"])
    
    # Tab 1: Browse Documents
    with kb_tabs[0]:
        st.markdown("### Browse Documents")
        
        # Get all documents
        documents = kb.get_all_documents()
        
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
            search_results = kb.search_documents(search_query)
            
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
    document = kb.get_document(doc_id)
    
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
    st.title("About the PDF Analysis Dashboard")
    
    st.markdown("""
    This dashboard provides tools for analyzing PDF documents, extracting insights, and managing a knowledge base of analyzed documents.
    
    ## Features
    
    - **Text Extraction**: Extract text content from PDF files
    - **Table Extraction**: Identify and extract tables from PDF documents
    - **Sentiment Analysis**: Analyze the sentiment of the extracted text
    - **Key Findings**: Extract important topics, entities, and generate a summary
    - **Knowledge Base**: Save analyzed documents for future reference
    - **Search**: Search through your knowledge base for specific documents
    
    ## Technologies Used
    
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation
    - **Python**: Core programming language
    
    ## System Information
    """)
    
    # Display system information
    st.write(f"Python Version: {sys.version}")
    st.write(f"Python Executable: {sys.executable}")
    st.write(f"Current Working Directory: {os.getcwd()}")

if __name__ == "__main__":
    main()