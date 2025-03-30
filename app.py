import streamlit as st
import os
import time
import pandas as pd
from io import BytesIO

# Import custom modules
import pdf_extraction as pdf_ext
import sentiment_analysis as sa
import knowledge_base as kb
import visualization as viz

# Set page configuration
st.set_page_config(
    page_title="PDF Analysis Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize knowledge base
kb.initialize_db()

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
            # Extract text from PDF
            text_content = pdf_ext.extract_text("temp.pdf")
            
            # Extract tables from PDF
            tables = pdf_ext.extract_tables("temp.pdf")
            
            # Analyze sentiment
            sentiment = sa.analyze_sentiment(text_content)
            
            # Extract key findings
            key_findings = sa.extract_key_findings(text_content)
        
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
                            fig = viz.create_visualization(table, x_col, y_col, viz_type)
                            st.plotly_chart(fig)
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
            fig = viz.create_sentiment_gauge(sentiment['sentiment_score'])
            st.plotly_chart(fig)
            
            # Display emotion breakdown
            st.markdown("#### Emotion Breakdown")
            emotion_df = pd.DataFrame({
                'Emotion': list(sentiment['emotions'].keys()),
                'Score': list(sentiment['emotions'].values())
            })
            emotion_df = emotion_df.sort_values('Score', ascending=False)
            
            # Bar chart for emotions
            fig = viz.create_emotion_chart(emotion_df)
            st.plotly_chart(fig)
            
            # Display detailed sentiment by section
            if 'section_sentiments' in sentiment and sentiment['section_sentiments']:
                st.markdown("#### Sentiment by Document Section")
                
                section_df = pd.DataFrame({
                    'Section': [f"Section {i+1}" for i in range(len(sentiment['section_sentiments']))],
                    'Score': sentiment['section_sentiments']
                })
                
                fig = viz.create_section_sentiment_chart(section_df)
                st.plotly_chart(fig)
        
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
                fig = viz.create_entity_chart(key_findings['entities'])
                st.plotly_chart(fig)
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
            doc_df = pd.DataFrame(documents)
            st.dataframe(doc_df[['id', 'title', 'description', 'upload_time']])
            
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
                search_df = pd.DataFrame(search_results)
                st.dataframe(search_df[['id', 'title', 'description', 'relevance']])
                
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
                for i, table_data in enumerate(document['tables']):
                    try:
                        # Convert JSON table data back to DataFrame
                        table = pd.DataFrame(table_data)
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
                fig = viz.create_sentiment_gauge(sentiment['sentiment_score'])
                st.plotly_chart(fig)
                
                # Display emotion breakdown
                if 'emotions' in sentiment:
                    st.markdown("#### Emotion Breakdown")
                    emotion_df = pd.DataFrame({
                        'Emotion': list(sentiment['emotions'].keys()),
                        'Score': list(sentiment['emotions'].values())
                    })
                    emotion_df = emotion_df.sort_values('Score', ascending=False)
                    
                    # Bar chart for emotions
                    fig = viz.create_emotion_chart(emotion_df)
                    st.plotly_chart(fig)
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
                    fig = viz.create_entity_chart(key_findings['entities'])
                    st.plotly_chart(fig)
                else:
                    st.info("No significant entities detected")
                
                # Display summary
                st.markdown("#### Summary")
                if key_findings['summary']:
                    st.write(key_findings['summary'])
                else:
                    st.info("No summary available")
            else:
                st.info("No key findings available for this document")
        
        # Delete document option
        if st.button("Delete Document", key=f"delete_{doc_id}"):
            if kb.delete_document(doc_id):
                st.success("Document successfully deleted")
                st.rerun()
            else:
                st.error("Failed to delete document")

def about_page():
    st.title("About PDF Analysis Dashboard")
    
    st.markdown("""
    ## Overview
    
    This application provides a comprehensive toolkit for analyzing PDF documents. 
    You can upload PDFs, extract text and tabular data, perform sentiment analysis, 
    identify key insights, and save everything to a knowledge base for future reference.
    
    ## Features
    
    - **PDF Upload & Text Extraction**: Extract and view the textual content of PDF files
    - **Table Extraction**: Identify and extract tables from PDFs with formatting preserved
    - **Sentiment Analysis**: Analyze the emotional tone and subjectivity of the document
    - **Key Findings**: Extract important entities, topics, and generate a summary
    - **Knowledge Base**: Store analyzed documents for future reference and searching
    - **Visualization**: Visualize data extracted from PDFs for better understanding
    
    ## How to Use
    
    1. Navigate to the "Upload & Analyze PDF" section
    2. Upload a PDF document using the file uploader
    3. View the extracted text, tables, sentiment analysis, and key findings
    4. Save important documents to the knowledge base
    5. Use the knowledge base to browse or search for previously analyzed documents
    
    ## Technologies Used
    
    - **Streamlit**: Web interface
    - **PyPDF2/PDFPlumber**: PDF text extraction
    - **Camelot/Tabula**: Table extraction
    - **NLTK/spaCy**: Natural language processing and sentiment analysis
    - **Pandas**: Data manipulation
    - **Plotly**: Interactive data visualization
    - **SQLite**: Lightweight database for knowledge base storage
    """)

if __name__ == "__main__":
    main()
