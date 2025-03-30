import PyPDF2
import pdfplumber
import pandas as pd
import camelot
import io
import os
import traceback
import tempfile
from typing import List, Dict, Any, Optional, Tuple

def extract_text(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    text = ""
    
    # Try using PyPDF2 first
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
    except Exception as e:
        print(f"PyPDF2 extraction failed: {str(e)}")
        # If PyPDF2 fails, try pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                
                for page_num in range(num_pages):
                    page = pdf.pages[page_num]
                    text += page.extract_text() or "" + "\n\n"
        except Exception as e2:
            print(f"pdfplumber extraction failed: {str(e2)}")
            text = f"Text extraction failed: {str(e2)}"
    
    return text.strip()

def extract_tables(pdf_path: str) -> List[pd.DataFrame]:
    """
    Extract tables from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[pd.DataFrame]: List of extracted tables as pandas DataFrames
    """
    tables = []
    
    # Try camelot first (better for tables with borders)
    try:
        # Check if file exists and is readable
        if not os.path.exists(pdf_path):
            return tables
        
        # Extract tables using camelot
        camelot_tables = camelot.read_pdf(
            pdf_path,
            pages='all',
            flavor='lattice',  # Try lattice first (for tables with borders)
            suppress_stdout=True
        )
        
        # If no tables found with lattice, try stream flavor
        if len(camelot_tables) == 0:
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages='all',
                flavor='stream',  # For tables without clear borders
                suppress_stdout=True
            )
        
        # Convert to pandas DataFrames and add to list
        for table in camelot_tables:
            df = table.df
            # Clean up table by setting the first row as header if it doesn't have one
            if not any(col.strip() for col in df.columns):
                header_row = df.iloc[0]
                df = df[1:]
                df.columns = header_row
            
            # Drop empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            if not df.empty:
                tables.append(df)
    
    # If camelot fails, try pdfplumber
    except Exception as e:
        print(f"Camelot extraction failed: {str(e)}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table_data in page_tables:
                        if table_data:
                            # Convert to DataFrame
                            df = pd.DataFrame(table_data)
                            
                            # Use first row as header
                            if len(df) > 0:
                                header_row = df.iloc[0]
                                df = df[1:]
                                df.columns = header_row
                                
                                # Drop empty rows and columns
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                
                                # Reset index
                                df = df.reset_index(drop=True)
                                
                                if not df.empty:
                                    tables.append(df)
        except Exception as e2:
            print(f"pdfplumber table extraction failed: {str(e2)}")
    
    return tables

def process_pdf(pdf_file) -> Tuple[str, List[pd.DataFrame]]:
    """
    Process a PDF file to extract text and tables
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        Tuple[str, List[pd.DataFrame]]: Extracted text and tables
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_path = temp_file.name
        # Write uploaded file to temporary file
        temp_file.write(pdf_file.read())
    
    try:
        # Extract text and tables
        text = extract_text(temp_path)
        tables = extract_tables(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return text, tables
    except Exception as e:
        # Clean up temporary file
        os.unlink(temp_path)
        raise e
