import pandas as pd
import os
import base64
from io import BytesIO
import streamlit as st

# Create a simple PDF file for testing
def create_test_pdf():
    try:
        # Create a simple text file
        with open('simple_test.txt', 'w') as f:
            f.write("""# Financial Report 2023
            
This is a simple test document that contains some financial information for testing.

The company has shown strong growth with revenue increasing by 15% compared to last year.

Here are some key statistics:
- Revenue: $5.2 million
- Profit: $1.2 million
- Expenses: $4.0 million

This document is meant for testing the PDF Analysis Dashboard application.
            """)
            
        # Try to convert to PDF using pandoc if available
        try:
            os.system('pandoc simple_test.txt -o test_file.pdf')
            if os.path.exists('test_file.pdf'):
                return 'test_file.pdf'
        except:
            pass
            
        # If pandoc fails, create a simple HTML file and print instructions
        with open('simple_test.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Test PDF</title>
</head>
<body>
    <h1>Financial Report 2023</h1>
    <p>This is a simple test document that contains some financial information for testing.</p>
    <p>The company has shown strong growth with revenue increasing by 15% compared to last year.</p>
    <h2>Key Statistics:</h2>
    <ul>
        <li>Revenue: $5.2 million</li>
        <li>Profit: $1.2 million</li>
        <li>Expenses: $4.0 million</li>
    </ul>
    <p>This document is meant for testing the PDF Analysis Dashboard application.</p>
    <table border="1">
        <tr>
            <th>Quarter</th>
            <th>Revenue</th>
            <th>Profit</th>
        </tr>
        <tr>
            <td>Q1</td>
            <td>$1.2M</td>
            <td>$0.3M</td>
        </tr>
        <tr>
            <td>Q2</td>
            <td>$1.3M</td>
            <td>$0.3M</td>
        </tr>
        <tr>
            <td>Q3</td>
            <td>$1.3M</td>
            <td>$0.3M</td>
        </tr>
        <tr>
            <td>Q4</td>
            <td>$1.4M</td>
            <td>$0.3M</td>
        </tr>
    </table>
</body>
</html>
            """)
            
        return "Could not create PDF. Please use a real PDF file for testing."
    except Exception as e:
        return f"Error creating test file: {str(e)}"

if __name__ == "__main__":
    result = create_test_pdf()
    print(result)