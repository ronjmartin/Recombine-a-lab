import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# Create a PDF with sample data
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Financial Report Analysis', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(1)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_table(self, header, data):
        # Table header
        self.set_font('Arial', 'B', 10)
        line_height = 7
        col_width = 40
        
        for item in header:
            self.cell(col_width, line_height, item, 1, 0, 'C')
        self.ln()
        
        # Table data
        self.set_font('Arial', '', 10)
        for row in data:
            for item in row:
                self.cell(col_width, line_height, str(item), 1, 0, 'C')
            self.ln()

# Create PDF
pdf = PDF()
pdf.add_page()

# Executive Summary
pdf.chapter_title("Executive Summary")
pdf.chapter_body("This financial report presents a comprehensive overview of the company's performance for the fiscal year 2023. Overall, the company has demonstrated strong growth in revenue and profitability, with a 12% increase in total revenue compared to the previous year.")

# Key Performance Indicators
pdf.chapter_title("Key Performance Indicators")
pdf.chapter_body("""The company's EBITDA margin improved from 18% to 21%, indicating enhanced operational efficiency. 
Return on capital employed increased from 14.5% to 16.2%, reflecting improved asset utilization.
Customer acquisition cost decreased by 8%, while the customer retention rate improved from 82% to 87%.""")

# Market Analysis
pdf.chapter_title("Market Analysis")
pdf.chapter_body("""The global market size is estimated at $45 billion, with a projected CAGR of 7.8% for the next five years.
Our company currently holds a market share of 12.3%, up from 10.8% in the previous year.
The competitive landscape remains challenging, with three major competitors holding a combined market share of 35%.""")

# Financial Results
pdf.chapter_title("Financial Results")
pdf.chapter_body("The financial results for fiscal year 2023 are as follows:")

# Add table
header = ['Metric', '2022 (millions)', '2023 (millions)', 'YoY Change']
data = [
    ['Revenue', '$342.5', '$383.6', '+12.0%'],
    ['Gross Profit', '$187.4', '$217.6', '+16.1%'],
    ['Operating Expenses', '$125.7', '$136.9', '+8.9%'],
    ['EBITDA', '$61.7', '$80.7', '+30.8%'],
    ['Net Income', '$42.3', '$56.8', '+34.3%'],
    ['Free Cash Flow', '$38.9', '$51.2', '+31.6%']
]
pdf.add_table(header, data)
pdf.ln(10)

# Regional Performance
pdf.chapter_title("Regional Performance")
pdf.chapter_body("""North America continues to be our strongest market, contributing 45% of total revenue.
The European market showed robust growth of 18%, despite economic challenges.
Emerging markets in Asia-Pacific region grew by 22%, representing an increasing share of our total revenue.""")

# Operational Highlights
pdf.chapter_title("Operational Highlights")
pdf.chapter_body("""Successfully launched 3 new product lines, contributing $28.4 million to revenue.
Expanded manufacturing capacity by 15% with the opening of a new facility.
Reduced production costs by 7% through supply chain optimizations and automation initiatives.""")

# Add a new page
pdf.add_page()

# Challenges and Risks
pdf.chapter_title("Challenges and Risks")
pdf.chapter_body("""Increasing raw material costs affected gross margins in Q3 and Q4.
Regulatory changes in European markets may impact operations in the coming year.
Foreign exchange volatility, particularly with the Euro and Yuan, represents an ongoing challenge.""")

# Outlook for 2024
pdf.chapter_title("Outlook for 2024")
pdf.chapter_body("""Projected revenue growth of 10-12% for fiscal year 2024.
Planned capital expenditure of $42 million for technology upgrades and capacity expansion.
Strategic focus on digital transformation and sustainable operations to drive long-term growth.""")

# Conclusion
pdf.chapter_title("Conclusion")
pdf.chapter_body("""The company's strong performance in fiscal year 2023 has positioned it well for continued growth in 2024.
Management remains confident in the company's ability to navigate challenges and capitalize on market opportunities.""")

# Save PDF
pdf.output('sample_data.pdf')
print("PDF created successfully!")