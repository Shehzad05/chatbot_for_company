import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

def scrape_website_to_pdf(url, output_filename):
    try:
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Parse the webpage content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract content (modify selectors as needed)
        title = soup.title.string if soup.title else "No Title"
        paragraphs = soup.find_all('p')
        
        # Initialize PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add Title
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(0, 10, title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
        pdf.ln(10)  # Add a blank line
        
        # Add Paragraphs
        pdf.set_font("Arial", size=12)
        for para in paragraphs:
            text = para.get_text().strip()
            if text:  # Avoid empty lines
                pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
                pdf.ln(5)  # Add spacing between paragraphs
        
        # Save PDF
        pdf.output(output_filename)
        print(f"Content saved to {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
url = "https://reactivespace.com/"  # Replace with your target URL
output_filename = "company_profile.pdf"
scrape_website_to_pdf(url, output_filename)
