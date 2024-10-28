from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
