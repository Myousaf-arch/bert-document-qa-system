import os
from pypdf import PdfReader

def extract_and_chunk_pdfs(uploaded_files, chunk_size=300, overlap=50):
    """
    Multiple uploaded PDF files ko Pure Python pypdf se read karke 
    text extract karta hai aur unhe overlapping chunks mein split karta hai.
    """
    all_text_chunks = []
    
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            file_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    file_text += extracted + " "
            
            # Word-level chunking with overlap
            words = file_text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk.strip()) > 20:
                    all_text_chunks.append(chunk)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            
    return all_text_chunks