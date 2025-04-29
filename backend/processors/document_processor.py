from typing import Dict, List
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

def debug_print(msg: str, obj=None):
    """Debug printing with optional object inspection."""
    print(f"\nDEBUG: {msg}")
    if obj is not None:
        print(f"Type: {type(obj)}")
        print(f"Content: {str(obj)[:200]}...")

def extract_subheadings(text: str, chat_model) -> List[str]:
    """Extract subheadings using NVIDIA chat model."""
    prompt = f"""
    You are an AI assistant specialized in analyzing medical documents. Your task is to extract all subheadings from the given medical note.
    Instructions:
    1. Carefully read through the entire medical note.
    2. Identify all subheadings that introduce new sections of the note.
    3. Subheadings are typically short, capitalized phrases, often followed by a colon.
    4. Include numeric or bullet-point headings if present.
    5. Correct any obvious spelling mistakes in subheadings (e.g., "DIAGNSOIS" should be "DIAGNOSIS").
    6. Do not include patient identifiers or metadata.
    7. Exclude any repeated subheadings.
    Output Format:
    - List only the extracted subheadings, one per line.
    - Use consistent capitalization (preferably all caps).
    - Do not include any additional text, explanations, or content under the subheadings.
    
    Analyze the following medical note and extract all subheadings:
    {text}
    List of extracted subheadings:
    """
    
    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        subheadings = response.content.strip().split('\n')
        debug_print(f"Successfully extracted {len(subheadings)} subheadings", subheadings)
        return subheadings
    except Exception as e:
        print(f"Error extracting subheadings: {e}")
        return []

def extract_section_content(text: str, subheading: str, chat_model) -> str:
    """Extract content for a specific subheading."""
    prompt = f"""
    You are an AI assistant specialized in analyzing medical documents. Your task is to extract key information related to a specific subheading from a medical note.
    Instructions:
    1. Focus on the subheading: "{subheading}"
    2. Extract all relevant key-value pairs and information related to this subheading from the provided document.
    3. Include all important details, multiple key-value pairs, and any lists (e.g., medications) under this subheading.
    4. Provide only factual information explicitly stated in the document. Do not infer or add details not present.
    5. Format the output concisely, without explanations or commentary.
    6. Include time information with laboratory results if available.
    7. If information is missing or redacted, indicate this with [REDACTED] or [NOT PROVIDED].
    8. Ensure all lists and information are complete, not cut off mid-sentence or mid-list.
    9. For lab results, include the date/time and all relevant values.
    
    Medical Note:
    {text}
    Extract and summarize all relevant information for the subheading: {subheading}
    """
    
    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        debug_print(f"Extracted content for {subheading}", content)
        return content
    except Exception as e:
        print(f"Error extracting content for {subheading}: {e}")
        return f"{subheading}: ERROR EXTRACTING INFORMATION"

def process_note_sections(note_text: str, metadata: Dict, chat_model) -> List[Dict]:
    """Process note text into sections and chunks."""
    debug_print("Starting note processing")
    
    # Extract subheadings
    subheadings = extract_subheadings(note_text, chat_model)
    debug_print(f"Found {len(subheadings)} subheadings")
    
    # Process each section
    processed_chunks = []
    
    for subheading in subheadings:
        try:
            # Extract content for this subheading
            section_content = extract_section_content(note_text, subheading, chat_model)
            
            # Split into chunks if content is long
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = splitter.split_text(section_content)
            debug_print(f"Split section '{subheading}' into {len(chunks)} chunks")
            
            # Store chunks with metadata
            for chunk in chunks:
                processed_chunks.append({
                    "section": subheading,
                    "content": chunk,
                    "metadata": metadata
                })
                
        except Exception as e:
            print(f"Error processing section {subheading}: {e}")
            continue
    
    return processed_chunks
