from PyPDF2 import PdfReader
import pandas as pd
import json
import os

def process_discharge_notes(csv_file_path, save_dir):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the CSV file
    discharge_note = pd.read_csv(csv_file_path)
    
    # Process each row
    for _, row in discharge_note.iterrows():
        # Convert row to a dictionary
        row_dict = row.to_dict()
        
        # Get the admission ID (hadm_id) for naming the JSON file
        admission_id = row_dict['hadm_id']
        
        # Define the JSON file name with the specified directory
        json_file_name = os.path.join(save_dir, f"note_discharge_{admission_id}_13180007.json")
        
        # Write the row data to a JSON file
        with open(json_file_name, 'w') as json_file:
            json.dump(row_dict, json_file, indent=4)

'''
# Define paths
csv_file_path_discharge = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/csv/note_discharge_13180007.csv"
json_save_dirdischarge = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge"

# Run the function with the defined paths
process_discharge_notes(csv_file_pathdischarge, json_save_dirdischarge)
'''

def process_radiology_notes(csv_file_path, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the CSV file
    radiology_note = pd.read_csv(csv_file_path)
    
    # Drop rows where hadm_id is empty
    radiology_note = radiology_note.dropna(subset=['hadm_id'])
    
    # Drop the first column
    radiology_note = radiology_note.drop(radiology_note.columns[0], axis=1) 
        
    # Convert hadm_id and note_seq to integers
    radiology_note['hadm_id'] = radiology_note['hadm_id'].astype(int)
    radiology_note['note_seq'] = radiology_note['note_seq'].astype(int)
    
    # Group by hadm_id and sort each group by note_seq
    grouped = radiology_note.groupby('hadm_id')
    
    for hadm_id, group in grouped:
        # Sort the group by note_seq
        group = group.sort_values(by='note_seq')
        
        # Convert each row in the group to a dictionary, excluding 'hadm_id' as it will be the key
        notes = group.to_dict(orient='records')
        
        # Create the structured JSON format
        json_content = {
            "hadm_id": hadm_id,
            "notes": notes
        }
        
        # Define the JSON file name
        json_file_name = os.path.join(save_dir, f"note_radiology_{hadm_id}_13180007.json")
        
        # Write the JSON file
        with open(json_file_name, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)

'''
# Example usage
csv_file_path_radiology = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/csv/note_radiology_13180007.csv"
json_save_dir_radiology = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/json/radiology"

process_radiology_notes(csv_file_path_radiology, json_save_dir_radiology)
'''

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text