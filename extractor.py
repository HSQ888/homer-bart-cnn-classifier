import zipfile
import os

def extract_zip(zip_path):
    """Extract a zip file to its directory."""
    extract_dir = os.path.dirname(zip_path)

    if not os.path.exists(extract_dir):
        print('Error : Zip file not found')
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"File Extracted to the main directory")

if __name__ == "__main__":
    zip_file = "data.zip"  # Change to your zip file name
    extract_zip(zip_file)