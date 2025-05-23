import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch
import os

# Set your Gemini API key here
GOOGLE_API_KEY = "AIzaSyBtnIa3ZOht48GMTOHnoXRf23i4K8hp3Po"  # <-- Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# List of known medicine names (expand as needed)
common_medicines = [
    'paracetamol', 'amoxicillin', 'metformin', 'atorvastatin', 'ibuprofen',
    'omeprazole', 'azithromycin', 'aspirin', 'lisinopril', 'simvastatin'
]

def extract_text_from_image(image_path):
    """Use Gemini Vision API to extract text from a handwritten prescription image."""
    model = genai.GenerativeModel("models/gemini-1.5-flash")  # Updated model
    with Image.open(image_path) as img:
        response = model.generate_content(
            [img, "Extract the handwritten prescription text clearly."]
        )
    return response.text.strip()




def analyze_text_with_clinicalbert(text):
    """Tokenize the text using ClinicalBERT for context-aware processing."""
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens

def extract_medicines_from_text(text):
    """Match known medicines from the text."""
    found = []
    text_lower = text.lower()
    for med in common_medicines:
        if med in text_lower:
            found.append(med)
    return found

def main():
    image_path = "/home/nazar/Muhammad Raees Azam/Medical Reporting/1 (2).png"  # <-- Replace with your actual image filename
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path '{image_path}'")
        return

    print("🔍 Extracting prescription text from image...")
    prescription_text = extract_text_from_image(image_path)
    print("\n📄 Extracted Text:\n", prescription_text)

    print("\n🔬 Analyzing with ClinicalBERT...")
    tokens = analyze_text_with_clinicalbert(prescription_text)
    print("\n🧠 Tokens:\n", tokens)

    print("\n💊 Identifying medicines...")
    medicines = extract_medicines_from_text(prescription_text)
    print("\n✅ Medicines Found in Prescription:\n", medicines if medicines else "No known medicines found.")

if __name__ == "__main__":
    main()