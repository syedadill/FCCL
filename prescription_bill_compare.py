from flask import Flask, request, render_template, flash, redirect, url_for
from PIL import Image
import os
import google.generativeai as genai
import json # Import json for parsing Gemini's output
from google.api_core import exceptions
import re # Keep re for some basic text cleaning/pre-filtering for the prompt

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Required for flash messages
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Gemini API (replace with your API key)
GEMINI_API_KEY = "AIzaSyCXE-EbXPuk69gbERDUrzK37gBJEfioomU" # IMPORTANT: Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Updated prompt for extract_text_from_image_with_gemini_vision
def extract_text_from_image_with_gemini_vision(image_path):
    """Use Gemini Vision API to extract text from a medical document image (handwritten or computerized)."""
    model = genai.GenerativeModel("models/gemini-1.5-flash") # Using a vision-capable model
    try:
        with Image.open(image_path) as img:
            # More general prompt for both handwritten and computerized text
            response = model.generate_content(
                [img, "Extract all text from this medical document image. Pay close attention to details like medication names, dosages, and other relevant medical information."]
            )
        return response.text.strip()
    except Exception as e:
        return f"Gemini Vision OCR Error: {str(e)}"

# --- Gemini LLM for Medication and Dosage Extraction ---
def extract_medications_and_dosages_with_gemini_llm(text):
    """
    Uses Gemini's language model to extract medication names and their dosages
    from the provided text.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Use a capable model
    
    # Clean up text a bit before sending to Gemini for better results
    cleaned_text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])

    # Craft a precise prompt for Gemini to extract medications and dosages
    # Emphasize JSON output format for easy parsing
    prompt = (
        "Analyze the following medical prescription text. "
        "Identify and extract all medication names along with their exact dosages and frequencies (if present). "
        "Return the results as a JSON array of objects. "
        "Use the provided list of known medicine names to validate and correct potential misspellings due to handwriting. "
        "For example, if a line says 'Inj. Oxydil 1gm in 100ml N/S', extract 'Oxidil' as name and '1gm in 100ml N/S' as dosage. "
        "If a line says 'Cap. Risek 20mg 1+0+1', extract 'Risek' as name and '20mg 1+0+1' as dosage. "
        "Each object should have two keys: 'name' for the medication name and 'dosage' for the dosage and frequency information. "
        "If a medication is mentioned without an explicit dosage or frequency, set 'dosage' to an empty string. "
        "Prioritize accuracy and handle variations in handwriting and common medical abbreviations.\n\n"
        f"Prescription Text:\n{cleaned_text}\n\n"
        "JSON Output:"
    )

    try:
        # Use generation_config to ensure JSON output
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.1, # Lower temperature for more deterministic output
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        json_str = response.text.strip()

        meds_data = json.loads(json_str)
        
        results = []
        # Gemini might return a list directly or a dict with a key like 'medications'
        if isinstance(meds_data, list):
            items_to_process = meds_data
        elif isinstance(meds_data, dict) and "medications" in meds_data:
            items_to_process = meds_data["medications"]
        else: # Fallback if structure is unexpected
            print(f"Unexpected JSON structure from Gemini: {json_str}")
            return []

        for item in items_to_process:
            name = item.get("name", "").strip()
            dosage = item.get("dosage", "").strip()
            if name: # Only add if a name was extracted
                formatted_item = name
                if dosage:
                    formatted_item += f" ({dosage})"
                results.append(formatted_item.lower()) # Normalize to lowercase for comparison

        return results

    except json.JSONDecodeError:
        print(f"[Gemini LLM] JSON parsing failed. Raw LLM response: {json_str}")
        # If Gemini doesn't return perfect JSON, try a simpler regex fallback
        # This is a last resort, as the goal is for Gemini to return JSON
        meds = re.findall(r'"name"\s*:\s*"([^"]+)"(?:,\s*"dosage"\s*:\s*"([^"]*)")?', json_str, re.IGNORECASE)
        results = []
        for name, dosage in meds:
            formatted = name.strip()
            if dosage:
                formatted += f" ({dosage.strip()})"
            results.append(formatted.lower())
        return results
    except exceptions.ResourceExhausted as e:
        print(f"[Gemini LLM] Resource Exhausted: {str(e)}. Cannot extract medications.")
        return []
    except Exception as e:
        print(f"[Gemini LLM] Error with Gemini API for medication extraction: {str(e)}")
        return []

# --- Comparison Function (remains largely the same) ---
def compare_lists(prescription_items, bill_items):
    """Compare prescription and bill items, return matches and discrepancies."""
    # Strip dosages for comparison (e.g., "vibramycin (100MG CAPS)" -> "vibramycin")
    # This also normalizes to lowercase for case-insensitive comparison
    prescription_names = [item.split(' (')[0].lower() for item in prescription_items]
    bill_names = [item.split(' (')[0].lower() for item in bill_items]

    prescription_set = set(prescription_names)
    bill_set = set(bill_names)

    matches_names = prescription_set.intersection(bill_set)
    extra_names = bill_set - prescription_set
    missing_names = prescription_set - bill_set

    # Reconstruct original items (with dosages) for display
    matches = [item for item in prescription_items if item.split(' (')[0].lower() in matches_names]
    extra_items = [item for item in bill_items if item.split(' (')[0].lower() in extra_names]
    missing_items = [item for item in prescription_items if item.split(' (')[0].lower() in missing_names]

    return matches, extra_items, missing_items

# --- Flask Routes (updated to use new functions) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'prescription' not in request.files or 'bill' not in request.files:
        flash('Please upload both prescription and bill images.')
        return redirect(url_for('index'))

    prescription_file = request.files['prescription']
    bill_file = request.files['bill']

    if prescription_file.filename == '' or bill_file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    prescription_path = os.path.join(app.config['UPLOAD_FOLDER'], prescription_file.filename)
    bill_path = os.path.join(app.config['UPLOAD_FOLDER'], bill_file.filename)
    prescription_file.save(prescription_path)
    bill_file.save(bill_path)

    # Use Gemini Vision API for initial text extraction
    prescription_text = extract_text_from_image_with_gemini_vision(prescription_path)
    bill_text = extract_text_from_image_with_gemini_vision(bill_path)

    if "Gemini Vision OCR Error" in prescription_text or "Gemini Vision OCR Error" in bill_text:
        flash(f"Error extracting text from one or both images using Gemini Vision. Prescription OCR: {prescription_text}, Bill OCR: {bill_text}")
        os.remove(prescription_path)
        os.remove(bill_path)
        return redirect(url_for('index'))

    # Use Gemini LLM for medication and dosage extraction from the extracted text
    prescription_items = extract_medications_and_dosages_with_gemini_llm(prescription_text)
    bill_items = extract_medications_and_dosages_with_gemini_llm(bill_text)

    # Compare lists
    matches, extra_items, missing_items = compare_lists(prescription_items, bill_items)

    result = {
        'matches': list(matches),
        'extra_items': list(extra_items),
        'missing_items': list(missing_items),
        'prescription_text': prescription_text,
        'bill_text': bill_text
    }

    os.remove(prescription_path)
    os.remove(bill_path)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)