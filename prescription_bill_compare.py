from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from PIL import Image
import os
import google.generativeai as genai
import json
from google.api_core import exceptions
import re

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

GEMINI_API_KEY = "AIzaSyCXE-EbXPuk69gbERDUrzK37gBJEfioomU"
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini Vision API for Text Extraction (Slightly refined prompt) ---
def extract_text_from_image_with_gemini_vision(image_path):
    """Use Gemini Vision API to extract text from a medical document image (handwritten or computerized)."""
    model = genai.GenerativeModel("models/gemini-1.5-flash") # Using a vision-capable model
    try:
        with Image.open(image_path) as img:
            # More general prompt for both handwritten and computerized text, emphasizing all key details
            response = model.generate_content(
                [img, "Extract all text from this medical document image. Pay close attention to details like patient name, hospital name, bill number, pharmacy name, medication names, dosages, individual item amounts, and total bill amount."]
            )
        return response.text.strip()
    except Exception as e:
        return f"Gemini Vision OCR Error: {str(e)}"

# --- Gemini LLM for Prescription Details Extraction ---
def extract_prescription_details_with_gemini_llm(text):
    """
    Uses Gemini's language model to extract patient, hospital, bill number,
    and medication details from the prescription text.
    """
    model = genai.GenerativeModel('models/gemini-1.5-flash')

    cleaned_text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])

    prompt = (
        "Analyze the following medical prescription text. "
        "Identify and extract all medication names either its injection, sachets, capsules, what ever it is in pharmacy terms. "
        "Extract the following details: Patient Name, Hospital Name, and Bill Number. "
        "Also, list all medication names along with their dosages (if present). "
        "Return the results as a JSON object. "
        "The object should have 'patient_name', 'hospital_name', 'bill_number' keys for general details. "
        "It should also have a 'medications' key, which is an array of objects. "
        "For example, if a line says 'Inj. Oxidil 1gm in 100ml N/S', extract 'Oxidil' as name and '1gm in 100ml N/S' as dosage. "
        "If a line says 'Cap. Risek 20mg 1+0+1', extract 'Risek' as name and '20mg 1+0+1' as dosage. "
        "We only need medicine names we dont need to extracct the dosages to compare with the bill. "
        "Each medication object should have only 'name' (for the medication name) keys. for example if a line says 'Inj. Oxidil 1gm in 100ml N/S , it should only extract Oxidil as its name   "
        "If a detail is not found, its value should be an empty string. "
        "If a medication is mentioned without an explicit dosage, set 'dosage' to an empty string. "
        "Prioritize accuracy and handle variations in handwriting and medical abbreviations.\n\n"
        f"Prescription Text:\n{cleaned_text}\n\n"
        "JSON Output:"
    )

    try:
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        }

        response = model.generate_content(prompt, generation_config=generation_config)
        json_str = response.text.strip()

        parsed_data = json.loads(json_str)

        patient_name = parsed_data.get('patient_name', '').strip()
        hospital_name = parsed_data.get('hospital_name', '').strip()
        bill_number = parsed_data.get('bill_number', '').strip()

        medications_list = []
        if 'medications' in parsed_data and isinstance(parsed_data['medications'], list):
            for item in parsed_data['medications']:
                name = item.get('name', '').strip()
                dosage = item.get('dosage', '').strip()
                if name:
                    medications_list.append({'name': name.lower(), 'dosage': dosage.lower()}) # Normalize for comparison

        return {
            'patient_name': patient_name,
            'hospital_name': hospital_name,
            'bill_number': bill_number,
            'medications': medications_list
        }

    except json.JSONDecodeError:
        print(f"[Gemini LLM Prescription] JSON parsing failed. Raw LLM response: {json_str}")
        return {
            'patient_name': '', 'hospital_name': '', 'bill_number': '', 'medications': []
        }
    except exceptions.ResourceExhausted as e:
        print(f"[Gemini LLM Prescription] Resource Exhausted: {str(e)}. Cannot extract prescription details.")
        return {
            'patient_name': '', 'hospital_name': '', 'bill_number': '', 'medications': []
        }
    except Exception as e:
        print(f"[Gemini LLM Prescription] Error with Gemini API for prescription extraction: {str(e)}")
        return {
            'patient_name': '', 'hospital_name': '', 'bill_number': '', 'medications': []
        }

# --- Gemini LLM for Bill Details Extraction ---
def extract_bill_details_with_gemini_llm(text):
    """
    Uses Gemini's language model to extract pharmacy name, individual medication
    items with amounts, and total bill amount from the bill text.
    """
    model = genai.GenerativeModel('models/gemini-1.5-flash')

    cleaned_text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])

    prompt = (
        "Analyze the following medical bill text. "
        "Extract the Pharmacy Name and the Total Bill amount. "
        "Identify and extract all medication names either its injection, sachets, capsules, what ever it is in pharmacy terms. "
        "Also, list all individual medication items shown , along with their associated amounts. "
        "For example, if a line says 'Inj. Oxydil 1gm in 100ml N/S', extract 'Oxidil' as name and '1gm in 100ml N/S' as dosage. "
        "If a line says 'Cap. Risek 20mg 1+0+1', extract 'Risek' as name and '20mg 1+0+1' as dosage. "
        "We only need medicine names we dont need to extracct the dosages to compare with the bill. "
        "Return the results as a JSON object. "
        "The object should have 'pharmacy_name' and 'total_bill' keys for general details. "
        "It should also have an 'items' key, which is an array of objects. "
        "Each item object in the 'items' array should have 'name' (for the medication/item only name for example if a line says 'Inj. Oxidil 1gm in 100ml N/S , it should only extract Oxidil as its name   ) and 'amount' (for its price/cost) keys. "
        "Ensure 'amount' is a float. "
        "If a detail is not found, its value should be an empty string for text, or 0.0 for numbers. "
        "Be robust to variations in formatting and potential OCR errors. Prioritize numerical accuracy for amounts.\n\n"
        f"Bill Text:\n{cleaned_text}\n\n"
        "JSON Output:"
    )

    try:
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        }

        response = model.generate_content(prompt, generation_config=generation_config)
        json_str = response.text.strip()

        parsed_data = json.loads(json_str)

        pharmacy_name = parsed_data.get('pharmacy_name', '').strip()
        total_bill_amount = float(parsed_data.get('total_bill', 0.0))

        items_list = []
        if 'items' in parsed_data and isinstance(parsed_data['items'], list):
            for item in parsed_data['items']:
                name = item.get('name', '').strip()
                amount = float(item.get('amount', 0.0))
                if name:
                    items_list.append({'name': name.lower(), 'amount': amount}) # Normalize name for comparison

        return {
            'pharmacy_name': pharmacy_name,
            'total_bill_amount': total_bill_amount,
            'items': items_list
        }

    except json.JSONDecodeError:
        print(f"[Gemini LLM Bill] JSON parsing failed. Raw LLM response: {json_str}")
        return {
            'pharmacy_name': '', 'total_bill_amount': 0.0, 'items': []
        }
    except exceptions.ResourceExhausted as e:
        print(f"[Gemini LLM Bill] Resource Exhausted: {str(e)}. Cannot extract bill details.")
        return {
            'pharmacy_name': '', 'total_bill_amount': 0.0, 'items': []
        }
    except Exception as e:
        print(f"[Gemini LLM Bill] Error with Gemini API for bill extraction: {str(e)}")
        return {
            'pharmacy_name': '', 'total_bill_amount': 0.0, 'items': []
        }

# --- Comparison and Calculation Function ---
def analyze_data_for_frontend(prescription_data, bill_data):
    """
    Compares prescription medications with bill items, calculates amounts,
    and aggregates all required details for the frontend.
    """
    prescription_med_names = set([med['name'] for med in prescription_data['medications']])

    reimburse_medicines = []
    non_reimburse_medicines = []
    
    reimbursed_amount = 0.0
    non_reimbursed_amount = 0.0
    
    # Track which prescription items were matched to identify truly missing ones
    matched_prescription_names = set()

    for bill_item in bill_data['items']:
        item_name = bill_item['name']
        item_amount = bill_item['amount']

        if item_name in prescription_med_names:
            reimburse_medicines.append({'name': item_name.capitalize(), 'amount': item_amount})
            reimbursed_amount += item_amount
            matched_prescription_names.add(item_name)
        else:
            non_reimburse_medicines.append({'name': item_name.capitalize(), 'amount': item_amount})
            non_reimbursed_amount += item_amount
    
    # Calculate counts
    count_total_medicines = len(bill_data['items'])
    count_reimbursed_medicines = len(reimburse_medicines)
    count_non_reimbursed_medicines = len(non_reimburse_medicines)

    # Ensure total bill matches LLM extracted total if available, otherwise sum items
    # The frontend image explicitly shows 'Total Bill' as a single value.
    # It's best to use the LLM-extracted total bill amount from the bill if it's reliable.
    # If not reliable (e.g., 0), then sum the individual item amounts.
    final_total_bill = bill_data['total_bill_amount']
    if final_total_bill == 0.0 and bill_data['items']:
        final_total_bill = sum(item['amount'] for item in bill_data['items'])
        print("Warning: LLM did not extract total bill. Summing individual item amounts.")
    
    # Assemble data for the frontend
    result = {
        # Top Card Details
        'employee_name': prescription_data['patient_name'], # Mapping Patient Name to Employee Name as per frontend
        'bill_number': prescription_data['bill_number'],
        'hospital_name': prescription_data['hospital_name'],
        'pharmacy_name': bill_data['pharmacy_name'],

        # Reimburse Medicines Table Data
        'reimburse_medicines_data': reimburse_medicines,

        # Non-Reimburse Medicines Table Data
        'non_reimburse_medicines_data': non_reimburse_medicines,

        # Medicine Counts
        'total_medicines_count': count_total_medicines,
        'non_reimburse_medicines_count': count_non_reimbursed_medicines,
        'reimburse_medicines_count': count_reimbursed_medicines,

        # Amount Details
        'total_bill_amount': round(final_total_bill, 2),
        'non_reimburse_amount': round(non_reimbursed_amount, 2),
        'reimburse_amount': round(reimbursed_amount, 2),

        # Raw extracted text for debugging/display if needed
        'raw_prescription_text': prescription_data['raw_text'] if 'raw_text' in prescription_data else 'N/A',
        'raw_bill_text': bill_data['raw_text'] if 'raw_text' in bill_data else 'N/A'
    }
    
    return result

# --- Flask Routes ---
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
        error_msg = f"Error extracting text from one or both images using Gemini Vision. Prescription OCR: {prescription_text}, Bill OCR: {bill_text}"
        print(error_msg) # Log to console
        # Return a JSON error response
        return jsonify({"status": "error", "message": error_msg}), 500

    # Extract detailed data using specific LLM functions
    prescription_parsed_data = extract_prescription_details_with_gemini_llm(prescription_text)
    bill_parsed_data = extract_bill_details_with_gemini_llm(bill_text)

    # Add raw text to the parsed data for potential debugging/display if needed
    prescription_parsed_data['raw_text'] = prescription_text
    bill_parsed_data['raw_text'] = bill_text

    # Perform analysis and calculations to get data structured for frontend
    frontend_data = analyze_data_for_frontend(
        prescription_parsed_data,
        bill_parsed_data
    )
    
    os.remove(prescription_path)
    os.remove(bill_path)

    # Return the structured data as JSON
    return jsonify(frontend_data)

if __name__ == '__main__':
    app.run(debug=True)