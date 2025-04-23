import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz, process
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download all necessary NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    print("NLTK data download completed.")

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and model
print("Initializing BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the model weights
try:
    model_path = os.environ.get('MODEL_PATH', 'internship_selection_model.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Using base BERT model.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Continuing with base BERT model...")

model.to(device)
model.eval()

def analyze_regional_priority(town):
    """Uses fuzzy matching to determine regional priority based on Avalon vs non-Avalon regions"""
    if pd.isna(town):
        return 0.0, "No location provided"

    town = str(town).lower()

    # Define Avalon region locations (standard priority)
    avalon_locations = {
        'st. john\'s', 'mount pearl', 'paradise', 'conception bay south',
        'torbay', 'portugal cove-st. philip\'s', 'flatrock', 'pouch cove',
        'bauline', 'witless bay', 'bay bulls', 'holyrood', 'seal cove',
        'fox trap', 'long pond', 'kelligrews', 'upper gullies', 'foxtrap',
        'topsail', 'chamberlains', 'logy bay', 'middle cove', 'outer cove',
        'petty harbour', 'maddox cove', 'goulds', 'kilbride', 'bell island',
        'wabana', 'placentia', 'st. bride\'s', 'dunville', 'freshwater'
    }

    best_match = process.extractOne(
        town,
        avalon_locations,
        scorer=fuzz.token_set_ratio,
        score_cutoff=80
    )

    if best_match:
        return 0.5, f"Standard priority - Avalon Peninsula region (matched with {best_match[0]})"
    else:
        return 1.0, f"High priority - Non-Avalon region ({town})"

def clean_and_lemmatize(text):
    """Clean and lemmatize text"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def calculate_diversity_score(student_data):
    """Calculate diversity score based on student data"""
    score = 0.0
    reasons = []

    # Gender diversity
    if student_data.get('identify_as', '').lower() in ['female', 'non-binary', 'prefer not to say']:
        score += 0.5
        reasons.append(f"Gender diversity factor: +0.5 ({student_data.get('identify_as')})")

    # Location diversity using regional priority
    location_score, location_reason = analyze_regional_priority(student_data.get('town', ''))
    score += location_score
    reasons.append(f"Location diversity factor: +{location_score} ({location_reason})")

    final_score = min(score, 1.0)
    return final_score, reasons

def get_detailed_reasoning(student_data, predicted_label, confidence, diversity_reasons):
    """Generate detailed reasoning based on the model's decision and student data"""

    def extract_significant_terms(text, min_word_length=4):
        """Extract significant terms from text, excluding common words"""
        if not isinstance(text, str):
            return []

        # Tokenize and clean the text
        text = text.lower()
        words = word_tokenize(text)

        # Filter out common words, short words, and non-alphabetic words
        significant_terms = [
            word for word in words
            if word not in stop_words
            and len(word) >= min_word_length
            and word.isalpha()
        ]

        return list(set(significant_terms))  # Remove duplicates

    # Perform detailed analysis
    experience_analysis = {
        'technical_terms': extract_significant_terms(student_data['tech_experience']),
        'non_technical_terms': extract_significant_terms(student_data.get('non_tech_experience', '')),
        'tech_depth': len(extract_significant_terms(student_data['tech_experience'])),
        'non_tech_depth': len(extract_significant_terms(student_data.get('non_tech_experience', '')))
    }

    motivation_analysis = {
        'motivation_terms': extract_significant_terms(student_data['why_internship']),
        'goal_terms': extract_significant_terms(student_data['goals']),
        'motivation_depth': len(extract_significant_terms(student_data['why_internship'])),
        'goal_clarity': len(extract_significant_terms(student_data['goals']))
    }

    # Generate reasoning
    reasoning = f"\nSelection Decision (Model is {'very confident' if confidence >= 0.80 else 'confident' if confidence >= 0.70 else 'moderately confident'} - {confidence*100:.2f}%)\n"
    reasoning += "="*80 + "\n"

    if predicted_label == 1:  # Selected
        reasoning += "SELECTED for internship\n\n"

        # Technical Analysis
        reasoning += "Technical Background Analysis:\n"
        reasoning += f"Key technical terms mentioned: {', '.join(experience_analysis['technical_terms'])}\n"
        reasoning += f"Technical depth indicators: {experience_analysis['tech_depth']} unique significant terms\n"

        # Motivation Analysis
        reasoning += "\nMotivation and Goals Analysis:\n"
        reasoning += f"Motivation indicators: {', '.join(motivation_analysis['motivation_terms'])}\n"
        reasoning += f"Career goal indicators: {', '.join(motivation_analysis['goal_terms'])}\n"

    else:  # Not Selected
        reasoning += "NOT SELECTED for internship\n\n"

        # Technical Analysis
        reasoning += "Technical Background Analysis:\n"
        if experience_analysis['tech_depth'] > 0:
            reasoning += f"Technical terms mentioned: {', '.join(experience_analysis['technical_terms'])}\n"
            reasoning += "While showing some technical knowledge, may need more depth or practical application\n"
        else:
            reasoning += "Limited technical terminology or specific technical experience mentioned\n"

    # Add diversity factors
    reasoning += "\nDiversity Considerations:\n"
    for reason in diversity_reasons:
        reasoning += f"- {reason}\n"

    return reasoning

def predict_for_student(sample_data):
    """Make prediction for a single student"""
    # Calculate diversity score with reasons
    diversity_score, diversity_reasons = calculate_diversity_score(sample_data)
    sample_data['diversity_score'] = diversity_score

    # Prepare input data
    fields_to_combine = [
        'town', 'diversity_score', 'why_internship',
        'tech_experience', 'non_tech_experience',
        'goals', 'other_comments'
    ]

    # Clean and combine fields
    cleaned_fields = []
    for field in fields_to_combine:
        if field == 'diversity_score':
            cleaned_fields.append(str(sample_data.get(field, 0.0)))
        else:
            cleaned_fields.append(clean_and_lemmatize(str(sample_data.get(field, ""))))

    combined_text = " ".join(cleaned_fields)

    # Tokenize
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Move to appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probs, dim=1)[0].item()
        confidence = probs[0][predicted_label].item()

    # Generate reasoning
    detailed_reasoning = get_detailed_reasoning(sample_data, predicted_label, confidence, diversity_reasons)

    return predicted_label, confidence, detailed_reasoning

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        # Get JSON data from the request
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Check if data is a single student or multiple students
        if isinstance(data, dict):
            # Single student case
            students = [data]
        elif isinstance(data, list):
            # Multiple students case
            students = data
        else:
            return jsonify({'error': 'Invalid input format. Expected a JSON object or array of objects'}), 400

        selected_students = []
        non_selected_students = []
        
        # Process each student
        for student_data in students:
            # Ensure all required fields are present
            required_fields = ['town', 'identify_as', 'why_internship', 'tech_experience', 'goals']
            for field in required_fields:
                if field not in student_data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            # Predict for the student
            predicted_label, confidence, reasoning = predict_for_student(student_data)
            
            # Create result object
            result = {
                'student_data': student_data,
                'selected': bool(predicted_label),
                'confidence': confidence,
                'reasoning': reasoning
            }
            
            # Add to appropriate list
            if bool(predicted_label):
                selected_students.append(result)
            else:
                non_selected_students.append(result)

        # Sort both lists by confidence score in descending order
        sorted_selected = sorted(selected_students, key=lambda x: x['confidence'], reverse=True)
        sorted_non_selected = sorted(non_selected_students, key=lambda x: x['confidence'], reverse=True)
        
        # Combine lists with selected students first
        all_ranked_students = sorted_selected + sorted_non_selected
        
        # Add ranking to all students
        for i, result in enumerate(all_ranked_students, 1):
            result['rank'] = i

        # Return the ranked results
        return jsonify({
            'total_applications': len(all_ranked_students),
            'total_selected': len(sorted_selected),
            'ranked_results': all_ranked_students
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

