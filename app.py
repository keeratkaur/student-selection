# app.py

import os
import json
import re
import nltk
import pandas as pd
import torch
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz, process
import uvicorn

# ========== NLTK Setup ==========
# Download all necessary NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK data download completed.")

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ========== Device & Model Initialization ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading tokenizer and base BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Attempt to load custom weights if provided
model_path = os.environ.get("MODEL_PATH", "internship_selection_model.pt")
if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded custom weights from {model_path}")
    except Exception as e:
        print(f"Could not load custom weights: {e}. Using base model.")

model.to(device)
model.eval()

# ========== Helper Functions ==========

def analyze_regional_priority(town: str):
    """Fuzzy-match to Avalon vs non-Avalon regions."""
    if not town or not isinstance(town, str):
        return 0.0, "No location provided"

    avalon_locations = {
        "st. john's", "mount pearl", "paradise", "conception bay south",
        "torbay", "portugal cove-st. philip's", "flatrock", "pouch cove",
        "bauline", "witless bay", "bay bulls", "holyrood", "seal cove",
        "fox trap", "long pond", "kelligrews", "upper gullies", "foxtrap",
        "topsail", "chamberlains", "logy bay", "middle cove", "outer cove",
        "petty harbour", "maddox cove", "goulds", "kilbride", "bell island",
        "wabana", "placentia", "st. bride's", "dunville", "freshwater"
    }
    town_lower = town.lower()
    best = process.extractOne(town_lower, avalon_locations,
                              scorer=fuzz.token_set_ratio, score_cutoff=80)
    if best:
        return 0.5, f"Standard priority – matched Avalon location '{best[0]}'"
    else:
        return 1.0, f"High priority – non-Avalon region '{town}'"

def clean_and_lemmatize(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(t) for t in filtered]
    return " ".join(lemmatized)

def calculate_diversity_score(data: dict):
    """Compute a diversity score (0–1) and explanatory reasons."""
    score = 0.0
    reasons = []

    # Define identity priorities (higher score = higher priority)
    identity_priorities = {
        # Gender identities
        "female": 0.5,
        "non-binary": 0.6,
        "transgender": 0.7,
        "genderqueer": 0.6,
        "genderfluid": 0.6,
        "agender": 0.6,
        "two-spirit": 0.7,
        "prefer not to say": 0.3,
        "male": 0.1,
        
        # Racial/ethnic identities
        "indigenous": 0.8,
        "black": 0.7,
        "asian": 0.6,
        "hispanic": 0.6,
        "latino": 0.6,
        "latina": 0.6,
        "middle eastern": 0.6,
        "pacific islander": 0.7,
        "mixed race": 0.5,
        
        # Other diversity factors
        "disabled": 0.7,
        "veteran": 0.5,
        "first generation": 0.6,
        "low income": 0.6,
        "lgbtq+": 0.6
    }

    # Helper function to clean and normalize identity strings
    def clean_identity(identity: str) -> str:
        # Remove extra spaces, convert to lowercase, and handle common variations
        identity = identity.strip().lower()
        # Handle common variations and abbreviations
        variations = {
            "trans": "transgender",
            "nb": "non-binary",
            "enby": "non-binary",
            "queer": "lgbtq+",
            "poc": "person of color",
            "bipoc": "black, indigenous, person of color",
            "first gen": "first generation",
            "1st gen": "first generation",
            "low income": "low income",
            "low-income": "low income"
        }
        return variations.get(identity, identity)

    # Process the identify_as field
    identify_as = data.get("identify_as", "").strip()
    if not identify_as:
        reasons.append("No identity information provided")
        return 0.0, reasons

    # Split by commas and clean each identity
    identities = [clean_identity(id) for id in identify_as.split(",")]
    # Remove empty strings and duplicates
    identities = list(set(filter(None, identities)))
    
    unknown_identities = []
    processed_identities = set()
    
    for identity in identities:
        # Skip if we've already processed this identity
        if identity in processed_identities:
            continue
        processed_identities.add(identity)
        
        # Handle compound identities (e.g., "black, indigenous, person of color")
        if "," in identity:
            compound_identities = [clean_identity(id) for id in identity.split(",")]
            for comp_id in compound_identities:
                if comp_id in identity_priorities:
                    score += identity_priorities[comp_id]
                    reasons.append(f"Identity diversity +{identity_priorities[comp_id]:.1f} ({comp_id})")
                else:
                    unknown_identities.append(comp_id)
                    default_score = 0.4
                    score += default_score
                    reasons.append(f"Identity diversity +{default_score:.1f} (unrecognized identity: {comp_id})")
        else:
            if identity in identity_priorities:
                score += identity_priorities[identity]
                reasons.append(f"Identity diversity +{identity_priorities[identity]:.1f} ({identity})")
            else:
                # Try to find the closest match using fuzzy matching
                best_match = process.extractOne(identity, identity_priorities.keys(), 
                                              scorer=fuzz.token_set_ratio, score_cutoff=80)
                if best_match:
                    score += identity_priorities[best_match[0]]
                    reasons.append(f"Identity diversity +{identity_priorities[best_match[0]]:.1f} ({identity} → matched to {best_match[0]})")
                else:
                    unknown_identities.append(identity)
                    default_score = 0.4
                    score += default_score
                    reasons.append(f"Identity diversity +{default_score:.1f} (unrecognized identity: {identity})")

    # Add a note about unknown identities if any were found
    if unknown_identities:
        reasons.append(f"Note: The following identities were not recognized but still considered: {', '.join(unknown_identities)}")

    loc_score, loc_reason = analyze_regional_priority(data.get("town", ""))
    score += loc_score
    reasons.append(f"Location diversity +{loc_score:.1f} ({loc_reason})")

    return min(score, 1.0), reasons

def get_detailed_reasoning(data: dict, label: int, conf: float, div_reasons: list[str]) -> str:
    """Build the textual reasoning explaining the model decision."""
    def extract_terms(field: str, min_len=4):
        tokens = word_tokenize(field.lower())
        return sorted({w for w in tokens
                       if w.isalpha()
                       and w not in stop_words
                       and len(w) >= min_len})

    tech_terms = extract_terms(data.get("tech_experience", ""))
    nontech_terms = extract_terms(data.get("non_tech_experience", ""))
    motive_terms = extract_terms(data.get("why_internship", ""))
    goal_terms = extract_terms(data.get("goals", ""))

    conf_pct = conf * 100
    status = "SELECTED" if label == 1 else "NOT SELECTED"
    confidence_desc = (
        "very confident" if conf >= 0.80
        else "confident" if conf >= 0.70
        else "moderately confident"
    )

    lines = [
        f"Selection Decision: {status} ({confidence_desc}, {conf_pct:.1f}%)",
        "-" * 60,
        "Technical Analysis:",
    ]
    if tech_terms:
        lines.append(f" • Key terms: {', '.join(tech_terms)}")
    else:
        lines.append(" • Limited technical detail provided.")

    lines += [
        "",
        "Motivation Analysis:",
    ]
    if motive_terms:
        lines.append(f" • Motivation terms: {', '.join(motive_terms)}")
    if goal_terms:
        lines.append(f" • Goal terms: {', '.join(goal_terms)}")

    lines += [
        "",
        "Diversity Considerations:",
    ]
    for r in div_reasons:
        lines.append(f" • {r}")

    return "\n".join(lines)

def predict_for_student(data: dict) -> dict:
    """Run the full pipeline: diversity, text cleaning, BERT inference, reasoning."""
    div_score, div_reasons = calculate_diversity_score(data)
    data["diversity_score"] = div_score

    # Prepare combined text
    fields = [
        "town", "diversity_score", "why_internship",
        "tech_experience", "non_tech_experience",
        "goals", "other_comments"
    ]
    pieces = []
    for f in fields:
        val = data.get(f, "")
        pieces.append(str(val) if f == "diversity_score" else clean_and_lemmatize(str(val)))
    combined = " ".join(pieces)

    # Tokenize & predict
    inputs = tokenizer(
        combined,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].tolist()
        label = int(torch.argmax(torch.tensor(probs)))
        conf = probs[label]

    reasoning = get_detailed_reasoning(data, label, conf, div_reasons)
    return {
        "selected": bool(label),
        "confidence": conf,
        "reasoning": reasoning,
        "student_data": data
    }

def process_applications(json_payload: str) -> str:
    """Handle one-or-many JSON applications, rank them, and produce text output."""
    apps = json.loads(json_payload)
    if not isinstance(apps, list):
        apps = [apps]

    results = [predict_for_student(app) for app in apps]
    # Separate selected vs non-selected
    selected = [r for r in results if r["selected"]]
    non_selected = [r for r in results if not r["selected"]]
    selected.sort(key=lambda x: x["confidence"], reverse=True)
    non_selected.sort(key=lambda x: x["confidence"], reverse=True)

    ranked = selected + non_selected
    
    # Create a more visually appealing output
    output = []
    
    # Header
    output.append("=" * 80)
    output.append("RANKED APPLICATIONS".center(80))
    output.append("=" * 80)
    output.append(f"Total Applications: {len(ranked)}")
    output.append(f"Selected Candidates: {len(selected)}")
    output.append("=" * 80)
    output.append("")

    for idx, r in enumerate(ranked, start=1):
        # Candidate header
        output.append(f"RANK #{idx}".center(80, "-"))
        output.append("")
        
        # Basic info
        status = "SELECTED" if r['selected'] else "NOT SELECTED"
        confidence = r['confidence'] * 100
        output.append(f"Status: {status} ({confidence:.1f}%)")
        output.append(f"Location: {r['student_data'].get('town', 'N/A')}")
        output.append(f"Gender: {r['student_data'].get('identify_as', 'N/A')}")
        output.append("")
        
        # Detailed reasoning
        output.append("DETAILED ANALYSIS".center(80, "-"))
        output.append("")
        
        # Split the reasoning into sections
        reasoning_lines = r["reasoning"].split("\n")
        current_section = ""
        
        for line in reasoning_lines:
            if line.startswith("Selection Decision:"):
                output.append("DECISION".center(80, "-"))
                output.append(line)
                output.append("-" * 80)
            elif line.startswith("Technical Analysis:"):
                current_section = "TECHNICAL ANALYSIS"
                output.append(current_section.center(80, "-"))
            elif line.startswith("Motivation Analysis:"):
                current_section = "MOTIVATION ANALYSIS"
                output.append(current_section.center(80, "-"))
            elif line.startswith("Diversity Considerations:"):
                current_section = "DIVERSITY CONSIDERATIONS"
                output.append(current_section.center(80, "-"))
            elif line.strip():
                if line.startswith(" • "):
                    output.append(line)
                else:
                    output.append(line)
            else:
                output.append("")
        
        # Add spacing between candidates
        output.append("=" * 80)
        output.append("")

    return "\n".join(output)

# ========== FastAPI + Gradio Setup ==========

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input validation
class Application(BaseModel):
    town: str
    identify_as: str
    why_internship: str
    tech_experience: str
    non_tech_experience: str
    goals: str
    other_comments: str

@app.post("/api/process")
async def api_process(applications: list[Application]):
    # Convert Pydantic models to dicts
    data = [a.dict() for a in applications]
    output = process_applications(json.dumps(data))
    return {"output": output}

# Gradio interface
def gradio_fn(text: str) -> str:
    return process_applications(text)

gr_interface = gr.Interface(
    fn=gradio_fn,
    inputs=gr.Textbox(
        label="Applications JSON",
        lines=10,
        placeholder='[{"town":"St. John\'s", "identify_as":"Female", ...}]'
    ),
    outputs=gr.Textbox(label="Ranked Results", lines=30),
    title="Internship Applications Ranker",
    description="Paste one or more JSON applications to get a ranked analysis."
)

# Mount Gradio at root ("/")
app = gr.mount_gradio_app(app, gr_interface, path="/")

# Entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
