import spacy

# Try to load the language model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Model not found. Downloading the model...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")  # Load the model after downloading