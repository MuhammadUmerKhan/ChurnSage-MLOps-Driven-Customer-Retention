# load_spacy_model.py
import spacy

# Try to load the language model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    # If the model is not found, download it
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")  # Load the model after downloading

# Now you can use the nlp object for your NLP tasks