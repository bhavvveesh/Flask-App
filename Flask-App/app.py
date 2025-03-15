import os
import torch
import nltk
from flask import Flask, render_template, request, send_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.tokenize import sent_tokenize
from werkzeug.utils import secure_filename

nltk.download('punkt')

app = Flask(__name__, static_folder="static")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Summarization Model (BART-based)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load Translation Model (English to Kannada)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_text(text):
    tokenizer.src_lang = "eng_Latn"
    sentences = sent_tokenize(text)
    translated_sentences = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("kan_Knda")

        with torch.no_grad():
            translated = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)

        kannada_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        translated_sentences.append(kannada_text)

    return " ".join(translated_sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            english_text = file.read()

        translated_text = translate_text(english_text)
        summarized_text = summarizer(english_text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
        summarized_translated_text = translate_text(summarized_text)

        translated_output_file = os.path.join("static", "kannada_translation.txt")
        summarized_output_file = os.path.join("static", "kannada_summarized_translation.txt")

        with open(translated_output_file, "w", encoding="utf-8") as file:
            file.write(translated_text)

        with open(summarized_output_file, "w", encoding="utf-8") as file:
            file.write(summarized_translated_text)

        return render_template('result.html',
                               translated_file="kannada_translation.txt",
                               summarized_file="kannada_summarized_translation.txt")
    return "No file uploaded", 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


@app.route("/")
def home():
    return "Hello, Render!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no PORT is set
    app.run(host="0.0.0.0", port=port)
