from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS 
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image 
import pickle

# --- Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
MODEL_PATH     = 'model_checkpoints/final_caption_model.keras'
TOKENIZER_PATH = 'pickled_features/tokenizer.pkl'
PARAMS_PATH    = 'pickled_features/model_params.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# Universal CORS support for targeted endpoints handling multi-origin development contexts
CORS(app, resources={
    r"/predict": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"], "methods": ["POST", "OPTIONS"]},
    r"/": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"], "methods": ["GET", "OPTIONS"]}
})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# --- Global Engine Scope Bindings ---
captioning_model = None
feature_extractor_model = None
tokenizer = None
indices_to_words = None
max_len_caption = 0
VOCAB_SIZE = 0
END_TOKEN_STRING_FROM_PARAMS = "end" 

def load_resources():
    global captioning_model, feature_extractor_model, tokenizer, indices_to_words, max_len_caption, VOCAB_SIZE, END_TOKEN_STRING_FROM_PARAMS
    try:
        print("Loading Keras model...")
        captioning_model = load_model(MODEL_PATH)
        print("Model loaded.")

        print("Loading ResNet50 feature extractor...")
        feature_extractor_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
        print("Feature extractor loaded.")

        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded.")

        print(f"Loading model parameters from {PARAMS_PATH}...")
        with open(PARAMS_PATH, 'rb') as f:
            params = pickle.load(f)
        
        max_len_caption = params['max_length']
        VOCAB_SIZE = params['vocab_size']
        
        # Safe structural dictionary recovery handling file format modifications
        indices_to_words = params.get('indices_to_words') or params.get('idx_to_word')
        END_TOKEN_STRING_FROM_PARAMS = params.get('end_token_string', '<end>') 

        print(f"Resources loaded: max_length={max_len_caption}, vocab_size={VOCAB_SIZE}, END_TOKEN='{END_TOKEN_STRING_FROM_PARAMS}'")
        if not indices_to_words or not max_len_caption or not END_TOKEN_STRING_FROM_PARAMS:
            raise ValueError("Essential parameters missing or corrupted within index target file dictionary.")

    except Exception as e:
        print(f"An absolute failure occurred during target runtime asset allocation: {e}")
        exit()

# --- Feature Engineering Pipeline ---
def preprocess_image_and_extract_features(image_path):
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) 
        features = feature_extractor_model.predict(x, verbose=0)
        return features.squeeze()
    except Exception as e:
        print(f"Error processing image feature translation sequence {image_path}: {e}")
        return None

# --- Decoding Strategy 1: Greedy Search ---
def generate_caption_greedy(photo_features_encoded):
    global captioning_model, tokenizer, max_len_caption, indices_to_words, END_TOKEN_STRING_FROM_PARAMS
    
    photo_features_encoded = photo_features_encoded.reshape(1, -1)
    in_text = '<start>' 
    generated_words = [] 

    for _ in range(max_len_caption):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        if not sequence: 
            break
        sequence_padded = pad_sequences([sequence], maxlen=max_len_caption, padding='post')
        
        # Optimize internal loops using direct functional tensor matrix evaluation
        y_pred_probs = captioning_model([photo_features_encoded, sequence_padded], training=False).numpy()
        y_pred_idx = np.argmax(y_pred_probs[0])
        
        word = indices_to_words.get(y_pred_idx)

        if word is None or word in ['<pad>', '<unk>']: 
            break
        if word == END_TOKEN_STRING_FROM_PARAMS: 
            break 
        
        generated_words.append(word)
        in_text += ' ' + word
        
    return " ".join(generated_words)

# --- Decoding Strategy 2: Multi-Branch Beam Search ---
def generate_caption_beam(photo_features_encoded, beam_k=3):
    global captioning_model, tokenizer, max_len_caption, indices_to_words, END_TOKEN_STRING_FROM_PARAMS
    
    photo = photo_features_encoded.reshape(1, -1)
    beams = [('<start>', 0.0)]
    completed = []

    for _ in range(max_len_caption):
        candidates = []
        for text, log_prob in beams:
            if text.split()[-1] == END_TOKEN_STRING_FROM_PARAMS:
                completed.append((text, log_prob))
                continue

            sequence = tokenizer.texts_to_sequences([text])[0]
            sequence_padded = pad_sequences([sequence], maxlen=max_len_caption, padding='post')
            
            # CPU-Accelerated tensor logic bypasses tracking graph evaluations
            probs = captioning_model([photo, sequence_padded], training=False).numpy()[0]
            
            top_k_idx = np.argsort(probs)[::-1][:beam_k]
            for idx in top_k_idx:
                word = indices_to_words.get(idx)
                if word is None or word in ['<pad>', '<unk>']:
                    continue
                new_log = log_prob + np.log(probs[idx] + 1e-9)
                candidates.append((text + " " + word, new_log))

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_k]

    completed.extend(beams)
    if not completed:
        return ""

    best_sequence = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
    words = best_sequence.split()
    
    # Filter processing tags cleanly before returning string output
    cleaned_words = [w for w in words if w not in ['<start>', END_TOKEN_STRING_FROM_PARAMS]]
    return " ".join(cleaned_words)

# --- Routing Operations ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['file']
        strategy = request.form.get('strategy', 'greedy') # Catch frontend context parameter
        
        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_features = preprocess_image_and_extract_features(filepath)
            if image_features is None:
                return jsonify({'error': 'Feature extraction matrix pipeline failure.'}), 500

            # Route calculation using selected strategy pattern
            if strategy == 'beam':
                app.logger.info("Routing process towards Beam Search decoding matrix layer.")
                caption = generate_caption_beam(image_features, beam_k=3)
            else:
                app.logger.info("Routing process towards sequential Greedy token selection layer.")
                caption = generate_caption_greedy(image_features)

            if not caption:
                return jsonify({'error': 'Text transformation layer generated empty token arrays.'}), 500

            image_url = url_for('static', filename=f'uploads/{filename}', _external=True)
            return jsonify({
                'caption': caption,
                'image_url': image_url
            })

    except Exception as e:
        app.logger.error(f"Critical execution block fault: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal pipeline error encountered during execution.'}), 500

if __name__ == '__main__':
    load_resources() 
    app.run(debug=True, host='0.0.0.0', port=5000)