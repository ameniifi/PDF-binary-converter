from flask import Flask, request, redirect, url_for, send_from_directory
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import fitz  # PyMuPDF
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
# Load models
try:
    image_model = tf.keras.models.load_model('image_model.h5')
    text_model = tf.keras.models.load_model('text_model.h5')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
def load_image(file_path, target_size=(128, 128)):
    try:
        img = load_img(file_path, target_size=target_size, color_mode='rgb')
        img = img_to_array(img)
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error loading image: {e}")

def encode_text(text, max_length=500):
    text = text[:max_length].ljust(max_length)
    encoded = [ord(char) for char in text]
    return np.array(encoded)

def encode_image_to_binary(data):
    data = (data * 255).astype(np.uint8)
    binary_data = ''.join(format(byte, '08b') for byte in data.flatten())
    return binary_data

def encode_text_to_binary(data):
    binary_data = ''.join(format(byte, '08b') for byte in data)
    return binary_data

def extract_from_pdf(file_path):
    pdf_document = fitz.open(file_path)
    images = []
    texts = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        text = page.get_text()
        if text:
            texts.append(text)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            images.append(image)
    return images, texts

def classify_and_encode_pdf(file_path):
    images, texts = extract_from_pdf(file_path)
    binary_data = ""
    for image in images:
        image = image.resize((128, 128))
        data = img_to_array(image) / 255.0
        data = np.expand_dims(data, axis=0)
        binary_data += encode_image_to_binary(data)
    for text in texts:
        encoded_data = encode_text(text)
        data = np.expand_dims(encoded_data, axis=0)
        binary_data += encode_text_to_binary(encoded_data)
    return binary_data

def save_binary_data_to_file(pdf_path, binary_data):
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    binary_file_name = pdf_base_name + '_encoded.txt'
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], binary_file_name)
    try:
        with open(output_file_path, 'w') as f:
            f.write(binary_data)
        return output_file_path
    except Exception as e:
        print(f"Error saving binary data to file: {e}")

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>PDF Classifier</title>
    <h1>Upload a PDF file</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="pdf_file">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    file = request.files['pdf_file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            binary_data = classify_and_encode_pdf(file_path)
            output_file_path = save_binary_data_to_file(file_path, binary_data)
            return redirect(url_for('download_encoded_file', filename=os.path.basename(output_file_path)))
        except Exception as e:
            print(f"Error processing file: {e}")
            return redirect(request.url)
        
@app.route('/download/<filename>')
def download_encoded_file(filename):
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        print(f"Error sending file: {e}")
        return redirect(url_for('index'))
    
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error running the app: {e}")