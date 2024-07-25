from flask import Flask, request, send_from_directory, render_template_string
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import fitz  # PyMuPDF
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024  # 16 MB

# Ensure the upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load your pre-trained models
image_model = load_model('image_model.h5')
text_model = load_model('text_model.h5')

# Function to encode text data to fixed-size vector
def encode_text(text, max_length=500):
    text = text[:max_length].ljust(max_length)
    encoded = [ord(char) for char in text]
    return np.array(encoded)

# Function to encode image data to binary
def encode_image_to_binary(data):
    data = (data * 255).astype(np.uint8)  # Convert to uint8
    binary_data = ''.join(format(byte, '08b') for byte in data.flatten())
    return binary_data

# Function to encode text data to binary
def encode_text_to_binary(data):
    binary_data = ''.join(format(byte, '08b') for byte in data)
    return binary_data

# Function to extract images and text from a PDF
def extract_from_pdf(file_path):
    pdf_document = fitz.open(file_path)
    images = []
    texts = []

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Extract text
        text = page.get_text()
        if text:
            texts.append(text)

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')  # Ensure image is in RGB format
            images.append(image)

    return images, texts

# Function to classify and encode content from a PDF
def classify_and_encode_pdf(file_path):
    images, texts = extract_from_pdf(file_path)
    binary_data = ""

    for image in images:
        image = image.resize((128, 128))
        data = img_to_array(image) / 255.0
        data = np.expand_dims(data, axis=0)
        label = np.argmax(image_model.predict(data), axis=1)[0]
        binary_data += encode_image_to_binary(data)

    for text in texts:
        encoded_data = encode_text(text)
        data = np.expand_dims(encoded_data, axis=0)
        label = np.argmax(text_model.predict(data), axis=1)[0]
        binary_data += encode_text_to_binary(encoded_data)

    return binary_data

# Function to save binary data to a file
def save_binary_data_to_file(file_path, binary_data):
    filename = os.path.basename(file_path)
    binary_file_name = f"{filename}_encoded.txt"
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], binary_file_name)

    with open(output_file_path, 'w') as f:
        f.write(binary_data)

    return output_file_path

# Flask routes
@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <title>PDF Classifier</title>
        <style>
          body {
            background-color: #f8f9fa;
          }
          .container {
            margin-top: 50px;
            max-width: 600px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          }
          .btn-primary {
            background-color: #007bff;
            border-color: #007bff;
          }
          .btn-primary:hover {
            background-color: #0056b3;
            border-color: #0056b3;
          }
          h1 {
            text-align: center;
            margin-bottom: 20px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>PDF Classifier</h1>
          <form action="/upload" method="post" enctype="multipart/form-data" class="mt-3">
            <div class="form-group">
              <label for="file">Upload PDF</label>
              <input type="file" class="form-control-file" id="file" name="file">
            </div>
            <button type="submit" class="btn btn-primary btn-block">Upload</button>
          </form>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
      </body>
    </html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        binary_data = classify_and_encode_pdf(file_path)
        output_file_path = save_binary_data_to_file(file_path, binary_data)
        return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
            <title>PDF Classifier</title>
            <style>
              body {
                background-color: #f8f9fa;
              }
              .container {
                margin-top: 50px;
                max-width: 600px;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
              }
              h1 {
                text-align: center;
                margin-bottom: 20px;
              }
              .btn-primary {
                background-color: #007bff;
                border-color: #007bff;
              }
              .btn-primary:hover {
                background-color: #0056b3;
                border-color: #0056b3;
              }
              .btn-link {
                color: #007bff;
              }
              .btn-link:hover {
                color: #0056b3;
              }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>File Processed</h1>
              <p class="mt-3">Your file has been processed. <a href="/download/{{ filename }}" class="btn btn-link">Download the binary data file</a>.</p>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
          </body>
        </html>
        ''', filename=os.path.basename(output_file_path))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
