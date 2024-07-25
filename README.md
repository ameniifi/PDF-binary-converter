# PDF Classifier with AI

## Overview
This project leverages artificial intelligence to classify and process PDFs by analyzing their content. The solution extracts images and text from PDFs, classifies them using pre-trained models, encodes the results into binary format, and provides the binary data for download.

## Features
- **Image Classification**: Classifies images within PDFs using a Convolutional Neural Network (CNN).
- **Text Classification**: Classifies text extracted from PDFs using a feed-forward neural network.
- **Binary Encoding**: Converts classified content into binary data.
- **File Download**: Users can download a text file containing the binary data.

## Tech Stack
- **Python**: Programming language used for development.
- **Flask**: Web framework for creating the user interface and handling HTTP requests.
- **TensorFlow/Keras**: Libraries for building and training machine learning models.
- **PyMuPDF (fitz)**: Library for extracting images and text from PDFs.
- **PIL (Pillow)**: Python Imaging Library for image processing.

## Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ameniifi/pdf-classifier-ai.git
    ```

2. **Navigate to the Project Directory**:
    ```bash
    cd pdf-classifier-ai
    ```

3. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run the Flask Application**:
    ```bash
    python app.py
    ```

2. **Open Your Browser** and go to `http://127.0.0.1:5000`.

3. **Upload a PDF**: Use the web interface to upload a PDF file.

4. **Download the Binary Data**: Once the PDF is processed, download the resulting text file containing the binary data.

## Code Explanation
- **app.py**: Main Flask application file that handles user interface and file processing.
- **models.py**: Contains code to build, train, and use machine learning models for classification.
- **utils.py**: Helper functions for processing PDFs, encoding data, and saving results.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- TensorFlow/Keras for machine learning tools.
- Flask for the web framework.
- PyMuPDF for PDF processing.
- Pillow for image manipulation.
