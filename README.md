# Comprehensive Automated Document Verification System

### Problem Statement Number: **SIH1652**

This project aims to automate the verification of semi-structured and unstructured documents using advanced techniques in **OCR**, **Preprocessing**, **Natural Language Processing (NLP)**, **Dataset Management**, **AI Model Design**, and an **Interactive Web UI**. The goal is to streamline the document verification process efficiently and accurately.

---

## Team Members and Responsibilities

- **Rushikesh Satpute**: Text extraction using OCR (Tesseract) in Python.
- **Shruti**: Frontend UI Development using HTML, CSS, and JavaScript.
- **Shriniwas**: NLP-related tasks implemented in Jupyter Notebook.
- **Shweta**: Data Preprocessing.
- **Riya**: Dataset Collection and Management.
- **Gaurav**: AI Model Design and Development.

---

## Project Structure

```bash
├── backend/
│   ├── ocr_module.py         # Rushikesh's OCR
│   ├── preprocessing.py      # Shweta's preprocessing code
│   ├── requirements.txt      # Python dependencies for backend and OCR
├── frontend/
│   ├── index.html            # Shruti's frontend UI code
│   ├── style.css             # Styling for UI
│   └── script.js             # JavaScript for interactivity
├── nlp/
│   ├── nlp_notebook.ipynb    # Shriniwas's NLP work in Jupyter Notebook
├── dataset/
│   ├── dataset.csv           # Riya's dataset files for training the model
│   ├── dataset_cleaning.py   # Data cleaning script
├── model/
│   ├── model_design.py       # Gaurav's AI model design
│   ├── model_training.py     # AI model training script
└── README.md                 # Project documentation (this file)


#### OCR and Preprocessing Module (Rushikesh's Work)

##### Prerequisites:
- Python 3.x
- Tesseract OCR (ensure it is installed and configured correctly)

##### Setup Steps:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/SIH1652/SIH1652.git
    cd SIH1652/backend
    ```

2. **Install Dependencies**:  
   Ensure `tesseract` is installed and add its path to the environment variables.

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the OCR Module**:  
   Run the following command to preprocess and extract text from an image or PDF.

    ```bash
    python ocr_module.py
    ```

