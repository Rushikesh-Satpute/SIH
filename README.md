# Comprehensive Automated Document Verification System

### Problem Statement Number: **SIH1652**

This project aims to automate the verification of semi-structured and unstructured documents using advanced techniques in **OCR**, **Preprocessing**, **Natural Language Processing (NLP)**, **Dataset Management**, **AI Model Design**, and an **Interactive Web UI**. The goal is to streamline the document verification process efficiently and accurately.

---

## Team Members and Responsibilities

- **Shruti**: Frontend UI Development.
- **Shweta**: Preprocessing.
- **Rushikesh**: Text extraction using OCR (Tesseract) in Python.
- **Shriniwas**: NLP-related tasks implemented in Jupyter Notebook.
- **Riya**: Dataset Collection and Management.
- **Gaurav**: AI Model Design and Development.

---

## Project Structure

```
├── backend/
│   ├── ocr_module.py         # Rushikesh's OCR (It also include Preprocessing)
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
```

#### OCR and Preprocessing Module (Rushikesh's Work)

##### Prerequisites:
- Python 3.x
- Tesseract OCR (ensure it is installed and configured correctly)

##### Setup Steps:

1. **Clone the Repository**:

    ```
    git clone https://github.com/SIH1652/SIH1652.git
    cd SIH1652/backend
    ```

2. **Install Dependencies**:  
   Ensure `tesseract` is installed and add its path to the environment variables.

    ```
    pip install -r requirements.txt
    ```
3. **Install TeserracrOCR exe file**:
    Install following in Path which is specified in ocr_module.py file
    Default path: C:\Program Files\Tesseract-OCR\tesseract.exe 
    [Download](https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe)
    
3. **Run the OCR Module**:  
   Run the following command to preprocess and extract text from an image or PDF.

    ```
    python ocr_module.py
    ```

