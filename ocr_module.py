import cv2
from PIL import Image
import pytesseract
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pdf2image import convert_from_path
import os
import concurrent.futures
import time  # Import the time module

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Improved Image Preprocessing Function
def preprocess_image(image, file_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if file_size < 256 * 1024:
        # Light Preprocessing for Small Files
        return gray

    # Full Preprocessing for Larger Files
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized


# File Selector Function
def select_file():
    Tk().withdraw()  # Hide the root window
    filename = askopenfilename(title="Select an Image or PDF",
                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                                          ("PDF files", "*.pdf")])
    return filename


# Function to Process PDF Files in Parallel
def process_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    extracted_text = ""

    def process_page(image, page_num):
        open_cv_image = np.array(image)[:, :, ::-1].copy()  # Convert RGB to BGR
        file_size = os.path.getsize(pdf_path)

        # Preprocess the image
        start_time = time.time()  # Start timer for preprocessing
        processed_image = preprocess_image(open_cv_image, file_size)
        preprocessing_time = time.time() - start_time  # End timer for preprocessing

        cv2.imwrite(f'processed_page_{page_num + 1}.jpg', processed_image)

        img_for_ocr = Image.fromarray(processed_image)

        # Start timer for OCR
        ocr_start_time = time.time()
        custom_config = r'--oem 1 --psm 6'
        page_text = pytesseract.image_to_string(img_for_ocr, lang='hin+eng', config=custom_config)
        ocr_time = time.time() - ocr_start_time  # End timer for OCR

        return f"\n\n--- Page {page_num + 1} ---\n{page_text}\nPreprocessing Time: {preprocessing_time:.2f} seconds\nOCR Time: {ocr_time:.2f} seconds"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_page, images, range(len(images)))

    extracted_text = ''.join(results)
    return extracted_text


# Main Function
filename = select_file()

if filename:
    file_size = os.path.getsize(filename)  # Get the file size
    total_start_time = time.time()  # Start timer for total processing time

    if filename.endswith(".pdf"):
        # Process PDF
        extracted_text = process_pdf(filename)
    else:
        # Process Image
        img = cv2.imread(filename)

        # Ensure the image is loaded correctly
        if img is not None:
            # Preprocess the image
            start_time = time.time()  # Start timer for preprocessing
            processed_image = preprocess_image(img, file_size)
            preprocessing_time = time.time() - start_time  # End timer for preprocessing

            cv2.imwrite('processed_image.jpg', processed_image)

            img_for_ocr = Image.fromarray(processed_image)

            # Start timer for OCR
            ocr_start_time = time.time()
            custom_config = r'--oem 1 --psm 6'
            extracted_text = pytesseract.image_to_string(img_for_ocr, lang='hin+eng', config=custom_config)
            ocr_time = time.time() - ocr_start_time  # End timer for OCR

            # Total processing time
            total_time = time.time() - total_start_time  # Total time

            # Output the extracted text and processing times
            print("Extracted Text:\n", extracted_text)
            print(f"Preprocessing Time: {preprocessing_time:.2f} seconds")
            print(f"OCR Time: {ocr_time:.2f} seconds")
            print(f"Total Processing Time: {total_time:.2f} seconds")
        else:
            print(f"Error: Could not load image at {filename}")
else:
    print("No file selected.")
