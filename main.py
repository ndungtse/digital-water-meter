import cv2
import pytesseract
from PIL import Image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # save the preprocessed image
    cv2.imwrite('preprocessed_image.jpg', thresholded_image)
    return thresholded_image

def extract_readings(image_path):
    preprocessed_image = preprocess_image(image_path)
    pil_image = Image.fromarray(preprocessed_image)

    # Use Tesseract OCR to extract text from the image but exclude non-numeric characters
    extracted_text = pytesseract.image_to_string(pil_image, config='--psm 6 outputbase digits')
    return extracted_text.strip()

if __name__ == "__main__":
    image_path = 'readings.jpeg'
    readings = extract_readings(image_path)

    print(f"Extracted Readings: {readings}")
