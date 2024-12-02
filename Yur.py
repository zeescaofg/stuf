import cv2
import numpy as np
import pytesseract
import pyautogui
from PIL import Image
import time
import threading

# Set the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path if necessary

# List to store previously typed texts
previous_texts = []

# Debugging: Function to display matches on the screen
def draw_matches(image, locations, template_size):
    for loc in zip(*locations[::-1]):
        top_left = loc
        bottom_right = (top_left[0] + template_size[1], top_left[1] + template_size[0])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Matches", image)
    cv2.waitKey(1)

# Preprocessing to enhance visibility
def preprocess_image(image):
    if len(image.shape) == 3:  # Check if the image has 3 channels (e.g., RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image  # If already grayscale, use it directly

    enhanced = cv2.equalizeHist(gray)  # Improve contrast
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)  # Reduce noise
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresholded


# Function to capture, extract text, and simulate typing it on the keyboard
def capture_and_extract_text_and_type(template_path):
    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template = preprocess_image(template)

    # Capture a screenshot of the entire screen
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_processed = preprocess_image(screenshot_np)

    # Perform the template matching
    best_match = None
    best_loc = None
    best_val = 0
    for scale in np.linspace(0.5, 1.5, 10):  # Test scales from 50% to 150%
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if resized_template.shape[0] > screenshot_processed.shape[0] or resized_template.shape[1] > screenshot_processed.shape[1]:
            break

        result = cv2.matchTemplate(screenshot_processed, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_match = resized_template
            best_loc = max_loc

    if best_val > 0.5:  # Adjust threshold for confidence
        matched_top_left = best_loc
        matched_bottom_right = (
            matched_top_left[0] + best_match.shape[1],
            matched_top_left[1] + best_match.shape[0],
        )

        # Crop the region where the image was found
        cropped_region = screenshot_np[
            matched_top_left[1]:matched_bottom_right[1],
            matched_top_left[0]:matched_bottom_right[0],
        ]

        # Zoom in on the cropped region for better OCR accuracy
        zoom_factor = 2
        zoomed_in = cv2.resize(cropped_region, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        # Convert to a format suitable for OCR
        zoomed_in_pil = Image.fromarray(zoomed_in)
        extracted_text = pytesseract.image_to_string(zoomed_in_pil)

        # Clean up the extracted text
        extracted_text = extracted_text.strip()

        # Check if the text is already typed
        if extracted_text not in previous_texts and extracted_text:
            print("Extracted Text: ", extracted_text)

            # Type out the extracted text letter by letter
            for char in extracted_text:
                pyautogui.write(char)
                time.sleep(0.05)  # Delay between typing each character
            pyautogui.press("enter")  # Press Enter after typing

            # Add the text to the previously typed list
            previous_texts.append(extracted_text)

    # Introduce a small delay between checks (e.g., 0.5 seconds)
    time.sleep(10)


# Function to type `/gen roblox` every 63 seconds
def type_command():
    while True:
        time.sleep(61)
        for char in "/gen epicgames":
            pyautogui.write(char)
            time.sleep(0.05)  # Delay between typing each character
        pyautogui.press("enter")

# Start the command typing in a separate thread
command_thread = threading.Thread(target=type_command, daemon=True)
command_thread.start()


# Path to the template image you want to find
template_path = r'C:\Users\salmi\OneDrive\Desktop\yus.png'  # Replace with your image path

# Continuously search for the image and type the text when a new match is found
while True:
    capture_and_extract_text_and_type(template_path)
