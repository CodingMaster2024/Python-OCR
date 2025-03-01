import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


model = YOLO("license_plate_detector.pt")


image_path = "working2.png"
image = cv2.imread(image_path)


results = model(image)


plate_detected = False

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        license_plate = image[y1:y2, x1:x2]  
        cv2.imwrite("cropped_plate.jpg", license_plate)
        plate_detected = True


if plate_detected:
    plate_text = pytesseract.image_to_string(Image.open("cropped_plate.jpg"), config="--psm 8").strip()
    
    if plate_text:
        print(f"🔍 Detected License Plate Number: {plate_text}")
    else:
        print("⚠️ OCR failed to detect text on the plate.")
else:
    print("🚫 No license plate detected in the image.")
