# Important imports
from app import app
from flask import request, render_template
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image

# Adding path to config
NEW_FILE_PATH = 'app/static/generated/uploads'
ORIGINAL_FILE_PATH = 'app/static/generated/original'
GENERATED_FILE_PATH = 'app/static/generated'

# Route to get diff
@app.route("/", methods=["GET", "POST"])
def index():
    
    if request.method == "GET":
        props = {
            "show_result" : False
        }
        return render_template('image_diff_finder.html', **props)
    
    elif request.method == "POST":
        
        # Get uploaded images
        new_file_upload = request.files['new_file_upload']
        original_file_upload = request.files['original_file_upload']
                
        # Resize and save the uploaded images
        check_image = Image.open(new_file_upload).resize((250,160))
        check_image.save(os.path.join(NEW_FILE_PATH, 'image.jpg'))

        original_image = Image.open(original_file_upload).resize((250,160))
        original_image.save(os.path.join(ORIGINAL_FILE_PATH, 'image.jpg'))

        # Read uploaded and original images as array
        original_image = cv2.imread(os.path.join(ORIGINAL_FILE_PATH, 'image.jpg'))
        uploaded_image = cv2.imread(os.path.join(NEW_FILE_PATH, 'image.jpg'))

        # Convert images into grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Calculate threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
                
        # Draw contours on image
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save all output images (if required)
        cv2.imwrite(os.path.join(GENERATED_FILE_PATH, 'image_original.jpg'), original_image)
        cv2.imwrite(os.path.join(GENERATED_FILE_PATH, 'image_new.jpg'), uploaded_image)
        cv2.imwrite(os.path.join(GENERATED_FILE_PATH, 'image_diff.jpg'), diff)
        cv2.imwrite(os.path.join(GENERATED_FILE_PATH, 'image_thresh.jpg'), thresh)
        
        props = {
            "show_result" : True,
            "diff": f"{str(round(score*100,2))}% Similarity",
            "original_image":  "../static/generated/image_original.jpg",
            "new_image":  "../static/generated/image_new.jpg",
            "thresh_image": "../static/generated/image_thresh.jpg",
            "diff_image": "../static/generated/image_diff.jpg"
        }
        return render_template('image_diff_finder.html', **props)
       
# Main function
if __name__ == '__main__':
    app.run(debug=True)
