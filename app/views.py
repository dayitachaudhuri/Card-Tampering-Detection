# Important imports
from app import app
from flask import request, render_template
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import numpy as np

# Adding path to config
GENERATED_FILE_PATH = 'app/static/generated'
NEW_FILE_PATH = 'app/static/generated/uploads'
ORIGINAL_FILE_PATH = 'app/static/generated/original'
if not os.path.exists(GENERATED_FILE_PATH):
    os.makedirs(GENERATED_FILE_PATH)
if not os.path.exists(NEW_FILE_PATH):
    os.makedirs(NEW_FILE_PATH)
if not os.path.exists(ORIGINAL_FILE_PATH):
    os.makedirs(ORIGINAL_FILE_PATH)

# Route to get diff
@app.route("/image-diff-finder", methods=["GET", "POST"])
def imageDiffFinder():
    
    if request.method == "GET":
        props = {
            "show_result" : False
        }
        return render_template('image_diff_finder.html', **props)
    
    elif request.method == "POST":
        try:
            # Get uploaded images
            new_file_upload = request.files['new_file_upload']
            original_file_upload = request.files['original_file_upload']
                    
            # Resize and save the uploaded images
            check_image = Image.open(new_file_upload).convert("RGB").resize((250,160))
            check_image.save(os.path.join(NEW_FILE_PATH, 'image.jpg'))

            original_image = Image.open(original_file_upload).convert("RGB").resize((250,160))
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
        except Exception as e:
            print(f"Error in Backend: {e}")
            
''' 
Route to Watermark Image.

Path - /image-watermark
Methods - GET, POST
'''
@app.route("/image-watermark", methods=["GET", "POST"])
def imageWatermark():
    
    if request.method == "GET":
        props = {
            "show_result" : False
        }
        return render_template('image_watermark.html', **props)
    
    elif request.method == "POST":
        try:
            
            # Get Watermarking Type from Form
            watermark_type = request.form['watermark_type']
            
            # Open Cover Image.
            cover_image_ref = request.files['cover_image']
            if not cover_image_ref:
                return "Cover image not uploaded", 400
            cover_image = Image.open(cover_image_ref)
            
            # Convert Cover Image to numpy array of required type.
            cover_image = np.array(cover_image.convert('RGB'))
            cover_image = cv2.cvtColor(cover_image, cv2.COLOR_RGB2BGR)
            
            # CASE 1 - If User Requests for Image Watermark.
            if watermark_type == "image":
                
                # Open Watermarking Image.
                logo_image_ref = request.files['watermark_image']
                if not logo_image_ref:
                    return "Watermark image not uploaded", 400
                logo_image = Image.open(logo_image_ref)
                
                # Convert Watermarking Image to numpy array of required type.
                logo_image = np.array(logo_image.convert('RGB'))
                logo_image = cv2.cvtColor(logo_image, cv2.COLOR_RGB2BGR)

                # Find  dimensions of cover image and watermarking image and determine the center of the cover image.
                h_image, w_image, _ = cover_image.shape
                h_logo, w_logo, _ = logo_image.shape 
                
                center_y = int(h_image / 2)
                top_y = center_y - int(h_logo / 2)
                bottom_y = top_y + h_logo

                center_x = int(w_image / 2)
                left_x = center_x - int(w_logo / 2)
                right_x = left_x + w_logo   

                # Determine ROI (Region Of Interest) in the cover image where the watermarking image shall be embedded. 
                # ROI is chosen such that the centers of the Cover Image and Watermarking Image align.
                roi = cover_image[top_y: bottom_y, left_x: right_x]
                
                # Add the watermarking image to the cover image.
                result = cv2.addWeighted(roi, 1, logo_image, 1, 0)
                cover_image[top_y: bottom_y, left_x: right_x] = result
            
            # CASE 2 - If User Requests for Text Watermark.
            elif watermark_type == "Text":
                
                # Get Watermark Text.
                watermark_text = request.form["watermark_text"]
                if not watermark_text:
                    return "Watermark text not uploaded", 400
                
                # Get dimensions of Cover Image
                h_image, w_image, _ = cover_image.shape
                
                # Embed the watermark text in the cover image
                cv2.putText(cover_image, watermark_text, org=(w_image - len(watermark_text) * 10, h_image - 10), 
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

            # CASE 3 - If Invalid Watermark Type is received.
            else:
                return "Watermark type not supported", 400
                
            # Save result image.
            cv2.imwrite(os.path.join(GENERATED_FILE_PATH, 'watermarked_image.jpg'), cover_image)
            props = {
                "show_result" : True,
                "watermarked_image": "../static/generated/watermarked_image.jpg",
            }
            return render_template('image_watermark.html', **props)
        except Exception as e:
            print(f"Error in Backend: {e}")
       
# Main function
if __name__ == '__main__':
    app.run(debug=True)
