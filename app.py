import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import shutil
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Cannot read image file"
    
    original_filename = os.path.basename(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Canny Edge
    edges = cv2.Canny(blurred, 40, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = {
        'detected': False,
        'bottle_type': None,
        'level': None,
        'percentage': None,
        'contour_count': len(contours),
        'original_image': original_filename,
        'processed_image': None
    }
    
    image_area = image.shape[0] * image.shape[1]
    valid_contours = [c for c in contours if cv2.contourArea(c) > (image_area * 0.05)]
    
    if valid_contours:
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        contour = valid_contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = float(w) / h
        if 0.2 < aspect_ratio < 0.5:
            botol_type = "Medicine Bottle"
        elif aspect_ratio < 0.8:
            botol_type = "Blood Transfusion Bottle"
        else:
            botol_type = "Other Medicine Bottles"
            
        center_x = x + w // 2
        top_text_y = max(y - 10, 20)
        cv2.putText(image, botol_type, (x, top_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        roi_gray = blurred[y:y+h, x:x+w]
        
        _, liquid_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        liquid_height = 0
        start_scan = int(h * 0.1)
        end_scan = int(h * 0.95)
        
        min_liquid_width = int(w * 0.4) 
        
        for i in range(start_scan, end_scan):
            white_pixels = np.sum(liquid_thresh[i, :] == 255)
            if white_pixels > min_liquid_width:
                liquid_height = h - i
                break
                
        # Calculate percentage
        liquid_percentage = (liquid_height / h) * 100 if h > 0 else 0
        liquid_percentage = min(max(liquid_percentage, 0), 100) 
        
        level = "Full" if liquid_percentage > 80 else "Half" if liquid_percentage > 30 else "Empty"
        
        # Draw liquid level line
        cv2.line(image, (x, y + h - liquid_height), (x + w, y + h - liquid_height), (255, 0, 0), 3)
        
        # Display the fluid level information
        level_text = f"{level} ({liquid_percentage:.2f}%)"
        level_text_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        level_text_x = center_x - (level_text_size[0] // 2)
        level_text_y = top_text_y + 30
        
        cv2.putText(image, level_text, (level_text_x, level_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Save processed image
        processed_filename = f"processed_{original_filename}"
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_image_path, image)
        
        results['detected'] = True
        results['bottle_type'] = botol_type
        results['level'] = level
        results['percentage'] = round(liquid_percentage, 2)
        results['processed_image'] = processed_filename
        
    return results, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            ensure_upload_folder()
            
            # Generate unique filename to avoid collisions
            original_filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{unique_id}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            results, error = process_image(filepath)
            
            if error:
                flash(error)
                return redirect(request.url)
            
            return render_template('index.html', results=results)
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, BMP, TIFF).')
            return redirect(request.url)
    
    return render_template('index.html', results=None)

@app.route('/clear', methods=['POST'])
def clear_results():
    ensure_upload_folder()
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)