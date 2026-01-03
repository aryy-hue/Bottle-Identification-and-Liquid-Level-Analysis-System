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
    """Ensure upload folder exists and is clean"""
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_image(image_path):
    """Process image to detect bottle and liquid level"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Cannot read image file"
    
    # Simpan gambar asli untuk ditampilkan
    original_filename = os.path.basename(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove small noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = {
        'detected': False,
        'bottle_type': None,
        'level': None,
        'percentage': None,
        'contour_count': len(contours),
        'original_image': original_filename,
        'processed_image': None
    }
    
    if contours:
        # Sort contours by area to find the largest bottle
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Process only the largest contour
        contour = contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        
        # Identify bottle type based on aspect ratio (width/height)
        aspect_ratio = float(w) / h
        if 0.2 < aspect_ratio < 0.5:
            botol_type = "Medicine Bottle"
        elif aspect_ratio < 0.8:
            botol_type = "Blood Transfusion Bottle"
        else:
            botol_type = "Other Medicine Bottles"
        
        # Calculate the center position of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Display the bottle type at the top edge of the image
        top_text_y = max(y - 10, 20)  # Ensure text doesn't go above image
        cv2.putText(image, botol_type, (x, top_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Determine bottle height to calculate fluid level
        bottle_height = h
        
        # Segment the region to detect the liquid area
        liquid_region = gray[y:y+h, x:x+w]
        liquid_height = 0
        
        for i in range(liquid_region.shape[0]):
            row_mean = np.mean(liquid_region[i, :])
            if row_mean < 150:  # Threshold for liquid detection
                liquid_height = h - i
                break
        
        # Calculate the percentage of the liquid volume
        liquid_percentage = (liquid_height / bottle_height) * 100 if bottle_height > 0 else 0
        level = "Full" if liquid_percentage > 80 else "Half" if liquid_percentage > 30 else "Empty"
        
        # Draw liquid level line
        cv2.line(image, (x, y + h - liquid_height), (x + w, y + h - liquid_height), (255, 0, 0), 3)
        
        # Display the fluid level information
        level_text = f"{level} ({liquid_percentage:.2f}%)"
        level_text_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        level_text_x = center_x - (level_text_size[0] // 2)
        level_text_y = top_text_y + 30
        
        cv2.putText(image, level_text, (level_text_x, level_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw a bounding box around the contour
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Save processed image
        processed_filename = f"processed_{original_filename}"
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_image_path, image)
        
        # Update results
        results['detected'] = True
        results['bottle_type'] = botol_type
        results['level'] = level
        results['percentage'] = round(liquid_percentage, 2)
        results['processed_image'] = processed_filename
        results['bounding_box'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
    
    return results, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Create clean upload folder
            ensure_upload_folder()
            
            # Generate unique filename to avoid collisions
            original_filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{unique_id}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
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
    """Clear uploaded files and results"""
    ensure_upload_folder()
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)