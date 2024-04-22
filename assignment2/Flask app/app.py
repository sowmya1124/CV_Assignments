from flask import Flask, render_template, request
import imageio
import cv2 as cv
import numpy as np
import math
import os
from matplotlib import pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hardcoded camera matrix path
CAMERA_MATRIX_PATH = 'images/right/camera_matrix.txt'

def extract_features_sift_and_stitch_images(frame1, frame2):
    gray_1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray_2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    algortihm = cv.SIFT_create()

    kp_img1, desc_img1 = algortihm.detectAndCompute(gray_1, None)
    kp_img2, desc_img2 = algortihm.detectAndCompute(gray_2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc_img2, desc_img1, k=2)

    features = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            features.append(m)

    query_pts = np.float32([kp_img2[m.queryIdx]
                           .pt for m in features]).reshape(-1, 1, 2)
    train_pts = np.float32([kp_img1[m.trainIdx]
                           .pt for m in features]).reshape(-1, 1, 2)

    matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
    dst = cv.warpPerspective(frame2, matrix, ((frame1.shape[1] + frame2.shape[1]), frame2.shape[0]))
    dst[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
    
    return dst

def convert_milli_to_cm(x):
    x = x / 10
    return x / 25.4

def get_circle_diameter(image_path):
    # Load camera matrix
    camera_matrix = []
    with open(CAMERA_MATRIX_PATH, 'r') as f:
        for line in f:
            camera_matrix.append([float(num) for num in line.split()])

    # Load image
    image = cv.imread(image_path)

    # Define points (you may need to adjust these values)
    x, y, w, h = 15, 14, 18, 1
    Image_point1x = x
    Image_point1y = y
    Image_point2x = x + w
    Image_point2y = y + h

    # Draw rectangle and line on the image
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv.line(image, (Image_point1x, Image_point1y), (Image_point1x, Image_point2y), (0, 0, 255), 8)

    # Calculate real world points
    Z = 320
    FX = camera_matrix[0][0]
    FY = camera_matrix[1][1]
    Real_point1x = Z * (Image_point1x / FX)
    Real_point1y = Z * (Image_point1y / FY)
    Real_point2x = Z * (Image_point2x / FX)
    Real_point2y = Z * (Image_point2y / FY)

    # Calculate diameter in pixels
    dist = math.sqrt((Real_point2y - Real_point1y) * 2 + (Real_point2x - Real_point1x) * 2)

    # Draw dimensions result on the image
    font = cv.FONT_HERSHEY_SIMPLEX
    text = f'Diameter: {dist:.2f} cm'
    cv.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv.LINE_AA)

    # Save the image with drawn dimensions
    result_image_path = os.path.join(UPLOAD_FOLDER, 'result_image.jpg')
    cv.imwrite(result_image_path, image)

    return dist

def compute_integral_image(image_path):
    frame = cv.imread(image_path)
    gray_clr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = gray_clr.shape

    integral_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            integral_image[i][j] = int(gray_clr[i][j])

    for i in range(1, width):
        integral_image[0][i] += integral_image[0][i - 1]

    for j in range(1, height):
        integral_image[j][0] += integral_image[j - 1][0]

    for i in range(1, height):
        for j in range(1, width):
            integral_image[i][j] = integral_image[i - 1][j] + integral_image[i][j - 1] - integral_image[i - 1][j - 1] + gray_clr[i][j]

    return integral_image

def save_image_rgb(image_path, image_array):
    imageio.imwrite(image_path, image_array.astype(np.uint8))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            diameter_cm = get_circle_diameter(file_path)
            result_img = "static/uploads/result_image.jpg"
            message = f"Diameter of circle: { diameter_cm } cm"
            return render_template('result.html', message=message, result_img=result_img)
    return render_template('index.html')


@app.route('/image_integral', methods=['GET', 'POST'])
def image_integral():
    if request.method == 'POST':
        if 'img_file' not in request.files:
            return 'No file part'
        file = request.files['img_file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            integral_image = compute_integral_image(file_path)
            np.savetxt(os.path.join(app.config['UPLOAD_FOLDER'], 'integral_matrix.txt'), integral_image, fmt='%d')
            result_img = os.path.join(app.config['UPLOAD_FOLDER'], "integral_image.jpg")
            save_image_rgb(result_img, integral_image)
            message = f"Image Integral for given image is:"
            return render_template('result.html', message=message, result_img=result_img)
    return render_template('index.html')

@app.route('/stitch_images', methods=['GET', 'POST'])
def stitch_images():
    if 'files' not in request.files:
        return 'No file part'

    files = request.files.getlist('files')

    uploaded_files = []
    for file in files:
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)

    images = []
    for img_path in uploaded_files:
        images.append(cv.imread(img_path))
    
    pan_result = images[0]
    for img_ind in range(1, len(uploaded_files)):
        stitched_img = extract_features_sift_and_stitch_images(pan_result, images[img_ind])
        if stitched_img is not None:
            pan_result = stitched_img
            
    result_img = "static/stitched_image.jpg"
    cv.imwrite(result_img, pan_result)
    message = "Stitched Image"
    return render_template('result.html', message=message, result_img=result_img)

if __name__ == '__main__':
    app.run(debug=True)

