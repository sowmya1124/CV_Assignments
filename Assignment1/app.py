from flask import Flask, render_template, request, flash, redirect, url_for
import cv2 as cv
import math
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Set a secret key for flashing messages

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CAMERA_MATRIX_PATH = 'images/right/camera_matrix.txt'

def get_circle_diameter(image_path):
    try:
        camera_matrix = []
        with open(CAMERA_MATRIX_PATH, 'r') as f:
            for line in f:
                camera_matrix.append([float(num) for num in line.split()])

        image = cv.imread(image_path)
        x, y, w, h = 15, 14, 18, 1
        Image_point1x, Image_point1y = x, y
        Image_point2x, Image_point2y = x + w, y + h

        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv.line(image, (Image_point1x, Image_point1y), (Image_point1x, Image_point2y), (0, 0, 255), 8)

        Z = 320
        FX, FY = camera_matrix[0][0], camera_matrix[1][1]
        Real_point1x, Real_point1y = Z * (Image_point1x / FX), Z * (Image_point1y / FY)
        Real_point2x, Real_point2y = Z * (Image_point2x / FX), Z * (Image_point2y / FY)

        dist = math.sqrt((Real_point2y - Real_point1y) ** 2 + (Real_point2x - Real_point1x) ** 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        text = f'Diameter: {dist:.2f} cm'
        print(text)
        cv.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv.LINE_AA)

        result_image_path = os.path.join(UPLOAD_FOLDER, 'result_image.jpg')
        cv.imwrite(result_image_path, image)

        return dist
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            diameter_cm = get_circle_diameter(file_path)
            if diameter_cm is not None:
                return render_template('result.html', diameter=diameter_cm)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
