from ultralytics import YOLO
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, Response, jsonify
import cv2
import pandas as pd
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'  # Database SQLite
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load YOLO model
model = YOLO('model/w_masked.tflite', task='detect')

# Load class list
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Global variable
detected_name = "Unknown"  # Default name
status = "Not Recognized"  # Default status

# Model untuk tabel Attendance
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    tanggal = db.Column(db.String(20), nullable=False)  # Kolom tanggal

    def __repr__(self):
        return f"<Attendance {self.name} - {self.status} - {self.tanggal}>"

# Inisialisasi Database
with app.app_context():
    db.create_all()

# Fungsi untuk update kehadiran ke database
def update_attendance(detected_name, status):
    # Mendapatkan tanggal dan waktu sekarang
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format lengkap dengan jam
    current_date = datetime.now().strftime("%Y-%m-%d")  # Format hanya tahun-bulan-hari

    # Cek apakah orang tersebut sudah ada di database pada tanggal yang sama (hanya tahun-bulan-hari)
    attendance_today = Attendance.query.filter_by(name=detected_name).filter(Attendance.tanggal.like(f"{current_date}%")).first()

    if attendance_today:
        # Jika sudah ada entri untuk hari ini, tidak melakukan apa-apa
        print(f"Data untuk {detected_name} pada hari ini sudah ada.")
    else:
        # Jika tidak ada entri pada hari ini, tambahkan entri baru dengan tanggal dan waktu
        new_entry = Attendance(name=detected_name, status=status, tanggal=current_datetime)
        db.session.add(new_entry)
        db.session.commit()  # Simpan perubahan
        print(f"Entri baru untuk {detected_name} ditambahkan ke database.")

# Generate frames for video streaming
def generate_frames():
    global detected_name, status  # Use global variables
    cap = cv2.VideoCapture(0)  # Capture from camera

    if not cap.isOpened():
        print("Error: Camera is not accessible.")
        return

    prev_time = 0  # Initialize time for FPS calculation

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Run YOLO model on frame
            results = model(frame, imgsz=240)
            if results:  # Check if results is not empty
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")

                # Process each detected object
                for index, row in px.iterrows():
                    x1, y1, x2, y2 = map(int, row[:4])  # Coordinates of bounding box
                    confidence = float(row[4])  # Confidence score
                    d = int(row[5])  # Class ID
                    c = class_list[d]  # Class name

                    # Update detected name and status
                    detected_name = c  # Detected class name
                    status = "Hadir"  # Update status

                    # Update attendance to database within app context
                    with app.app_context():
                        update_attendance(detected_name, status)

                    # Draw bounding box (white color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    # Draw class label above bounding box (green color)
                    cv2.putText(frame, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw confidence score below bounding box (green color)
                    cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display FPS in the top left corner (green color)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Menampilkan halaman index dengan data kehadiran
    attendances = Attendance.query.all()  # Ambil semua data dari tabel Attendance
    return render_template('index.html', attendances=attendances)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_data')
def attendance_data():
    attendances = Attendance.query.all()
    attendance_list = [
        {"name": attendance.name, "status": attendance.status, "tanggal": attendance.tanggal}
        for attendance in attendances
    ]
    return jsonify(attendance_list)

if __name__ == "__main__":
    app.run(debug=True)
