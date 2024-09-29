from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify
import cv2
import pandas as pd
import time

app = Flask(__name__)

# Load tflite model
model = YOLO('best_float32.tflite')

# Load class list
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Global variable
# count = 0

detected_name = "Unknown"  # Default name
status = "Not Recognized"   # Default status

def generate_frames():
    # global count  # Declare global variable to modify it inside the function
    global detected_name, status  # Use global variables
    cap = cv2.VideoCapture(0)  # Capture from camera
    prev_time = 0  # To calculate FPS

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # count += 1
            # if count % 3 != 0:
            #     continue

            # Get current time to calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Run YOLO model on frame
            results = model(frame, imgsz=240)
            if len(results) > 0:
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")

                # Process each detected object
                for index, row in px.iterrows():
                    x1, y1, x2, y2 = map(int, row[:4])  # Coordinates of bounding box
                    confidence = float(row[4])  # Confidence score
                    d = int(row[5])  # Class ID
                    c = class_list[d]  # Class name

                     # Update detected name and status
                    detected_name = c  # Update detected name
                    status = "Hadir"    # Update status to "Hadir"

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
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(name=detected_name, status=status)  # Send detected name and status as JSON

if __name__ == "__main__":
    app.run(debug=True)
