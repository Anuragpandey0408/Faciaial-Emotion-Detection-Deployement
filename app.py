from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import cv2


app = Flask(__name__)


model = load_model("emotion_model.keras")


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            try:
                
                file_path = os.path.join("static/uploads", file.filename)
                file.save(file_path)

                
                frame = cv2.imread(file_path)  
                if frame is None:
                    error = "Error: Could not open or find the image."
                    return render_template("index.html", result=None, error=error)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                faces = face_cascade.detectMultiScale(gray, 1.3, 3)  

                for (x, y, w, h) in faces:
                    sub_face_img = gray[y:y+h, x:x+w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = model.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]

                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 255, 255), -1)  
                    cv2.putText(frame, emotion_labels[label], (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  

                
                output_image_path = os.path.join("static/uploads", "output_" + file.filename)
                cv2.imwrite(output_image_path, frame)

                return render_template("index.html", result=emotion_labels[label], image="output_" + file.filename)

            except UnidentifiedImageError:
                error = "Invalid image format. Please upload a valid image file."
            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template("index.html", result=None, error=error)


@app.route("/live")
def live_emotion_detection():
    return render_template("live.html")


def generate_frames():
    cap = cv2.VideoCapture(0)  
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        
        for (x, y, w, h) in faces:
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))  
            roi_gray = roi_gray / 255.0  
            roi_gray = np.expand_dims(roi_gray, axis=0)  
            roi_gray = np.expand_dims(roi_gray, axis=-1)  
            
            
            predictions = model.predict(roi_gray)
            emotion_index = np.argmax(predictions)  
            emotion_label = emotion_labels[emotion_index]
            
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  
            cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 255, 255), -1)  
            cv2.putText(frame, emotion_label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)  
        
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/contact')
def contact():
    return render_template('contact.html')  

if __name__ == "__main__":
    app.run(debug=True)