import os
import cv2
from flask import Flask, render_template, flash, request, redirect, Response , url_for, send_from_directory
import numpy as np 
from werkzeug.utils import secure_filename
import face_recognition as fr
from datetime import datetime
from face import Presensi, ShowRiwayat
import encodeFace  
import os

app = Flask(__name__)

# Basic Routing
@app.route('/')  
def home():  
    return render_template('index.html')  

@app.route('/<string:page_name>')
def html_page(page_name):
    return render_template(page_name)

@app.route('/riwayat.html')
def riwayat():
    return ShowRiwayat()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# 1 Uploaad Image
UPLOAD_FOLDER = './static/assets/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect('upload_sukses.html')
    return render_template('upload.html')   

# 2 Encoding    
@app.route('/encode.html')
def encode():
    return render_template('encode.html')

@app.route('/encodeFunction', methods = ['POST', 'GET'])
# def findEncodings():
#     encodeFace.findEncodings(gambar)
    
# Absen via Camera
@app.route('/absen.html', methods = ['POST', 'GET'])
def absen():
    """Video streaming home page."""
    return render_template('absen.html')

path = './static/assets/uploads'
gambar = []
listNama = []               
myList = os.listdir(path) 
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}') # membaca semua gambar di direktori
    gambar.append(curImg) # menambahkan semua gambar ke list gambar
    listNama.append(os.path.splitext(cls)[0]) # menambahkan nama file ke dalam list listNama

encodeKnownFace = encodeFace.findEncodings(gambar)
camera = cv2.VideoCapture(0)
def gen_frame():
    while True:
        success, frame = camera.read()
        # Mengecilkan ukuran gambar
        frameSmall = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        frameSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB) # Convert ke Grescale

        faceLocWeb = fr.face_locations(frameSmall) # Menemukan lokasi wajah dalam list berisi 4 koordinat
        encodeWebcam = fr.face_encodings(frameSmall, faceLocWeb) # Encode gambar yg ditangkap dr webcam
        # Membandingkan wajah 
        for encodeFace, faceLoc in zip(encodeWebcam, faceLocWeb):
            matches = fr.compare_faces(encodeKnownFace, encodeFace) # membandingkan webcam dgn daftar wajah yg sudah dikenali
            faceDist = fr.face_distance(encodeKnownFace, encodeFace)
            # print(faceDist)   
            matchesIndex = np.argmin(faceDist)

            if matches[matchesIndex]:
                nama = listNama[matchesIndex].upper()
                # print(nama)
                # Membuat kotak hijau dgn nama dibawahnya
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, nama, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                Presensi(nama)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
def isi_absen():
    return Presensi()

if __name__ == '__main__':
    app.run(debug=True)