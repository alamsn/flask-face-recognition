from flask import render_template
import os
import numpy as np
import face_recognition as fr
import csv
import cv2
import pandas as pd
from datetime import datetime
import mysql.connector

# Import Gambar
path = 'static/assets/uploads'
csvPath = 'static/assets/csv/daftarhadir.csv'
gambar = []
listNama = []
myList = os.listdir(path) 
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}') # membaca semua gambar di direktori
    gambar.append(curImg) # menambahkan semua gambar ke list gambar
    listNama.append(os.path.splitext(cls)[0]) # menambahkan nama file ke dalam list listNama

# Database
# 1 Koneksi DB
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="daftarhadir"
)

if db.is_connected():
  print("Berhasil terhubung ke database")

cursor = db.cursor()

# baca file csv
with open('static/assets/csv/encodeFace.csv', 'r') as f:
  file = csv.reader(f)
  encodeKnownFace = list(file)

def Presensi(nama):
    with open('static/assets/csv/daftarhadir.csv', 'r+') as f:
        DataList = f.readlines()
        listNama = []
        # print(DataList)
        for line in DataList:
            entry = line.split(',')
            listNama.append(entry[0])
        if nama not in listNama:
            now = datetime.now()
            dtString1 = now.strftime('%H:%M')
            dtString2 = now.strftime('%Y-%m-%d')
            f.writelines(f'\n{nama}, {dtString2}, {dtString1}')
    # Parsing CSV file 
    # CVS Column Names
    col_names = ['Nama','Hari','Jam']
    # Use Pandas to parse the CSV file
    csvData = pd.read_csv(csvPath,names=col_names, header=None)
    # Loop through the Rows
    for i,row in csvData.iterrows():
        sql = "INSERT INTO riwayat (Nama, Hari, Jam) VALUES (%s, %s, %s)"
        value = (row['Nama'],row['Hari'],row['Jam'])
        cursor.execute(sql, value)
        db.commit()
        print(i, row['Nama'],row['Hari'],row['Jam']) 

def ShowRiwayat():
    cursor = db.cursor()
    sql = "SELECT * FROM riwayat"
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    return render_template('riwayat.html', riwayat=result)
# Menggunakan Webcam untuk membandingkan wajah dgn listNama[]
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
    