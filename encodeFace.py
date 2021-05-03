import os 
import cv2
import numpy as np
import face_recognition as fr
from datetime import datetime
import csv

# Import Gambar
path = 'static/assets/uploads'
gambar = []
listNama = []
myList = os.listdir(path) 
# print(myList) # myList adalah list yg berisi nama file di folder gambarAbsen

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}') # membaca semua gambar di direktori
    gambar.append(curImg) # menambahkan semua gambar ke list gambar
    listNama.append(os.path.splitext(cls)[0]) # menambahkan nama file ke dalam list listNama
# print(listNama) # listNama adalah list yg berisi nama file tanpa ekstensi jpg

# Convert Gambar ke Greyscale & Encode Gambar
def findEncodings(gambar):
    encodeList = []
    for img in gambar:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeKnownFace = findEncodings(gambar)

with open('static/assets/csv/encodeFace.csv', 'w') as f: 
    # using csv.writer method from CSV package 
    write = csv.writer(f)  
    write.writerows(encodeKnownFace) 