import face_recognition

# 既知の顔の画像を読み込み
known_image = face_recognition.load_image_file("WIN_20230815_16_01_43_Pro.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# 新しい顔の画像を読み込み
unknown_image = face_recognition.load_image_file("new_face.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# 既知の顔のエンコーディングと新しい顔のエンコーディングを比較
results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)

if results[0]:
    print("This is the known person!")
else:
    print("This is NOT the known person!")
