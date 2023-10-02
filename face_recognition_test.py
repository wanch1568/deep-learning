import cv2

# 顔認識のためのカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# カメラのキャプチャ開始
cap = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔認識
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 顔が検出された部分を矩形で囲む
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # フレームを表示
    cv2.imshow('Face Recognition', frame)

    # 'q' キーを押すとループから抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
