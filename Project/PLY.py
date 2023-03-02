import cv2
import numpy as np
import open3d as o3d

# Загружаем классификаторы для распознавания лиц и глаз
face_cascade = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier ('haarcascade_eye.xml')

# Запускаем камеру
cap = cv2.VideoCapture (0)

while True:
    # Считываем кадр с камеры
    ret, frame = cap.read ()

    # Преобразуем кадр в оттенки серого
    gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)

    # Распознаем лица на кадре
    faces = face_cascade.detectMultiScale (gray, 1.3, 5)

    # Для каждого лица находим глаза
    for (x, y, w, h) in faces:
        cv2.rectangle (frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale (roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle (roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Отображаем кадр с распознанными лицами и глазами
    cv2.imshow ('frame', frame)

    # Если нажата клавиша 'q', выходим из цикла
    if cv2.waitKey (1) & 0xFF == ord ('q'):
        break

# Останавливаем работу камеры и закрываем все окна
cap.release ()
cv2.destroyAllWindows ()

