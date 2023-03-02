import cv2
import numpy as np
import open3d as o3d

# инициализируем камеру с индексом 0
cap = cv2.VideoCapture(0)

# получаем размеры кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# создаем массивы для хранения координат точек
points = []

# создаем цикл, в котором будем считывать кадры
while True:
    # считываем один кадр
    ret, frame = cap.read()

    # проверяем, не пустой ли он
    if not ret:
        break

    # преобразуем изображение в черно-белое
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # находим границы объектов на кадре
    edges = cv2.Canny(gray, 0.5, 0.5)

    # находим контуры
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # сохраняем координаты точек
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            points.append([cx, cy, frame[cy][cx][2]])

    # рисуем контуры
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # отображаем кадр
    cv2.imshow("Frame", frame)

    # ждем 25ms нажатия клавиши
    key = cv2.waitKey(20)

    # если нажат ESC, завершаем цикл
    if key == 27:
        break

# преобразуем точки в массив numpy
points = np.array(points)

# Поворачиваем изображение на 180 градусов
frame = cv2.rotate(frame, cv2.ROTATE_180)

# Создаем точки для сканирования
points_3d = np.array(points)

# Создаем объект PointCloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_3d[:, :3])

# Отображаем точки
o3d.visualization.draw_geometries([point_cloud])

# освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

