# Импортируем необходимые библиотеки
import pyrealsense2 as rs
import numpy as np
import cv2

# Конфигурация глубины камеры
pipeline = rs.pipeline ()
config = rs.config ()

# Получаем информацию о подключенной камере и проверяем наличие RGB-камеры
pipeline_wrapper = rs.pipeline_wrapper (pipeline)
pipeline_profile = config.resolve (pipeline_wrapper)
device = pipeline_profile.get_device ()
device_product_line = str (device.get_info (rs.camera_info.product_line))
found_rgb = False

for s in device.sensors:
    if s.get_info (rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print ("Для демонстрации требуется камера глубины с датчиком цвета.")
    exit (0)

# Настройка параметров потоков глубины и цвета в зависимости от модели камеры
if device_product_line == 'D455':
    depth_res = (640, 480)
    color_res = (960, 540)
else:
    depth_res = (640, 480)
    color_res = (640, 480)

config.enable_stream (rs.stream.depth, *depth_res, rs.format.z16, 30)
config.enable_stream (rs.stream.color, *color_res, rs.format.bgr8, 30)

# Начинаем потоковую передачу данных с камеры
pipeline.start (config)

try:
    while True:
        # Получаем кадры глубины и цвета
        frames = pipeline.wait_for_frames ()
        depth_frame = frames.get_depth_frame ()
        color_frame = frames.get_color_frame ()

        if not depth_frame or not color_frame:
            continue

        # Преобразуем изображения в массивы numpy
        depth_image = np.asanyarray (depth_frame.get_data ())
        color_image = np.asanyarray (color_frame.get_data ())

        # Применяем цветовую карту к изображению глубины и объединяем изображения глубины и цвета в одно
        depth_colormap = cv2.applyColorMap (cv2.convertScaleAbs (depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize (color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                              interpolation=cv2.INTER_AREA)
            images = np.hstack ((resized_color_image, depth_colormap))
        else:
            images = np.hstack ((color_image, depth_colormap))

        # Отображаем полученное изображение в окне с названием "Depht Camera"
        cv2.namedWindow ('Depht Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow ('Depht Camera', images)

        # Ожидаем нажатия клавиши "q" или "esc" для закрытия окна с изображением
        key = cv2.waitKey (1)

        if key & 0xFF == ord ('q') or key == 27:
            cv2.destroyAllWindows ()
            break

finally:
    # Останавливаем потоковую передачу данных и закрываем все окна
    pipeline.stop ()