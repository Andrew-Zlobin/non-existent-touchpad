import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

def get_rectangle_corners(frame):
    frame = cv2.flip(frame, 1)

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применение гауссового размытия
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Бинаризация изображения
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Аппроксимация контура
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Проверка на прямоугольник
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            # Упорядочивание углов
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Верхний левый
            rect[2] = pts[np.argmax(s)]  # Нижний правый

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Верхний правый
            rect[3] = pts[np.argmax(diff)]  # Нижний левый
    return rect

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Коэффициенты сглаживания и чувствительности
smooth_x, smooth_y = 0, 0
smooth_factor = 0.2
dragging = False

#src_pts = np.float32([
#    [100, 100],   # верхний левый
#    [500, 100],   # верхний правый
#    [500, 400],   # нижний правый
#    [100, 400],   # нижний левый
#])
success, frame = cap.read()
src_pts = get_rectangle_corners(frame)

# ====== Целевая "прямая" система координат (например, 640x480) ======
dst_width, dst_height = 640, 480
dst_pts = np.float32([
    [0, 0],
    [dst_width, 0],
    [dst_width, dst_height],
    [0, dst_height],
])

# Матрица гомографии
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Нарисовать белый прямоугольник
    cv2.polylines(img, [np.int32(src_pts)], isClosed=True, color=(255, 0, 0), thickness=2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = handLms.landmark
            h, w, _ = img.shape

            # Получаем координаты ладони (например, индекс 9)
            raw_x = int(lm_list[9].x * w)
            raw_y = int((1 - lm_list[9].y) * h)

            # ПРЕОБРАЗУЕМ КООРДИНАТЫ ЧЕРЕЗ гомографию
            point = np.array([[[raw_x, raw_y]]], dtype='float32')
            transformed_point = cv2.perspectiveTransform(point, matrix)[0][0]
            tx, ty = transformed_point

            # Обрезаем по целевой области
            tx = np.clip(tx, 0, dst_width)
            ty = np.clip(ty, 0, dst_height)

            # Переводим в экранные координаты
            screen_x = screen_width * (tx / dst_width)
            screen_y = screen_height * (ty / dst_height)

            # Сглаживание
            smooth_x += (screen_x - smooth_x) * smooth_factor
            smooth_y += (screen_y - smooth_y) * smooth_factor

            pyautogui.moveTo(smooth_x, smooth_y)

            # Расстояние между большим и указательным пальцем
            x1, y1 = int(lm_list[4].x * w), int(lm_list[4].y * h)
            x2, y2 = int(lm_list[8].x * w), int(lm_list[8].y * h)
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 40:
                if not dragging:
                    dragging = True
                    pyautogui.mouseDown()
            else:
                if dragging:
                    dragging = False
                    pyautogui.mouseUp()

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking with ROI", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
