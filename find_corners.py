import cv2
import numpy as np

# Захват видео с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Отзеркаливание изображения
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

            # Отображение прямоугольника
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            # Отображение углов
            for (x, y) in rect:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Здесь можно добавить код для перспективной трансформации

    # Отображение результата
    cv2.imshow("Frame", frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
