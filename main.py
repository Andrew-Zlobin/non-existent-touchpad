import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

def is_finger_extended(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y

def is_thumb_folded(hand_landmarks, hand_label):
    if hand_label == 'Right':
        return hand_landmarks[4].x < hand_landmarks[3].x
    else:
        return hand_landmarks[4].x > hand_landmarks[3].x

def is_index_finger_extended(hand_landmarks, hand_label='Right'):
    index_extended = is_finger_extended(hand_landmarks, 8, 6)
    middle_folded = not is_finger_extended(hand_landmarks, 12, 10)
    ring_folded = not is_finger_extended(hand_landmarks, 16, 14)
    pinky_folded = not is_finger_extended(hand_landmarks, 20, 18)
    thumb_folded = is_thumb_folded(hand_landmarks, hand_label)
    
    return all([index_extended, middle_folded, ring_folded, pinky_folded, thumb_folded])

def is_index_and_middle_extended_only(hand_landmarks, hand_label="Right"):
    is_index_extended = is_finger_extended(hand_landmarks, tip_id=8, pip_id=6)
    is_middle_extended = is_finger_extended(hand_landmarks, tip_id=12, pip_id=10)
    is_ring_folded = not is_finger_extended(hand_landmarks, tip_id=16, pip_id=14)
    is_pinky_folded = not is_finger_extended(hand_landmarks, tip_id=20, pip_id=18)

    is_thumb_folded_flag = is_thumb_folded(hand_landmarks, hand_label)

    return (
            is_index_extended and
            is_middle_extended and
            is_ring_folded and
            is_pinky_folded and
            is_thumb_folded_flag
        )

def is_index_and_thumb_finger_extended(hand_landmarks, hand_label='Right'):
    index_extended = is_finger_extended(hand_landmarks, 8, 6)
    middle_folded = not is_finger_extended(hand_landmarks, 12, 10)
    ring_folded = not is_finger_extended(hand_landmarks, 16, 14)
    pinky_folded = not is_finger_extended(hand_landmarks, 20, 18)
    thumb_folded = not is_thumb_folded(hand_landmarks, hand_label)
    
    return all([index_extended, middle_folded, ring_folded, pinky_folded, thumb_folded])

def is_only_pinky_extended(hand_landmarks, hand_label):
    is_thumb_folded_flag = is_thumb_folded(hand_landmarks, hand_label)
    is_index_folded = not is_finger_extended(hand_landmarks, tip_id=8, pip_id=6)
    is_middle_folded = not is_finger_extended(hand_landmarks, tip_id=12, pip_id=10)
    is_ring_folded = not is_finger_extended(hand_landmarks, tip_id=16, pip_id=14)
    is_pinky_extended = is_finger_extended(hand_landmarks, tip_id=20, pip_id=18)

    return (
        is_pinky_extended and
        is_index_folded and
        is_middle_folded and
        is_ring_folded and
        is_thumb_folded_flag
    )

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

def is_ok_sign(hand_landmarks, hand_label='Right', threshold=0.05):
    
    # Индексы landmark'ов:
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
    MIDDLE_PIP, RING_PIP, PINKY_PIP = 10, 14, 18

    thumb_tip = hand_landmarks[THUMB_TIP]
    index_tip = hand_landmarks[INDEX_TIP]
    thumb_index_dist = distance(thumb_tip, index_tip)

    touching = thumb_index_dist < threshold

    index_extended = hand_landmarks[8].y < hand_landmarks[6].y

    if hand_label == 'Right':
        thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x
    else:
        thumb_extended = hand_landmarks[4].x < hand_landmarks[3].x

    middle_folded = hand_landmarks[MIDDLE_TIP].y > hand_landmarks[MIDDLE_PIP].y
    ring_folded = hand_landmarks[RING_TIP].y > hand_landmarks[RING_PIP].y
    pinky_folded = hand_landmarks[PINKY_TIP].y > hand_landmarks[PINKY_PIP].y

    return all([
        touching,
        index_extended,
        thumb_extended,
        middle_folded,
        ring_folded,
        pinky_folded
    ])

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils
    
    screen_width, screen_height = pyautogui.size()
    cap = cv2.VideoCapture(0)
    max_x, max_y = pyautogui.size()
    min_x, min_y = 1, 1
    print(max_x, max_y)

    last_movment_time = time.time()

    # mouse control
    pointer_pos_x, pointer_pos_y = max_x / 2, max_y / 2
    pointer_pos_delta_x, pointer_pos_delta_y = 0, 0
    pointer_pos_acceleration_x, pointer_pos_acceleration_y, = 0, 0
    pointer_pos_jerk_x, pointer_pos_jerk_y = 2, 2

    # scrolling
    scrolling = False
    scrolling_value = 2
    scrolling_hand_pos_x, scrolling_hand_pos_y = -1, -1

    pyautogui.moveTo(pointer_pos_x, pointer_pos_y)
    hand_pos_x, hand_pos_y = -1, -1
    

    dragging = False
    while True:
        res, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        results = hands.process(frame)
        if time.time() - last_movment_time < 0.1:
            continue
        last_movment_time = time.time()
        pointer_pos_x, pointer_pos_y = pyautogui.position()
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_list = handLms.landmark
                h, w, _ = frame.shape
                
                # Получаем координаты ладони (например, индекс 9)
                raw_x = int(lm_list[9].x * w)
                raw_y = int((1 - lm_list[9].y) * h)
                if hand_pos_x < 0 or hand_pos_y < 0:
                    hand_pos_x = raw_x
                    hand_pos_y = raw_y
                    scrolling_hand_pos_x, scrolling_hand_pos_y = raw_x, raw_y
                else:
                    pointer_pos_delta_x = hand_pos_x - raw_x
                    pointer_pos_delta_y = hand_pos_y - raw_y
                    hand_pos_x = raw_x
                    hand_pos_y = raw_y

                if is_index_and_thumb_finger_extended(lm_list) or dragging:
                    # print(hand_pos_x, hand_pos_y)
                    # print(raw_x, raw_y)
                    # print(pointer_pos_delta_x, pointer_pos_delta_y)
                    # print()
                    if pointer_pos_x + pointer_pos_delta_x > 0 and \
                            pointer_pos_x + pointer_pos_delta_x < screen_width and \
                            pointer_pos_y + pointer_pos_delta_y > 0 and \
                            pointer_pos_y + pointer_pos_delta_y < screen_height:
                        pyautogui.move(pointer_pos_delta_x, pointer_pos_delta_y)
                        
                    
                
                if is_index_finger_extended(lm_list):
                # if is_ok_sign(lm_list, threshold=0.2):
                    if not dragging:
                        dragging = True
                        pyautogui.mouseDown()
                        
                else:
                    if dragging:
                        dragging = False
                        pyautogui.mouseUp()
                
                if is_index_and_middle_extended_only(lm_list):
                    if not scrolling:
                        scrolling = True
                    direction = 1 if raw_y > scrolling_hand_pos_y else -1
                    hdirection = 1 if raw_x > scrolling_hand_pos_x else -1
                    pyautogui.scroll(direction * scrolling_value)
                    pyautogui.hscroll(hdirection * scrolling_value)
                else:
                    if scrolling:
                        scrolling = False

                scrolling_hand_pos_x, scrolling_hand_pos_y = raw_x, raw_y

                # Добавление текста
                cv2.putText(frame, str(is_ok_sign(lm_list, threshold=0.1)), (10, 30), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, color=(0, 255, 0), thickness=2)
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking with ROI", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        

