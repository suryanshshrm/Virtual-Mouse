import cv2
import mediapipe as mp
import pyautogui
import math

#Screen size
screen_w,screen_h = pyautogui.size()

#Mediapipe Setup
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

#Webcam
cap = cv2.VideoCapture(0)

click_threshold = 30
right_click_threshold = 30

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)

    h,w,_= frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hand.HAND_CONNECTIONS
            )
            
            landmarks = hand_landmarks.landmark
            
            #Index finger tip
            index_tip = landmarks[8]

            x = int(index_tip.x*w)
            y = int(index_tip.y*h)

            cv2.circle(frame,(x,y),10,(0,255,0),-1)

            #Move mouse
            mouse_x = screen_w*index_tip.x
            mouse_y = screen_h*index_tip.y                                                     

            pyautogui.moveTo(mouse_x,mouse_y)

            #THumb tip
            thumb_tip = landmarks[4]

            tx = int(thumb_tip.x*w)
            ty = int(thumb_tip.y*h)

            #Distance for left click
            distance = math.hypot(x-tx,y-ty)

            if distance < click_threshold:
                pyautogui.click()
                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),
                    2
                )

            #MIddle finger tip
            middle_tip = landmarks[12]

            mx = int(middle_tip.x*w)  
            my = int(middle_tip.y*h)

            #Distance for right click
             
            right_distance = math.hypot(mx - tx, my - ty)

            if right_distance < right_click_threshold:
                pyautogui.rightClick()
                cv2.putText(
                    frame,
                    "RIGHT CLICK",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
