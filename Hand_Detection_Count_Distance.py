import cv2
import mediapipe as mp
import math
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing=mp.solutions.drawing_utils
# Video capture from webcam (you can change the parameter to the index of your camera if you have multiple cameras)
cap = cv2.VideoCapture(0)
fing=[8,12,16,20]
dic={}
while cap.isOpened():
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)



    if results[1].multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            for idd, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                dic[idd]=[x,y]

        count = []
        for i in fing:
           count = count
           if dic[i][1] < dic[i - 2][1]:
               count.append(1)
        if dic[4][0] >dic[3][0]:
              count.append(1)
       # Calculate the distance between two specific fingers (index finger and thumb)
        w = math.sqrt(pow(dic[9][0] - dic[0][0], 2) + pow(dic[9][1] - dic[0][1], 2))
        W = 6.3  # Width of the hand
        f = 900  # Focal length
        d = int((W * f) / w)  # Calculate distance in centimeters



        cv2.putText(frame,f"Count {count.count(1) } Distance{d}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        print(count)
    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
