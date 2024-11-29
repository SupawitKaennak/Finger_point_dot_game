import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize game parameters
cap = cv2.VideoCapture(0)
screen_width, screen_height = 640, 480
target_radius = 20
player_radius = 30
score = 0
start_time = time.time()
time_limit = 30  # 30 seconds countdown

# Randomly place the target
target_x = random.randint(target_radius, screen_width - target_radius)
target_y = random.randint(target_radius, screen_height - target_radius)

# Game loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate remaining time
    elapsed_time = time.time() - start_time
    remaining_time = max(0, time_limit - int(elapsed_time))

    # If time is up, end the game
    if remaining_time == 0:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw target
    cv2.circle(frame, (target_x, target_y), target_radius, (0, 0, 255), -1)

    # Draw player-controlled circle
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the landmark for the tip of the index finger (landmark #8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw the player's circle
            cv2.circle(frame, (cx, cy), player_radius, (255, 0, 0), -1)

            # Check collision with the target
            if np.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2) < target_radius + player_radius:
                score += 1
                target_x = random.randint(target_radius, screen_width - target_radius)
                target_y = random.randint(target_radius, screen_height - target_radius)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display score and remaining time
    cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {remaining_time}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Hand Tracking Game", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# End game message
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Game Over!")
print(f"Your final score is: {score}")
