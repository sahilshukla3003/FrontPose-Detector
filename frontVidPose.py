import cv2
import mediapipe as mp
import math
import time
import threading

# Function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if magnitude_v1 * magnitude_v2 == 0:
        return 0  # Avoid division by zero

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_in_radians = math.acos(cosine_angle)

    # Convert the angle from radians to degrees
    angle_in_degrees = math.degrees(angle_in_radians)
    return angle_in_degrees

# Function to check if the detected pose is a front pose
def is_front_pose(landmarks):
    # Check the visibility of specific landmarks
    required_landmarks = [11, 12, 23, 24, 3, 1, 6, 4, 7, 8, 13, 14, 25, 26, 31, 32]
    if not all(landmarks[i].visibility > 0.50 for i in required_landmarks):
        return False
    
    left_hand_landmarks = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
    if not all(landmarks[i].visibility > 0.2 for i in left_hand_landmarks):
        return False
    
    right_hand_landmarks = [4, 5, 14, 15, 19, 20, 21, 22, 29, 30]
    if not all(landmarks[i].visibility > 0.2 for i in right_hand_landmarks):
        return False

    # Inside the is_front_pose function
    left_eye_landmarks = [1,2,3]  # Update with correct indices
    right_eye_landmarks = [4,5,6]  # Update with correct indices

    # Ensure both eyes are visible and aligned at the same angle
    left_eye_visible = all(landmarks[i].visibility > 0.50 for i in left_eye_landmarks)
    right_eye_visible = all(landmarks[i].visibility > 0.50 for i in right_eye_landmarks)
    # eyes_aligned = abs(landmarks[5].x - landmarks[145].x) < 0.05  # Adjust the threshold as needed

    if not (left_eye_visible and right_eye_visible):
        return False



    # Check if both eyes are visible at the same angle and nose is in the middle and hands are down
    if (0.4 < landmarks[1].x < 0.6 and landmarks[7].x > landmarks[8].x and
            landmarks[11].y < landmarks[23].y and landmarks[12].y < landmarks[24].y):
        

        left_angle = [11,13,15]
        right_angle = [12,14,16]

        for side_landmarks in [left_angle, right_angle]:
            shoulder = (landmarks[side_landmarks[0]].x, landmarks[side_landmarks[0]].y)
            elbow = (landmarks[side_landmarks[1]].x, landmarks[side_landmarks[1]].y)
            wrist = (landmarks[side_landmarks[2]].x, landmarks[side_landmarks[2]].y)

            vector1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
            vector2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])

            angle = calculate_angle(vector1, vector2)
            print(angle)

            if angle <= 110:
                return False
            
        left_finger = [17,19,21]
        right_finger = [18,20,22]

        if not all(landmarks[i].visibility > 0.90 for i in left_finger):
            return False
        
        if not all(landmarks[i].visibility > 0.90 for i in right_finger):
            return False
        
        if landmarks[19].y < landmarks[23].y and landmarks[20].y < landmarks[24].y:
            return False

        #check if fingers are visible 
        if not (landmarks[8].y < landmarks[6].y and landmarks[7].y < landmarks[5].y and
            landmarks[12].y < landmarks[10].y and landmarks[11].y < landmarks[9].y):

            return True
        
    return False


# Function to check if any hand is in front of the chest
def is_hand_in_front_of_chest(landmarks):
    # print("1")
    if not landmarks:
        return False
    # print("2")
    # Indices of the relevant landmarks for the left and right hand
    left_hand_indices = [15, 17, 19, 21]
    right_hand_indices = [16, 18, 20, 22]

    # Get the y-coordinate of the chest landmark
    left_chest_y = landmarks[11].y  # Assuming landmark 11 corresponds to the chest
    right_chest_y = landmarks[12].y

    # Check if any one hand is in front of the chest
    left_hand_in_front = any(landmarks[i].y < left_chest_y for i in left_hand_indices)
    right_hand_in_front = any(landmarks[i].y < right_chest_y for i in right_hand_indices)

    return left_hand_in_front or right_hand_in_front

# Function to capture and process a photo
def capture_photo():
    global frame
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    photo_name = f"front_pose_{timestamp}.jpg"
    photo_name = "opencv_frame_{}.jpg"
    cv2.imwrite(photo_name, frame)
    print(f"Photo captured: {photo_name}")

     # Pass the captured image to is_front_pose
    image_for_front_pose = cv2.imread(photo_name)
    image_for_front_pose = cv2.cvtColor(image_for_front_pose, cv2.COLOR_BGR2RGB)

    # Process the image with is_front_pose
    results = posef.process(image_for_front_pose)
    landmarks = results.pose_landmarks

    if landmarks is not None:
        land = landmarks.landmark
        if is_front_pose(land):
            print("Front pose detected in the captured image!")
        else:
            print("Not in a front pose in the captured image.")
    else:
        print("No pose landmarks detected in the captured image.")

    
# Function for countdown and photo capture
def countdown_and_capture():
    for countdown in range(4, 0, -1):
        print(countdown)
        time.sleep(1)
    capture_photo()

# Function to handle hands detected in front
def handle_hands_in_front():
    print("Hands are in front! Starting countdown...")
    countdown_thread = threading.Thread(target=countdown_and_capture)
    countdown_thread.start()

# Mediapipe Pose Detection setup
pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
posef = pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

cap = cv2.VideoCapture(0)

skip_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (768, 768))
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Skip frames when hands are detected
    if skip_frames > 0:
        skip_frames -= 1
        continue

    # Skip frames when hands are detected    
    results = posef.process(frameRGB)
    landmarks = results.pose_landmarks
    if landmarks is not None:
        land = landmarks.landmark
        if is_hand_in_front_of_chest(land):
                handle_hands_in_front()
                skip_frames = 30 # Skip the next 30 frames after hands are detected
                    
    
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks,
                                  connections=pose.POSE_CONNECTIONS)

        height, width, _ = frame.shape
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              int(landmark.z * width)))
    
    print(len(landmarks))
    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
