import cv2
import mediapipe as mp
import math

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
    
    
    # print("Landmark information:")
    # for i, landmark in enumerate(landmarks):
    #     print(f"Landmark {i}: Visibility={landmark.visibility}, X={landmark.x}, Y={landmark.y}, Z={landmark.z}")

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
        
        # if  (landmarks[8].y < landmarks[6].y and landmarks[7].y < landmarks[5].y and
        #         landmarks[12].y < landmarks[10].y and landmarks[11].y < landmarks[9].y):
        #     return False

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

        #check if fingers are visible 
        if not (landmarks[8].y < landmarks[6].y and landmarks[7].y < landmarks[5].y and
            landmarks[12].y < landmarks[10].y and landmarks[11].y < landmarks[9].y):

            return True
        

    return False

image = cv2.imread('C:/Users/sahil/Desktop/Repository/FrontPose Detector/person2.jpg')
# image = cv2.imred("C:/Users/sahil/Desktop/Main/mohamad-khosravi--eb0moHDPBI-unsplash.jpg")


image = cv2.resize(image, (768, 768))
# Example usage with MediaPipe Pose model
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
posef = pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
results = posef.process(imageRGB)
# Your image processing code here...
# ...
# Example usage to check front pose
height, width, _ = image.shape

landmarks = results.pose_landmarks

if landmarks is not None:  # Check if pose landmarks are detected
    landmark=landmarks.landmark
    if is_front_pose(landmark):
        # print(landmarks_p)
        print("Front pose detected!")
    else:
        print("Not in a front pose.")
else:
    print("No pose landmarks detected in the image.")

landmarks = []
if results.pose_landmarks:
            # Draw Pose landmarks on the output image.
            mp_drawing.draw_landmarks(image=image, landmark_list=results.pose_landmarks,
                                    connections=pose.POSE_CONNECTIONS)

            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:

                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))
            cv2.imshow("last",image)
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            # mp_drawing.plot_landmarks(results.pose_world_landmarks, pose.POSE_CONNECTIONS)


