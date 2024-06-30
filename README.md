
# FrontPoseDetector 📸🤖

### FrontPoseDetector is a Python-based tool that leverages OpenCV and MediaPipe to detect whether a person is in a front-facing pose in both images and live video feeds. This tool can be useful for various applications, such as fitness tracking, gesture recognition, and many more.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe
```

## Files

### `frontposedetector.py`

This script processes an image to detect whether the subject is in a front pose.

#### Key Functions

- `calculate_angle(v1, v2)`: Calculates the angle between two vectors.
- `is_front_pose(landmarks)`: Determines if the detected pose is a front pose based on specific landmark visibility and angles.

#### Usage

1. Load an image using OpenCV.
2. Resize and convert the image to RGB.
3. Use MediaPipe to process the image and extract pose landmarks.
4. Check if the pose is a front pose using the `is_front_pose` function.
5. Draw landmarks on the image and display the result.

To run the script:
```bash
python frontPose.py
```

### `FrontVidpose.py`

This script captures video from the webcam and detects if the subject is in a front pose in real-time. It also captures a photo if hands are detected in front of the chest and verifies the front pose in the captured image.

#### Key Functions

- `calculate_angle(v1, v2)`: Calculates the angle between two vectors.
- `is_front_pose(landmarks)`: Determines if the detected pose is a front pose based on specific landmark visibility and angles.
- `is_hand_in_front_of_chest(landmarks)`: Checks if any hand is in front of the chest.
- `capture_photo()`: Captures a photo from the webcam.
- `countdown_and_capture()`: Counts down and captures a photo.
- `handle_hands_in_front()`: Handles the event when hands are detected in front.

#### Usage

1. Initialize the webcam capture.
2. Process each frame to detect pose landmarks.
3. Check if hands are in front of the chest and handle the event accordingly.
4. Display the pose landmarks on the video feed.

To run the script:
```bash
python frontVidpose.py
```

## Demo

### `frontposedetector.py`

**Front Pose** | **Not in Front Pose**
--- | ---
![Front Pose](https://github.com/sahilshukla3003/FrontPose-Detector/assets/124785012/902498f6-bf66-4e95-a52d-e129ddf8565b) | ![Not in Front Pose](https://github.com/sahilshukla3003/FrontPose-Detector/assets/124785012/392f2465-afab-484e-ba76-264945721cac)


### `FrontVidpose.py`

The `FrontVidpose.py` script opens the webcam and processes the video feed in real-time. When the user shows hand gestures in front of the chest, the tool starts a countdown (5 seconds). After the countdown, it captures a photo and checks if the user is in a front pose. The captured photo is then saved.

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
```
