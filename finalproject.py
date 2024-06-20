sudo apt update
sudo apt install python3-pip libopencv-dev python3-opencv python3-numpy



import cv2
import face_recognition  # Install using `pip install face_recognition`

# Load pre-trained face recognition model (Haar cascade or dlib's CNN)
# This example uses OpenCV's Haar cascade classifier for simplicity.
face_cascade = cv2.CascadeClassifier('/path/to/haarcascade_frontalface_default.xml')

# Known faces with their corresponding encodings (replace with your data)
known_face_encodings = []
known_face_names = []

# Function to load facial encodings from images in a directory
def load_known_faces(data_dir):
    for filename in os.listdir(data_dir):
        # Load image
        img = cv2.imread(os.path.join(data_dir, filename))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for face_recognition

        # Detect faces
        faces = face_cascade.detectMultiScale(rgb_img)

        # Extract encodings for detected faces (if any)
        if len(faces) > 0:
            face_encoding = face_recognition.face_encodings(rgb_img, faces)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Extract name from filename

# Main loop
def main():
    # Load known faces (replace with your data path)
    load_known_faces('/path/to/your/facial_data')

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert frame to grayscale (optional, may improve detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Recognize faces
        face_locations = []
        face_encodings = []
        face_names = []
        for (x, y, w, h) in faces:
            # Extract the ROI (Region of Interest) for the face
            face_roi = frame[y:y+h, x:x+w]
            # Convert the ROI to RGB color (for face_recognition)
            rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_face_roi)

            for face_encoding in face_encodings:
                # Compare face encoding with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Check for matches
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Add face location and name for drawing bounding boxes
                face_locations.append((y, x, y+h, x+w))
                face_names.append(name)

        # Draw bounding boxes and labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
