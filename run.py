import face_recognition
import cv2
import uuid
import time
import kairos_face
import datetime



kairos_face.settings.app_id = "4822f948"
kairos_face.settings.app_key = "5219a64222589acc6b1a2fb47c39febd"

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
]
known_face_names = [
    "Barack Obama",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def kairos(crop, cuantos):
    cv2.imwrite("faces/{}.jpg".format(unique_filename), crop)
    cv2.imwrite("faces/face.jpg".format(unique_filename), crop)
    recognized_faces = kairos_face.detect_face(file="faces/face.jpg")
    if recognized_faces != "Error":
        edad = recognized_faces['images'][0]['faces'][0]['attributes']['age']
        sexo = recognized_faces['images'][0]['faces'][0]['attributes']['gender']['type']
        if sexo == 'M':
            sexo = "HOMBRE"
        else:
            sexo = "MUJER"
        print("edad: {}, sexo: {}".format(edad, sexo))
        now = datetime.datetime.now()
        img = cv2.imread('p3.png',1)
        font = cv2.FONT_HERSHEY_DUPLEX
        hora = str(now.hour) + ":" + str(now.minute)
        age = str(edad)

        cv2.putText(img, hora, (950, 510), font, 1.7, (0, 0, 255), 2)
        cv2.putText(img, sexo, (680, 583), font, 1.7, (0, 0, 255), 2)
        cv2.putText(img, age, (870, 663), font, 1.7, (0, 0, 255), 2)

        s_img = cv2.imread("faces/face.jpg")
        s_img2 = cv2.resize(s_img,(int(443),int(590)))
        cv2.imwrite("faces/face2.jpg", s_img2)
        s_img3 = cv2.imread("faces/face2.jpg")
        cv2.imwrite("faces/face2.jpg", s_img2)
        x_offset=0
        y_offset=210
        img[y_offset:y_offset+590, x_offset:x_offset+443] = s_img3
        cv2.imwrite("faces/anuncio.jpg", img)
        from subprocess import call
        #call(["curl" ,"-F", "sampleFile=@p2.png", "http://localhost:3000/upload"])
        


framecount = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #
        # face_names = []
        # for face_encoding in face_encodings:
        #     # See if the face is a match for the known face(s)
        #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #     name = "Unknown"
        #
        #     # If a match was found in known_face_encodings, just use the first one.
        #     if True in matches:
        #         first_match_index = matches.index(True)
        #         name = known_face_names[first_match_index]
        #
        #     face_names.append(name)

    process_this_frame = not process_this_frame

    #time.sleep(3)

    cuantos = len(face_locations)
    # Display the results
    for (top, right, bottom, left) in face_locations:

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #Random name
        unique_filename = str(uuid.uuid4())

        #Crop face
        crop = frame[top:bottom, left:right]
        framecount += 1


        if framecount % 90 == 0:
            kairos(crop, cuantos)


        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
