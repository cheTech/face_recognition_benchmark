import face_recognition
import time

cnn_model = False

bidenImg = face_recognition.load_image_file("biden.jpg")
donaldImg = face_recognition.load_image_file("donald.jpg")

t0 = time.clock()

if cnn_model:
    donald_locations = face_recognition.face_locations(donaldImg, model="cnn")
    biden_locations = face_recognition.face_locations(bidenImg, model="cnn")

else:
    donald_locations = face_recognition.face_locations(donaldImg)
    biden_locations = face_recognition.face_locations(bidenImg)

donald_encodings = face_recognition.face_encodings(donaldImg, donald_locations)
biden_encodings = face_recognition.face_encodings(bidenImg, biden_locations)

print("Test Ended!\nScore:%s." % (int((time.clock() - t0) * 1000)))
