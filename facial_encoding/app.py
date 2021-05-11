"""Facial encoding application.

This application does the encodings (128-d vectors) of the faces dataset that
will be used by facial recongintion application.
"""
import edgeiq
import face_recognition
import test_set
import pickle
import cv2
import os


def main():
    """Run facial encoding application."""
    dataset_path = "dataset"
    test_path = "test_images"
    encodings_file = "encodings.pickle"
    known_encodings = []
    known_names = []
    test_set.create_test_images()
    image_paths = list(edgeiq.list_images(dataset_path))
    for(i, image_path) in enumerate(image_paths):
        print("processing image {}/{}".format(i+1, len(image_paths)))
        name = image_path.split(os.path.sep)[-2]
        print(name)
        image = cv2.imread(image_path)
        print(image.shape)
        image = edgeiq.resize(image, width=500, height=500)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image, model="cnn")
        encodings = face_recognition.face_encodings(rgb_image, boxes,
                                                    num_jitters=1)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)
    print("serializing facial encodings")
    data = {"encodings": known_encodings, "names": known_names}
    print(encodings_file)
    f = open(encodings_file, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Test encoding data")
    data = pickle.loads(open("encodings.pickle", "rb").read())
    test_images = list(edgeiq.list_images(test_path))
    for(i, test_image) in enumerate(test_images):
        image = cv2.imread(test_image)
        image = edgeiq.resize(image, width=500, height=500)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image, model="cnn")
        encodings = face_recognition.face_encodings(rgb_image, boxes)
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"
        if True in matches:
            match_indexs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in match_indexs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
            cv2.imshow("Face Prediction", image)
            cv2.waitKey(5000)
    print("Program Ending")


if __name__ == "__main__":
    main()
