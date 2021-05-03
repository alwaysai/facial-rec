"""Facial encoding application.

This application does the encodings (128-d vectors) of the faces dataset that
will be used by facial recongintion application.
"""
import edgeiq
import face_recognition
import pickle
import cv2
import os


def main():
    """Run facial encoding application."""
    dataset_path = "dataset"
    encodings_file = "encodings.pickle"
    image_paths = list(edgeiq.list_images(dataset_path))
    known_encodings = []
    known_names = []
    for(i, image_path) in enumerate(image_paths):
        print("processing image {}/{}".format(i+1, len(image_paths)))
        name = image_path.split(os.path.sep)[-2]
        print(name)
        image = cv2.imread(image_path)
        print(image.shape)
        image = edgeiq.resize(image, width=500, height=500)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image, model="cnn")
        encodings = face_recognition.face_encodings(rgb_image, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)
    print("serializing facial encodings")
    data = {"encodings": known_encodings, "names": known_names}
    print(encodings_file)
    f = open(encodings_file, "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == "__main__":
    main()
