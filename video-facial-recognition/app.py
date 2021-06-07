"""Video facial recongintion application.

This application does facial recongintion in real time video stream.
The application use the facial encodings created by the face_encodings
application to identify known faces.
"""
import time
import edgeiq
import face_recognition
import pickle
import cv2


def main():
    """Run real time facial recongintion application."""
    encodings_file = "encodings.pickle"
    data = pickle.loads(open(encodings_file, "rb").read())

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                frame = edgeiq.resize(frame, width=500, height=500)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start_encoding = time.time()
                boxes = face_recognition.face_locations(rgb_frame, model="cnn")
                encodings = face_recognition.face_encodings(rgb_frame, boxes,
                                                            num_jitters=1)
                stop_encoding = time.time()
                encoding_time = (stop_encoding - start_encoding)
                names = []
                for encoding in encodings:
                    matches = face_recognition.compare_faces(data["encodings"],
                                                             encoding)
                    name = "Unknown"
                    if True in matches:
                        match_indexs = [i for (i, b)
                                        in enumerate(matches) if b]
                        counts = {}
                        for i in match_indexs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)
                    names.append(name)
                    for ((top, right, bottom, left), name) in zip(boxes, names):
                        cv2.rectangle(frame, (left, top), (right, bottom),
                                      (0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name, (left, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 255, 0), 2)
                # Generate text to display on streamer
                text = ["Model: {}".format("Dlib ResNet-34")]
                text.append(
                        "Facial encoding time: {:1.3f} s".format(encoding_time))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
