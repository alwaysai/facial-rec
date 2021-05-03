"""
Facial enrollment application.

Use object detection to detect human faces in the frame in realtime and save
the image into dataset for use in facial encoding model.

Running Facial enrollment application
This application is designed to run on a alwaysai native environment
(Windows PC or Mac) and not an edge device. To start the application at the
command line enter:

aai app start -- --name frist_last

To save an image press the k button on the computer and to exit the application
press q button.

To properly do facial encoding the goal is to create a diverse set of images
representative of the particular person’s.  You will need a minimum of six
images and ideally if the process can be performed over multiple days or weeks
will create a superior  facial dataset.  You will want the faces capture in :
    1. Different lighting conditions
    2. Times of day
    3. Moods and emotional states
    4. Glasses on and off
"""

import time
import edgeiq
import argparse
import os
import cv2


def main(name):
    """Run facial enrollment application."""
    total = 0
    data_directory = os.path.abspath(os.path.join("dataset", name))
    if not os.path.isdir(data_directory):
        print('The directory is not present. Creating a new one..')
        os.mkdir(data_directory)
    else:
        print('The directory is present.')
        total = len(os.listdir(data_directory))
        print(total)
    facial_detector = edgeiq.ObjectDetection(
            "alwaysai/res10_300x300_ssd_iter_140000")
    facial_detector.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(facial_detector.engine))
    print("Accelerator: {}\n".format(facial_detector.accelerator))
    print("Model:\n{}\n".format(facial_detector.model_id))

    try:
        with edgeiq.WebcamVideoStream(cam=0) as webcam:
            # Allow webcam to warm up
            time.sleep(2.0)


            # loop detection
            while True:
                frame = webcam.read()
                original_frame = frame.copy()

                # detect human faces
                results = facial_detector.detect_objects(
                        frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, results.predictions,
                        colors=facial_detector.colors)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(3) & 0xFF
                if key == ord("k"):
                    path = os.path.sep.join([data_directory, "{}.png".
                                            format(str(total).zfill(5))])
                    cv2.imwrite(path, original_frame)
                    total += 1
                    print("k key has been pressed")
                elif key == ord("q"):
                    break
    finally:
        print("Program Ending")
        print("Total {} face images stored for this person".format(total))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Facial Dataset')
    parser.add_argument('--name', required=True,
                        help="name of the personfor example joe_smith")
    args = parser.parse_args()
    main(args.name)
