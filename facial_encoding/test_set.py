"""Read through test images.

This application reads through the the dataset and test the last element in
each sub-directory and move it to test.

"""

import os
import glob
import shutil


def create_test_images():
    """Run application."""
    directory = "dataset"
    test_directory = "test_images"
    total = 0
    sub_directories = []
    sub_directories = [x[0] for x in os.walk(directory)]
    sub_directories.pop(0)
    if not os.path.isdir(test_directory):
        print('The directory is not present. Creating a new one..')
        os.mkdir(test_directory)
    else:
        print('The directory is present.')
        total = len(os.listdir(test_directory))
        print(total)
    # print(sub_directories)
    for sub_directory in sub_directories:
        list_of_files = glob.glob(os.path.join(sub_directory, '*'))
        # print(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        destination_path = os.path.sep.join([test_directory,
                                            "test{}.png".
                                             format(str(total).zfill(5))])
        shutil.move(latest_file, destination_path)
        total += 1
        print(destination_path)
    print("Test image set created")
