"""Read facial encoding pickle file.

This application reads the encodings pickle file (128-d vectors) and names.
"""
import pickle


def main():
    """Run read pickle file application."""
    data = pickle.loads(open("encodings.pickle", "rb").read())
    print(data)


if __name__ == "__main__":
    main()
