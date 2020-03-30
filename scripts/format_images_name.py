import argparse
import os
import time

import cv2


def main():
    directory = os.listdir(argv.path)
    for index, file_path in enumerate(directory):
        file_ext = os.path.splitext(file_path)[-1][1:]
        if file_ext in argv.extension_file:
            img = cv2.imread(os.path.join(argv.path, file_path), cv2.IMREAD_COLOR)
            size_y = img.shape[0]
            size_x = img.shape[1]
            os.rename(os.path.join(argv.path, file_path),
                      os.path.join(argv.output_path, f"{'{:08d}'.format(index + 1)}_{size_x}_{size_y}.{file_ext}"))
        if index % 1000 == 0:
            print(f"{index} terminé...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--extension-file", nargs='+', default=["jpg"], required=False)
    parser.add_argument("--output-path", nargs='+', required=False)

    argv = parser.parse_args()

    if argv.output_path is None:
        argv.output_path = argv.path
    main()
