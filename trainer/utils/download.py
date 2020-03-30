import asyncio
import itertools
import time

import cv2
import numpy as np
import requests
from tensorflow.python.lib.io import file_io


async def _download_images_async(work_queue, extension_file, read_flag):
    dataset = []
    while not work_queue.empty():
        blob = await work_queue.get()
        if extension_file is None or blob.name.endswith(extension_file):
            content = blob.download_as_string()
            image = np.asarray(bytearray(content), dtype="uint8")
            image = cv2.imdecode(image, read_flag)
            image = np.atleast_3d(image)
            dataset.append(image)
    return dataset


def download_images_async(blobs, read_flag=cv2.IMREAD_GRAYSCALE, nb_worker=5, extension_file=None):
    async def task():
        nonlocal blobs, extension_file, read_flag, loop, nb_worker
        work_queue = asyncio.Queue()
        for blob in blobs:
            work_queue.put_nowait(blob)

        dataset = await asyncio.gather(
            *[loop.create_task(_download_images_async(work_queue, extension_file, read_flag))
              for _ in range(nb_worker)]
        )
        return dataset

    loop = asyncio.get_event_loop()
    dataset = loop.run_until_complete(task())

    return list(itertools.chain(*dataset))


def download_images(blobs, read_flag=cv2.IMREAD_GRAYSCALE, extension_file=None):
    dataset = []
    for blob in blobs:
        if extension_file is None or blob.name.endswith(extension_file):
            content = blob.download_as_string()
            image = np.asarray(bytearray(content), dtype="uint8")
            image = cv2.imdecode(image, read_flag)
            image = np.atleast_3d(image)
            dataset.append(image)
    return dataset


def download_image(url, read_flag=cv2.IMREAD_GRAYSCALE):
    content = requests.get(url).content
    image = np.asarray(bytearray(content), dtype="uint8")
    image = cv2.imdecode(image, read_flag)
    image = np.atleast_3d(image)
    return image


def download_weight(bucket_name, path, storage):
    sub_folder, file = path.split("/", 1)
    blobs = storage.list_blobs(bucket_name, prefix=sub_folder)
    for blob in blobs:
        if blob.name == path:
            content = blob.download_as_string()
            tmp_file = f"weights_{time.time()}.h5"
            with file_io.FileIO(tmp_file, mode='wb+') as output_f:
                output_f.write(content)
            return tmp_file
    raise (f"No file at 'gs://{bucket_name}/{path}'")


def split_bucket(bucket_path):
    return bucket_path[5:].split("/", 1)
