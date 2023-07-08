__author__ = "Simon Nilsson"


# MODIFIED FROM https://gist.github.com/HaydenFaulkner/54318fd3e9b9bdb66c5440c44e4e08b8
# Medium article https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661
# All cred to Hayden Faulkner, ta!

from concurrent.futures import ProcessPoolExecutor
import cv2
import multiprocessing
import os

def extract_frames(video_filename, frames_dir, overwrite=True, start=-1, end=-1, every=1):
    capture = cv2.VideoCapture(video_filename)
    if start < 0: 
        start = 0
    if end < 0: 
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)
    frame = start 
    while_safety = 0 
    saved_count = 0

    while frame < end:

        _, image = capture.read()

        if while_safety > 500:
            break
        if image is None: 
            while_safety += 1 
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, "{:0d}.png".format(frame))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_filename, frames_dir, overwrite=True, every=1, chunk_size=1000):

    capture = cv2.VideoCapture(video_filename)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("The video has no frames")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total)  # make sure last chunk has correct end frame

    print("Extracting " + str(total) + " frames from " + str(os.path.basename(video_filename)) + '...')

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_filename, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

