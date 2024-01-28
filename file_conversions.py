import numpy as np
import cv2

SIM_THRESH = 0.8

def convert():
    pass

def seq_convert():
    pass

def seq_convert_dynamic():
    pass

def mp4_convert(input, interval=0):
    cap = cv2.VideoCapture(input)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    frames = []
    frame_num = 0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    zfill = len(str(int(total_frames)))
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Save the resulting frame as a .png image
            if not cv2.imwrite(f'frame_{str(frame_num).zfill(zfill)}.png', frame):
                frames.append()
                frame_num += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    return frames

def different(x, y):
    assert x.dim == y.dim
    assert x.shape == y.shape

    similarity_arr = x == y
    return np.sum(similarity_arr) / similarity_arr.size < SIM_THRESH

