import numpy as np
import cv2
from django.core.files.base import ContentFile
from fnv.file import ImagerFile

SIM_THRESH = 0.8


def getContentFiles(video_file, conv_option='', interval=0):
    dynamic = True if conv_option == 'dynamic' else False

    if video_file.name.endswith('.mp4'):
        return getContentFiles_mp4(video_file, dynamic, interval)    
    
    elif video_file.name.endswith('.seq'):
        return getContentFiles_seq(video_file, dynamic, interval)
    

def getContentFiles_seq(input, dynamic, interval):
    pass


def getContentFiles_mp4(input, dynamic, interval):
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        print("Error opening video file")

    frames = []
    frame_num = 0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    zfill = len(str(int(total_frames)))

    if not dynamic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            if frame_num % interval:
                ret, buf = cv2.imencode(".png", frame)
                if not ret:
                    print("Error encoding frame {}".format(frame_num))
                
                f = ContentFile(buf.tobytes())
                f.name = "{}.png".format(str(frame_num).zfill(zfill))

                frames.append(f)
            
            frame_num += 1

    else:
        prev_frame = None
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret: 
                break

            if prev_frame is None or different(prev_frame, frame):
                ret, buf = cv2.imencode(".png", frame)
                if not ret:
                    print("Error encoding frame {}".format(frame_num))
                
                f = ContentFile(buf.tobytes())
                f.name = "{}.png".format(str(frame_num).zfill(zfill))

                frames.append(f)
            
            prev_frame = frame
            frame_num += 1

    # When everything done, release the video capture object
    cap.release()
    return frames

def different(x, y):
    assert x.ndim == y.ndim
    assert x.shape == y.shape

    similarity_arr = x == y
    return np.sum(similarity_arr) / similarity_arr.size < SIM_THRESH

