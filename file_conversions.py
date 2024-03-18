import numpy as np
import cv2
from django.core.files.base import ContentFile
from .seq_converter import open_file, get_scaled_frame, check_length

SIM_THRESH = 0.8


def get_content_files(video_file, conv_option='', interval=1):
    dynamic = True if conv_option == 'dynamic' else False
    file_type = video_file.name.split('.')[-1].lower()

    if file_type == 'mp4':
        return gcf_mp4(video_file, dynamic, interval)    
    elif file_type == 'seq':
        return gcf_seq(video_file, dynamic, interval)


def gcf_seq(input, dynamic, interval):
    """Extract frames from the .seq video without saving them anywhere. Not part of the commandline tool.

    Args:
        input (str): The full path of the seq file
        dynamic (bool): Whether to extract frames only when they are different from the previous frame
        output_format (str, optional): Defaults to 'png'.

    Raises:
        OSError: If the file couldn't be read by the ImagerFile module.
        EOFError: If the frame couldn't be encoded as png.

    Returns:
        list[ContentFile]: A list of ContentFile objects representing the extracted frames.
    """
    # List containing the frame files
    frames = []
    # Open .seq file for reading
    im_1, im_2, im_3 = open_file(input)
    # Get the padding number for frame file names
    len_frame_num = len(str(im_1.num_frames)) + 1
    # Set frame units
    allowed_units_list = list(im_1.supported_units)
    check_length(allowed_units_list)
    im_1.unit = allowed_units_list[0]
    im_2.unit = allowed_units_list[1]
    im_3.unit = allowed_units_list[2]
    # Extract and save frames
    if not dynamic:
        for i in range(im_1.num_frames):
            frame_1 = get_scaled_frame(im_1, i)
            frame_2 = get_scaled_frame(im_2, i)
            frame_3 = get_scaled_frame(im_3, i)
            # Concatenate the frames to get a single three-channel image
            concated = np.array([frame_1, frame_2, frame_3]).swapaxes(0, 2).swapaxes(1, 0)
            # Save the frame
            if i % interval == 0:
                try:
                    # Encode frame as png
                    ret, buf = cv2.imencode('.png', concated)
                    if not ret:
                        raise EOFError(f"Error encoding frame {i}")
                    # Create ContentFile object and append it to list
                    f = ContentFile(buf.tobytes())
                    f.name = "{}.png".format(str(i).zfill(len_frame_num))
                    frames.append(f)
                except Exception as e:
                    print(f"Frame {i} couldn't be saved")
                    raise e
    else:
        prev_frame = None
        for i in range(im_1.num_frames):
            frame_1 = get_scaled_frame(im_1, i)
            frame_2 = get_scaled_frame(im_2, i)
            frame_3 = get_scaled_frame(im_3, i)
            # Concatenate the frames to get a single three-channel image
            concated = np.array([frame_1, frame_2, frame_3]).swapaxes(0, 2).swapaxes(1, 0)
            # Save the frame
            if prev_frame is None or different(prev_frame, concated):
                try:
                    # Encode frame as png
                    ret, buf = cv2.imencode('.png', concated)
                    if not ret:
                        raise EOFError(f"Error encoding frame {i}")
                    # Create ContentFile object and append it to list
                    f = ContentFile(buf.tobytes())
                    f.name = "{}.png".format(str(i).zfill(len_frame_num))
                    frames.append(f)
                except Exception as e:
                    print(f"Frame {i} couldn't be saved")
                    raise e
            prev_frame = concated
    # Close the file readers
    im_1.close()
    im_2.close()
    im_3.close()

    return frames


def gcf_mp4(input, dynamic, interval):
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        print("Error opening video file")

    i = 0
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    len_frame_num = len(str(total_frames)) + 1

    if not dynamic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            if i % interval == 0:
                ret, buf = cv2.imencode(".png", frame)
                if not ret:
                    raise EOFError(f"Error encoding frame {i}")
                
                f = ContentFile(buf.tobytes())
                f.name = "{}.png".format(str(i).zfill(len_frame_num))

                frames.append(f)
            
            i += 1

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
                    print("Error encoding frame {}".format(i))
                
                f = ContentFile(buf.tobytes())
                f.name = "{}.png".format(str(i).zfill(len_frame_num))

                frames.append(f)
            
            prev_frame = frame
            i += 1

    # When everything done, release the video capture object
    cap.release()
    return frames

def different(x, y):
    assert x.ndim == y.ndim
    assert x.shape == y.shape

    similarity_arr = x == y
    return np.sum(similarity_arr) / similarity_arr.size < SIM_THRESH

