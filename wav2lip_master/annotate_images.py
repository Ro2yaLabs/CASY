import torch
import face_detection
import cv2
import matplotlib.pyplot as plt

import numpy as np
import argparse
import os
from os import listdir, path
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
                                 
parser.add_argument('--video', type=str, 
					help='Filepath of video/image that contains faces to use', 
					required=True)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

args = parser.parse_args()
# fps = 25
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

def face_detect(images):

	batch_size = 1
	
	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = [0, 0, 0, 0]
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	results = [[image[y1: y2, x1:x2], (x1, y1, x2, y2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results

def cvt_to_yolo_format(coords, image_width, image_height):
    x1, y1, x2, y2 = coords

    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return x_center, y_center, width, height



def main():
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, 
											device=device)
    
    vidcap = cv2.VideoCapture(args.video)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Reading video frames from start...')

    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 0
	
    for frameNumber in tqdm(range(total_frames)):
        _, frame = vidcap.read()
        
        _, coords = face_detect([frame])[0]

        x_center, y_center, width, height = cvt_to_yolo_format(coords, width, height)

        cv2.imwrite(f"yolo/data/face{i}.jpg", frame)
        with open(f"yolo/data/face{i}.txt", 'w') as f:
            f.write(f"{0} {x_center} {y_center} {width} {height}")

        i += 1
		
if __name__ == "__main__":
	main()