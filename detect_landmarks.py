# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import copy

def convert_to_bbox(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def convert_to_numpy(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def main():
    img_path = "Faces.jpg"            # path to image
    model_path = "shape_predictor_68_face_landmarks.dat"     #path to the face detection model

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    jawline_img = copy.copy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     #convert BGR image to gray
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):  # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = convert_to_numpy(shape)
        #print(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
    	# [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = convert_to_bbox(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    	# show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw them on the image
        for (x, y) in shape:
            jawline = shape[0:16]
            pts = np.array(jawline, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(jawline_img, [pts], False, (255,0,0))   # show jawline curve on the image
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1) # show the output image with the face detections + facial landmarks
    cv2.imwrite("Output_faces.jpg", image)        # write the new image array in new file
    cv2.imwrite("Jawlines.jpg", jawline_img)      #write new jawline image array in new file

if __name__ == '__main__':
    main()
