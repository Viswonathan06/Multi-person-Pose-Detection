# Importing necessary files
import numpy as np
import time
import cv2
import imutils
from imutils.video import FPS
import cv2
import mediapipe as mp
import time
from imutils.video import VideoStream


# Input file name
INPUT_FILE='trial_004_kinect_webcam.avi'
# OUTPUT_FILE='output.avi'
# Getting the YOLO files and defining confidence threshold
LABELS_FILE='data/coco.names'
CONFIG_FILE='cfg/yolov3.cfg'
WEIGHTS_FILE='yolov3.weights'
CONFIDENCE_THRESHOLD=0.3

H=None
W=None

# Defining the Mediapipe pose detection class with all functions
class poseDetector():

	# Initializing the variables with default values
    def __init__(self, mode= False, upBody = False, smooth= True,
                 detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,
                                        self.upBody,
                                        self.smooth,
                                        self.detectionConf,
                                        self.trackConf)

	# Detecting the pose with image as input
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if (self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    # Drawing landmarks and printing results.
    def findPosition(self, img, draw = True):
        lmlist = []
        if(self.results.pose_landmarks):
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
        return lmlist


fps = FPS().start()

# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
# 	(800, 600), True)

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# Reading the neural network from Yolo framework for detection of "person"
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
# Defining the video input
vs = cv2.VideoCapture(INPUT_FILE)


# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cnt =0;

# Defining the poseDetector class
detector = poseDetector()

# Looping through the frames
while True:
	cnt+=1
	print ("Frame number", cnt)
	try:
		(grabbed, image) = vs.read()
	except:
		break
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	if W is None or H is None:
		(H, W) = image.shape[:2]
	layerOutputs = net.forward(ln)






	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# print(classID, " Class id IS/")
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				if LABELS[classID] == "person":
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))

					classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
		CONFIDENCE_THRESHOLD)
	imageArr = []
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]
			xmin, ymin, xmax, ymax = boxes[i]
			# crop detection from image (take an additional 5 pixels around all edges)
			cropped_img = image[int(y) - 5:int(y+h) + 5, int(x) - 5:int(x+w) + 5]
			imageArr.append(cropped_img)
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)


	# show the output image

	cv2.imshow("output", cv2.resize(image,(800, 600)))
	i = 0
	for cropped_img in imageArr:

		img = detector.findPose(cropped_img)
		lmlist = detector.findPosition(img, True)
		print(lmlist)
		cv2.imshow("crooped" + str(i), cv2.resize(cropped_img, (200, 600)))
		i = i+1
	# writer.write(cv2.resize(image,(800, 600)))
	fps.update()
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

# release the file pointers
print("[INFO] cleaning up...")
# writer.release()
vs.release()