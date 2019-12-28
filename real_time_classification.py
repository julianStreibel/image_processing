import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
sys.path.append("./models/research/object_detection/")
from utils import label_map_util
from utils import visualization_utils as vis_util
from helpers import get_id_of_num_in_list
import datetime

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to the actual model for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap("./models/research/object_detection/data/mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# for recording of a special class
class_id = 1 # bird = 16, person = 1
recording = False
fill_frames = 200 # frames without detected class
fill_frames_left = fill_frames
cap=cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
num_of_files = 0

print
print
print("ON AIR")
print
print

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    ret=True
    while (ret):
        ret, image_np=cap.read() 
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)

        # get percent of class
        class_posiotion = get_id_of_num_in_list(classes[0], class_id)
        class_score = scores[0][class_posiotion]
        if class_posiotion >= 0 and class_score > 0.5:
            fill_frames_left = fill_frames
            if not recording:
                time = datetime.datetime.now().strftime("%d_%m_%y-%H_%M_%S")
                print('started at ' + datetime.datetime.now().strftime("%d_%m_%y-%H_%M_%S"))
                out = cv2.VideoWriter('video_content/' + time + '.mp4', fourcc, 7.0, (width, height))
            recording = True
        elif fill_frames_left > 0:
            fill_frames_left -= 1
        elif recording:
            recording = False
            out.release()
            num_of_files += 1
		
        if recording:
            out.write(image_np)

        cv2.imshow('live_detection',image_np)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            out.release()
            cv2.destroyAllWindows()
            cap.release()
            break