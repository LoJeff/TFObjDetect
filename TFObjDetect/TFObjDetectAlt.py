
import numpy as np
import argparse
import imutils
import time
import cv2
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util



# Path to frozen detect graph
PATH_TO_CKPT = "mac_n_cheese_graph/frozen_inference_graph.pb"

# Path to list of objects
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

# Number of objects
NUM_CLASSES = 1

def init_video():
    print("opening camera...")
    vc = cv2.VideoCapture(1)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
        frame = None
        print("failed to open")

    return vc, rval, frame

if __name__ == "__main__":
    # Load frozen Tensorflow model into memory

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Load label map

    labelMap = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=NUM_CLASSES, use_display_name=True)
    categoryIndex = label_map_util.create_category_index(categories)

    # camera set up
    vc, rval, frame = init_video()

    # read from camera
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while rval:
                rval, image_np = vc.read()

                # Change for compatibility with matplotlib
                frame_mpl = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes,scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    categoryIndex,
                    use_normalized_coordinates=True,
                    line_thickness=8)
        
                # resize to have max width of 400 pixels
                frame = imutils.resize(frame, width=1000)

                # show the output frame
                cv2.imshow("object detection", frame)

                key = cv2.waitKey(1)
                if key == 27: # exit on ESC
                    break

    cv2.destroyAllWindows()