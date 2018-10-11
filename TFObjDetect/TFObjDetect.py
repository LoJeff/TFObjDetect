
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

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == "__main__":
    # Load frozen Tensorflow model into memory

    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        odGraphDef = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

    # Load label map

    labelMap = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=NUM_CLASSES, use_display_name=True)
    categoryIndex = label_map_util.create_category_index(categories)

    # camera set up
    vc, rval, frame = init_video()

    # read from camera
    while rval:
        rval, frame = vc.read()

        # Change for compatibility with matplotlib
        frame_mpl = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detection
        output = run_inference_for_single_image(frame_mpl, detectionGraph)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output['detection_boxes'],
            output['detection_classes'],
            output['detection_scores'],
            categoryIndex,
            instance_masks=output.get('detection_masks'),
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