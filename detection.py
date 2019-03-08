# -*- coding:utf-8 -*-

# Tensorflow obj detection api wrapped as a class

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util as lab_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops


class Detection:
    def __init__(self, graph, classes, labels):

        self.graph_path = graph  # 模型路径
        self.classes = classes  # 类别数，1表示只检测person
        self.label_map = lab_util.load_labelmap(labels)  # label 映射
        self.categories = lab_util.convert_label_map_to_categories(self.label_map,
                                                                   max_num_classes=self.classes,
                                                                   use_display_name=True)
        self.category_index = lab_util.create_category_index(self.categories)
        self.img_size = (640, 480)  # 可以更改
        self.img = None
        self.output_dict = {}

        print('loading tf graph')

        # 从文件中读取tf.Graph的配置，加载到当前图中
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.gfile.GFile(self.graph_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def)

        # 创建当前图的session
        self.sess = tf.Session(graph=self.graph)
        self.ops = self.graph.get_operations()

        print('essentials loaded !')

    # 转换图像
    def _set_img(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        return

    # 调用session 处理图像
    def process(self, img):
        self._set_img(img)

        all_tensor_names = {output.name for op in self.ops for output in op.outputs}
        tensor_dict = {}

        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
            tensor_name = 'import/' + key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
                # print('found',tensor_name)

        if 'detection_masks' in tensor_dict:
            # for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, self.img_size[0], self.img_size[1]
            )
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8
            )
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = self.graph.get_tensor_by_name('import/image_tensor:0')

        # 检测
        self.output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(self.img, 0)})

        # 输出结果的数据类型为float32,需要转换为合适的类型
        self.output_dict['num_detections'] = int(self.output_dict['num_detections'][0])
        self.output_dict['detection_classes'] = self.output_dict['detection_classes'][0].astype(np.uint8)
        self.output_dict['detection_boxes'] = self.output_dict['detection_boxes'][0]
        self.output_dict['detection_scores'] = self.output_dict['detection_scores'][0]

        if 'detection_masks' in self.output_dict:
            self.output_dict['detection_masks'] = self.output_dict['detection_masks'][0]

        return self.output_dict

    def draw_detection(self):
        vis_util.visualize_boxes_and_labels_on_image_array(
            self.img,
            self.output_dict['detection_boxes'],
            self.output_dict['detection_classes'],
            self.output_dict['detection_scores'],
            self.category_index,
            instance_masks=self.output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=3,
            skip_labels=True,
            skip_scores=True)
        # image = np.uint8(image_np)
        output_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        return output_img

    def terminate(self):
        self.sess.close()

# if __name__=="__main__":
#
#     BASE_DIR = 'D:/Code/web/py/display/src/'
#     MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#     MODEL_FILE = MODEL_NAME + '.tar.gz'
#     DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#     PATH_TO_GRAPH = BASE_DIR + MODEL_NAME + '/frozen_inference_graph.pb'
#     PATH_TO_LABELS = BASE_DIR + 'data/' + 'mscoco_label_map.pbtxt'
#     NUM_CLASSES = 1
#
#     detect=Detection(
#         model=BASE_DIR+MODEL_FILE,
#         graph=PATH_TO_GRAPH,
#         labels=PATH_TO_LABELS,
#         classes=NUM_CLASSES
#     )
#
#     img=cv2.imread('./sample.jpg')
#     out=detect.process(img)
#     pre=detect.draw_detection()
#     cv2.imwrite('prediction.jpg',pre)
