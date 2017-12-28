# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys

# import matplotlib.pyplot as plt
import scipy.misc
import tensorflow as tf


def load_labels():
    return [line.rstrip() for line in tf.gfile.GFile('output_labels.txt')]


def load_graph():
    with tf.Session() as sess:
        with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


def run_graph(image, graph):
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions, = sess.run(softmax_tensor, {'Cast:0': image})
    return predictions


TILE_SIZE = 40
def get_tile_dims(image): 
    w,h,_ = image.shape
    return int(w / TILE_SIZE), int(h / TILE_SIZE)


def tiles_generator(image): 
    image = image[:,:,:3]
    w,h,_ = image.shape
    array = np.zeros(image.shape) 
    for i in range(0,w,TILE_SIZE):
        for j in range(0,h,TILE_SIZE):
            array[i:i+TILE_SIZE,j:j+TILE_SIZE] = image[i:i+TILE_SIZE,j:j+TILE_SIZE]
            yield array 
            array[i:i+TILE_SIZE,j:j+TILE_SIZE] = 0

def get_image_heatmap(image, graph): 
    tiles = tiles_generator(image)
    width, height = get_tile_dims(image) 
    heatmap = np.zeros((width,height,2))
    for i in range(width): 
        for j in range(height): 
            heatmap[i][j] = run_graph(next(tiles), graph)
    return heatmap


def get_images(fpaths): 
    return map(scipy.misc.imread, fpaths) 


def main():
    print('loading labels') 
    labels = load_labels()
    print('loading graph') 
    graph = load_graph()
    malware_path = 'images/malware'
    malware_fpaths = [os.path.join(malware_path, fname) for fname in os.listdir(malware_path)][:2]
    safe_path = 'images/safe'
    safe_fpaths = [os.path.join(safe_path, fname) for fname in os.listdir(safe_path)][:2]
    for class_fpaths in [malware_fpaths, safe_fpaths]:
        for i, img_path in enumerate(class_fpaths):
            image = scipy.misc.imread(img_path)
            heatmap = get_image_heatmap(image, graph) 
            heatmap_path = img_path.replace('.png', '.npy') 
            print(i+1, len(class_fpaths), heatmap_path)
            np.save(heatmap_path, heatmap)


if __name__ == '__main__': 
    main()


