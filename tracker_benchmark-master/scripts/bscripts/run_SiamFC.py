#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Support integration with OTB benchmark"""





import logging
import os
import sys
import time
from os.path import join as opjoin
import tensorflow as tf
import config
# Code root absolute path

CODE_ROOT = os.path.join(config.WORKDIR,'SiamFC-TensorFlow-master')

# Checkpoint for evaluation
CHECKPOINT = opjoin(CODE_ROOT,opjoin('Logs',opjoin('SiamFC', opjoin('track_model_checkpoints'
                                    ,'SiamFC-3s-color-pretrained'  )
                                                    )
                                    )
                    )

sys.path.insert(0, CODE_ROOT)

from siamfc.utils.misc_utils import auto_select_gpu, load_cfgs
from siamfc.inference import inference_wrapper
from siamfc.inference.tracker import Tracker
from siamfc.utils.infer_utils import Rectangle

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


def run_SiamFC(seq, rp, bSaveImage):
  checkpoint_path = CHECKPOINT
  logging.info('Evaluating {}...'.format(checkpoint_path))

  # Read configurations from json
  model_config, _, track_config = load_cfgs(checkpoint_path)

  track_config['log_level'] = 0  # Skip verbose logging for speed

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint_path)
  g.finalize()

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.compat.v1.Session(graph=g, config=sess_config) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    tracker = Tracker(model, model_config, track_config)

    tic = time.clock()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_bb = Rectangle(x - 1, y - 1, width, height)

    trajectory_py = tracker.track(sess, init_bb, frames)
    trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in
                  trajectory_py]  # x, y add one to match OTB format
    duration = time.clock() - tic

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result
