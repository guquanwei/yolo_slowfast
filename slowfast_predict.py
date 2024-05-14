#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import numpy, cv2, json
import torch
# import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
# from slowfast.datasets.ava_helper import parse_bboxes_file
from slowfast.datasets import cv2_transform
# from slowfast.datasets.utils import get_sequence
from slowfast.models.video_model_builder import SlowFast
# from slowfast.utils import misc
from slowfast.utils.env import pathmgr
from slowfast.visualization.utils import process_cv2_inputs
# from slowfast.visualization.video_visualizer import VideoVisualizer

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

logger = logging.get_logger(__name__)


class AVAVisualizerWithPrecomputedBox:
    """
    Visualize action predictions for videos or folder of images with precomputed
    and ground-truth boxes in AVA format.
    """

    def __init__(self, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        # self.source = pathmgr.get_local_path(path=cfg.DEMO.INPUT_VIDEO)
        # self.fps = None
        # if pathmgr.isdir(self.source):
        #     self.fps = cfg.DEMO.FPS
        #     self.video_name = self.source.split("/")[-1]
        #     self.source = os.path.join(
        #         self.source, "{}_%06d.jpg".format(self.video_name)
        #     )
        # else:
        #     self.video_name = self.source.split("/")[-1]
        #     self.video_name = self.video_name.split(".")[0]

        args = parse_args()
        args.cfg_files = 'SLOWFAST_32x2_R101_50_50.yaml' 
        cfg = load_config(args, args.cfg_files)
        cfg = assert_and_infer_cfg(cfg)
        self.cfg = cfg

        self.class_names, _, _ = self.get_class_names("ava.json", None, None)

        # Set random seed from configs.
        np.random.seed(self.cfg.RNG_SEED)
        torch.manual_seed(self.cfg.RNG_SEED)

        # Setup logging format.
        # logging.setup_logging(self.cfg.OUTPUT_DIR)

        # Print config.
        # logger.info("Run demo with config:")
        logger.info(self.cfg)
        assert (
            self.cfg.NUM_GPUS <= 1
        ), "Cannot run demo visualization on multiple GPUs."

        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        if torch.cuda.is_available():
            assert (
                cfg.NUM_GPUS <= torch.cuda.device_count()
            ), "Cannot use more GPU devices than available"
        else:
            assert (
                cfg.NUM_GPUS == 0
            ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

        # Construct the model
        # name = cfg.MODEL.MODEL_NAME
        self.model = SlowFast(cfg)

        if cfg.BN.NORM_TYPE == "sync_batchnorm_apex":
            try:
                import apex
            except ImportError:
                raise ImportError("APEX is required for this model, pelase install")

            logger.info("Converting BN layers to Apex SyncBN")
            process_group = apex.parallel.create_syncbn_process_group(
                group_size=cfg.BN.NUM_SYNC_DEVICES
            )
            self.model = apex.parallel.convert_syncbn_model(
                self.model, process_group=process_group
            )

        if cfg.NUM_GPUS:
            if gpu_id is None:
                # Determine the GPU used by the current process
                cur_device = torch.cuda.current_device()
            else:
                cur_device = gpu_id
            # Transfer the model to the current GPU device
            self.model = self.model.cuda(device=cur_device)
        self.model.eval()
        self.cfg = cfg

        # if cfg.DETECTION.ENABLE:
        #     self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    

    def get_class_names(self, path, parent_path=None, subset_path=None):
        """
        Read json file with entries {classname: index} and return
        an array of class names in order.
        If parent_path is provided, load and map all children to their ids.
        Args:
            path (str): path to class ids json file.
                File must be in the format {"class1": id1, "class2": id2, ...}
            parent_path (Optional[str]): path to parent-child json file.
                File must be in the format {"parent1": ["child1", "child2", ...], ...}
            subset_path (Optional[str]): path to text file containing a subset
                of class names, separated by newline characters.
        Returns:
            class_names (list of strs): list of class names.
            class_parents (dict): a dictionary where key is the name of the parent class
                and value is a list of ids of the children classes.
            subset_ids (list of ints): list of ids of the classes provided in the
                subset file.
        """
        try:
            with pathmgr.open(path, "r") as f:
                class2idx = json.load(f)
        except Exception as err:
            print("Fail to load file from {} with error {}".format(path, err))
            return

        max_key = max(class2idx.values())
        class_names = [None] * (max_key + 1)

        for k, i in class2idx.items():
            class_names[i] = k

        class_parent = None
        if parent_path is not None and parent_path != "":
            try:
                with pathmgr.open(parent_path, "r") as f:
                    d_parent = json.load(f)
            except EnvironmentError as err:
                print(
                    "Fail to load file from {} with error {}".format(
                        parent_path, err
                    )
                )
                return
            class_parent = {}
            for parent, children in d_parent.items():
                indices = [
                    class2idx[c] for c in children if class2idx.get(c) is not None
                ]
                class_parent[parent] = indices

        subset_ids = None
        if subset_path is not None and subset_path != "":
            try:
                with pathmgr.open(subset_path, "r") as f:
                    subset = f.read().split("\n")
                    subset_ids = [
                        class2idx[name]
                        for name in subset
                        if class2idx.get(name) is not None
                    ]
            except EnvironmentError as err:
                print(
                    "Fail to load file from {} with error {}".format(
                        subset_path, err
                    )
                )
                return

        return class_names, class_parent, subset_ids

    def tensor_normalize(self, tensor, mean, std, func=None):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize.
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        if func is not None:
            tensor = func(tensor)
        tensor = tensor - mean
        tensor = tensor / std
        return tensor

    def pack_pathway_output(self, cfg, frames):
        """
        Prepare output as a list of tensors. Each tensor corresponding to a
        unique pathway.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frame_list (list): list of tensors with the dimension of
                `channel` x `num frames` x `height` x `width`.
        """
        if cfg.DATA.REVERSE_INPUT_CHANNEL:
            frames = frames[[2, 1, 0], :, :, :]
        if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
            frame_list = [frames]
        elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
        else:
            raise NotImplementedError(
                "Model arch {} is not in {}".format(
                    cfg.MODEL.ARCH,
                    cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
                )
            )
        return frame_list

    def predict(self, frames, bboxes):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        if bboxes is not None:
            bboxes = torch.from_numpy(np.array(bboxes)).cuda(device=torch.device(self.gpu_id))
            bboxes = cv2_transform.scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                bboxes,
                frames[0].shape[0],
                frames[0].shape[1],
            )
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]

        frames = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
            for frame in frames
        ]
        inputs = process_cv2_inputs(frames, self.cfg)
        if bboxes is not None:
            index_pad = torch.full(
                size=(bboxes.shape[0], 1),
                fill_value=float(0),
                device=bboxes.device,
            )

            # Pad frame index for each box.
            bboxes = torch.cat([index_pad, bboxes], axis=1)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )

        preds = self.model(inputs, bboxes)

        top_scores, top_classes = torch.topk(preds, k=1)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.detach().cpu()
                top_classes = top_classes.detach().cpu()

        preds = preds.detach()

        return top_classes.numpy().tolist()



if __name__ == "__main__":

    bboxes = [[614.9788,  70.8798, 699.1706, 315.4135],
            [442.4589, 155.1308, 622.3549, 471.4146],
            [718.3026,  10.5990, 764.2662, 175.1927],
            [855.9326,  66.6147, 920.4806, 319.1182],
            [793.0071,  61.0188, 875.1038, 271.1492],
            [762.9031,  55.4653, 833.8232, 229.9462]]

    frames = [numpy.load('frames.npy')[i] for i in range(64)]
    


    slowfast = AVAVisualizerWithPrecomputedBox()

    ### frames: [img_frame0, img_frame1, img_frame2, ..., img_frame63]
    ### bboxes: 从第32帧执行yolo检测到的所有行人的框，检测时可以适当条件置信度，比如0.8
    slowfast_pred = slowfast.predict(frames, bboxes) 


    print(slowfast_pred)
    """
    [[11],
    [ 9],
    [11],
    [13],
    [13],
    [11]]
    """
