from .trainer_base import Trainer, MultipleOptimizerHandler


import os

import collections
import contextlib
import inspect
import math
import reprlib
import pathlib
import time
import glob

import torch
import torch.nn.functional
import torch.utils.data


class AdvancedFitter(Trainer):

    criterion: None

    def __init__(
            self,
            model_c,
            model_d,
            optimizer_c,
            optimizer_d,
            epoch: int,
            train_iter=None,
            val_iter=None,
            test_iter=None,
            snapshot_dir=None,
            verbose: bool = True,
            timer: bool = False,
            log_interval=20,
    ) -> None:

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.model_c = model_c
        self.model_d = model_d
        self.optimizer_c = optimizer_c
        self.optimizer_d = optimizer_d
        self.total_epoch: int = epoch
        self.snapshot_dir: pathlib.Path = pathlib.Path(snapshot_dir).resolve()
        self.save_and_load = bool(snapshot_dir is not None and val_iter is not None)
        self.verbose: bool = verbose
        self.use_timer: bool = timer
        self.log_interval: int = log_interval
        self.save_and_load: bool = bool(snapshot_dir is not None and val_iter is not None)

        self.criterion = None
        self.step_task = None
        self.step_task_mode = None

        super().__init__()

        # Do not set attribute of instance after super().__init__()
        print("Advanced Fitter Initialized.")

    __setattr__ = object.__setattr__
    __delattr__ = object.__delattr__

    def _train(self):
        self._require_context()

        self.model_c.train()
        self.model_d.train()

        verbose = self.verbose
        log_interval = self.log_interval

        total_loss = 0.
        total_batch = 0
        det_loss = 0.
        det_batch = 0

        datasets = self.train_iter

        for datum in datasets:

            whole = len(datum)
            for iteration, data in enumerate(datum, 1):

                if isinstance(data, dict):
                    l = self._train_detection(data)
                    det_loss += l; det_batch += 1
                    if iteration % log_interval == 0 and verbose:
                        self._log_train_doing(l, iteration, whole)

                else:

                    l = self._train_classification(data)
                    total_loss += l; total_batch += 1
                    if iteration % log_interval == 0 and verbose:
                        self._log_train_doing(l, iteration, whole)

        avg_loss = total_loss / total_batch if total_batch else -1

        self._log_train_done(avg_loss)

        det_avg_loss = det_loss / det_batch if det_batch else -1

        self._log_train_done(det_avg_loss)

        return avg_loss, None

    def _train_classification(self, dataset):

        image, label = self._to_apply_multi_tensor(*dataset)
        image = image.float()

        logit = self.model_c(image)
        loss = torch.nn.functional.cross_entropy(logit, label)

        loss.backward()

        # images = self._to_apply_tensor(images).float()
        # prediction = self.model(images)[0]
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets)

        optimizer = self.optimizer_c
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    def _train_detection(self, dataset):

        loss_dict = self.model_d(*self._convert_detection_dataset_to_tensor(dataset))

        # ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
        loss = sum(loss_dict.values())
        loss.backward()

        optimizer = self.optimizer_d
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    @torch.no_grad()
    def _evaluate(self, *, test=False):

        self.model_c.eval()
        self.model_d.eval()

        datasets = self.test_iter if test else self.val_iter

        total_loss = total_accuracy = 0.
        total_batch = 0
        det_loss = 0.
        det_batch = 0

        for datum in datasets:

            for data in datum:

                if isinstance(data, dict):
                    with self._force_train_mode_d():
                        l = self._eval_detection(data)
                        det_loss += l; det_batch += 1

                else:
                    l = self._eval_classification(data)
                    total_loss += l; total_batch += 1

        avg_loss = total_loss / total_batch if total_batch else -1
        avg_accuracy = total_accuracy / total_batch if total_batch else -1

        self._log_eval(avg_loss, avg_accuracy, test=test)

        det_avg_loss = det_loss / det_batch if det_batch else -1

        self._log_eval(det_avg_loss, test=test)

        return max(det_avg_loss, 0) + max(avg_loss, 0),  det_avg_loss, avg_loss, avg_accuracy

    def _eval_classification(self, dataset):

        image, label = self._to_apply_multi_tensor(*dataset)
        image = image.float()

        logit = self.model_c(image)
        loss = torch.nn.functional.cross_entropy(logit, label)

        # images = self._to_apply_tensor(images).float()
        # prediction = self.model(images)[0]
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets)

        # l = loss.item()
        # a = torch.eq(torch.argmax(prediction, 1), targets).float().mean().item()

#         # loss_dict = self.model_c(*self._convert_classification_dataset_to_tensor(dataset))
#         #
#         # loss = loss_dict['loss_classifier']
        return loss.item()

    def _eval_detection(self, dataset):

        loss_dict = self.model_d(*self._convert_detection_dataset_to_tensor(dataset))

        # ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
        loss = sum(loss_dict.values())
        return loss.item()

    def _convert_detection_dataset_to_tensor(self, dataset: dict):

        images = self._to_apply_tensor(dataset['image']).float()
        boxes = self._to_apply_tensor(dataset['boxes']).float()
        labels = self._to_apply_tensor(dataset['labels']).to(torch.int64)
        targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]

        return images, targets

    def _convert_classification_dataset_to_tensor(self, dataset: tuple):

        images, labels = dataset

        images = self._to_apply_tensor(images).float()
        labels = self._to_apply_tensor(labels).to(torch.int64)

        if labels.ndim == 1:
            boxes = torch.zeros(0, 4)
        else:
            boxes = torch.zeros(labels.size(0), 0, 4)
        boxes = self._to_apply_tensor(boxes).float()

        images = list(images)
        targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]

        return images, targets

    @contextlib.contextmanager
    def _force_train_mode_d(self):
        prev_mode = self.model_d.training
        try:
            self.model_d.train()
            yield
        finally:
            self.model_d.train(mode=prev_mode)

    def to(self, *args, **kwargs):  # overwrite this in subclass, for further features

        self._to_set(*args, **kwargs)
        self._to_apply_inner(self.model_c)
        self._to_apply_inner(self.model_d)
        return self

    def state_dict(self) -> collections.OrderedDict:

        state_dict = collections.OrderedDict()
        state_dict['epoch'] = self._current_epoch
        state_dict['best_loss'] = self._best_loss
        state_dict['model_c'] = self.model_c.state_dict()
        state_dict['model_d'] = self.model_d.state_dict()
        state_dict['optimizer_c'] = self.optimizer_c.state_dict()
        state_dict['optimizer_d'] = self.optimizer_d.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: collections.OrderedDict) -> None:

        self._current_epoch = state_dict['epoch']
        self._best_loss = state_dict['best_loss']
        self.model_c.load_state_dict(state_dict['model_c'])
        self.model_d.load_state_dict(state_dict['model_d'])
        self.optimizer_c.load_state_dict(state_dict['optimizer_c'])
        self.optimizer_d.load_state_dict(state_dict['optimizer_d'])
