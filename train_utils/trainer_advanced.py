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

    def __init__(
            self,
            model,
            optimizer,
            epoch: int,
            train_iter=None,
            val_iter=None,
            test_iter=None,
            snapshot_dir=None,
            verbose: bool=True,
            timer: bool=False,
            log_interval = 20,
    ) -> None:

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.model = model
        self.criterion = torch.nn.Sequential()
        self.optimizer = optimizer
        self.total_epoch: int = epoch
        self.snapshot_dir: pathlib.Path = pathlib.Path(snapshot_dir).resolve()
        self.save_and_load = bool(snapshot_dir is not None and val_iter is not None)
        self.verbose: bool = verbose
        self.use_timer: bool = timer
        self.log_interval: int = log_interval
        self.save_and_load: bool = bool(snapshot_dir is not None and val_iter is not None)

        self.step_task = None
        self.step_task_mode = None

        super().__init__()

        # Do not set attribute of instance after super().__init__()
        print("Advanced Fitter Initialized.")

    __setattr__ = object.__setattr__
    __delattr__ = object.__delattr__

    def _train(self):
        self._require_context()

        self.model.train()

        verbose = self.verbose
        log_interval = self.log_interval

        total_loss = total_accuracy = 0.
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

                    l, a = self._train_classification(*data)
                    total_loss += l; total_accuracy += a; total_batch += 1
                    if iteration % log_interval == 0 and verbose:
                        self._log_train_doing(l, iteration, whole)

        avg_loss = total_loss / total_batch if total_batch else -1
        avg_accuracy = total_accuracy / total_batch if total_batch else -1

        self._log_train_done(avg_loss, avg_accuracy)

        det_avg_loss = det_loss / det_batch if det_batch else -1

        self._log_train_done(det_avg_loss)

        return avg_loss, avg_accuracy

    def _train_classification(self, images, targets):

        images = self._to_apply_tensor(images).float()
        prediction = self.model(images)[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets)

        with torch.no_grad():
            l = loss.item()
            a = torch.eq(torch.argmax(prediction, 1), targets).float().mean().item()

        loss.backward()

        optimizer = self.optimizer
        optimizer.step()
        optimizer.zero_grad()

        return l, a

    def _train_detection(self, dataset):

        images = dataset['image']
        labels = dataset['labels']
        boxes = dataset['boxes']

        images = self._to_apply_tensor(images).float()
        boxes = self._to_apply_tensor(boxes).float()
        labels = self._to_apply_tensor(labels).to(torch.int64)

        targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]

        loss_dict = self.model(images, targets)

        # ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
        loss = sum(loss_dict.values())
        loss.backward()

        optimizer = self.optimizer
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    @torch.no_grad()
    def _evaluate(self, *, test=False):

        self.model.eval()

        datasets = self.test_iter if test else self.val_iter

        total_loss = total_accuracy = 0.
        total_batch = 0
        det_loss = 0.
        det_batch = 0

        for datum in datasets:

            for data in datum:

                if isinstance(data, dict):
                    with self._force_train_mode():
                        l = self._eval_detection(data)
                        det_loss += l; det_batch += 1

                else:
                    l, a = self._eval_classification(*data)
                    total_loss += l; total_accuracy += a; total_batch += 1

        avg_loss = total_loss / total_batch if total_batch else -1
        avg_accuracy = total_accuracy / total_batch if total_batch else -1

        self._log_eval(avg_loss, avg_accuracy, test=test)

        det_avg_loss = det_loss / det_batch if det_batch else -1

        self._log_eval(det_avg_loss, test=test)

        return avg_loss, avg_accuracy

    def _eval_classification(self, images, targets):

        images = self._to_apply_tensor(images).float()
        prediction = self.model(images)[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets)

        l = loss.item()
        a = torch.eq(torch.argmax(prediction, 1), targets).float().mean().item()

        return l, a

    def _eval_detection(self, dataset):

        images = dataset['image']
        labels = dataset['labels']
        boxes = dataset['boxes']

        images = self._to_apply_tensor(images).float()
        boxes = self._to_apply_tensor(boxes).float()
        labels = self._to_apply_tensor(labels).to(torch.int64)

        targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]

        loss_dict = self.model(images, targets)

        # ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
        loss = sum(loss_dict.values())
        return loss.item()

    @contextlib.contextmanager
    def _force_train_mode(self):
        prev_mode = self.model.training
        try:
            self.model.train()
            yield
        finally:
            self.model.train(mode=prev_mode)
