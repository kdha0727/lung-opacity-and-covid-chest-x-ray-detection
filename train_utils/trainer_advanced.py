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
            train_iter = None,
            val_iter = None,
            test_iter = None,
            snapshot_dir = None,
            verbose: bool = True,
            timer: bool = False,
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
        self.verbose: bool = verbose
        self.use_timer: bool = timer
        self.log_interval: int = log_interval
        self.save_and_load: bool = bool(snapshot_dir is not None and val_iter is not None)

        # FIXME
        super().__init__()

        # Do not set attribute of instance.
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

        for data in datasets:

            whole = len(data)
            for iteration, (images, targets) in enumerate(data, 1):

                if isinstance(targets, dict):
                    l = self._train_detection(images, targets)
                    det_loss += l; det_batch += 1
                    if iteration % log_interval == 0 and verbose:
                        self._log_train_doing(l, iteration, whole)

                else:
                    l, a = self._train_classification(images, targets)
                    total_loss += l; total_accuracy += a; total_batch += 1
                    if iteration % log_interval == 0 and verbose:
                        self._log_train_doing(l, iteration, whole)

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch

        self._log_train_done(avg_loss, avg_accuracy)

        det_avg_loss = det_loss / det_batch

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

    def _train_detection(self, images, targets):

        print(targets)

        images = self._to_apply_tensor(images).double()
        boxes = self._to_apply_tensor(targets['boxes']).double()
        labels = self._to_apply_tensor(targets['labels']).to(torch.int64)

        images = torch.stack([images])
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

        for data in datasets:

            for images, targets in data:

                if isinstance(targets, dict):
                    l = self._eval_detection(images, targets)
                    det_loss += l; det_batch += 1

                else:
                    l, a = self._eval_classification(images, targets)
                    total_loss += l; total_accuracy += a; total_batch += 1

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch

        self._log_eval(avg_loss, avg_accuracy, test=test)

        det_avg_loss = det_loss / det_batch

        self._log_eval(det_avg_loss, test=test)

        return avg_loss, avg_accuracy

    def _eval_classification(self, images, targets):

        images = self._to_apply_tensor(images).float()
        prediction = self.model(images)[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets)

        l = loss.item()
        a = torch.eq(torch.argmax(prediction, 1), targets).float().mean().item()

        return l, a

    def _eval_detection(self, images, targets):

        images = self._to_apply_tensor(images).float()
        boxes = [self._to_apply_tensor(target['boxes']).float() for target in targets]
        labels = [self._to_apply_tensor(target['labels']).float() for target in targets]

        loss, _, _ = self.criterion(images, boxes, labels)

        return loss.item()
