"""Even with: https://github.com/kdha0727/easyrun-pytorch/blob/main/easyrun.py"""

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


# Typing Variables

import typing
from os import PathLike
_t = typing.TypeVar('_t')
_v = typing.TypeVar('_v')
_model_type = typing.Union[
    typing.Mapping[str, torch.nn.Module],
    torch.nn.Module,
]
_optim_type = typing.Union[
    typing.Sequence[torch.optim.Optimizer],
    torch.optim.Optimizer
]
_loss_type = typing.Union[
    torch.nn.Module,
    typing.Callable[[_v, _v], _v],
    str,
]
_data_type = typing.Optional[
    typing.Union[
        torch.utils.data.Dataset,
        torch.utils.data.DataLoader,
    ]
]
_step_func_type = typing.Optional[
    typing.Union[
        typing.Callable[[], None],
        typing.Callable[[int], None],
        typing.Callable[[int, typing.List], None],
    ]
]
_path_type = typing.Optional[
    typing.Union[
        PathLike,
        str
    ]
]
_to_parse_type = typing.Tuple[
    typing.Optional[torch.device],
    typing.Optional[torch.dtype],
    bool,
    typing.Optional[torch.memory_format]
]


#
# Multi Object Train Helper Class
#

class MultipleObjectHandlerBase(collections.OrderedDict):

    def __init__(self, other=(), baseclass=None, **kw):
        super(MultipleObjectHandlerBase, self).__init__()
        if isinstance(other, collections.Mapping):
            if baseclass is not None:
                for key, value in other.items():
                    assert isinstance(key, str), "Mapping key must be string, got %s" % type(key).__name__
                    assert isinstance(value, baseclass), "Invalid mapping value object: %s" % value
            self.update(other)
        else:
            iterable = iter(other)
            for idx, value in enumerate(iterable):
                if baseclass is not None:
                    assert isinstance(value, baseclass), "Invalid sequence object: %s" % value
                self[idx] = value
        if baseclass is not None:
            for value in kw.values():
                assert isinstance(value, baseclass), "Invalid keyword argument: %s" % value
        self.update(kw)
        if not self:
            raise ValueError("Empty handler not allowed")

    def _format(self, indent=0):
        format_string = self.__class__.__name__ + ' ('
        for key in self:
            format_string += '\n{0}({1}): '.format(' ' * indent, key)
            value = repr(self[key])
            if not value:
                continue
            first, *rest = value.splitlines()
            format_string += first
            for line in rest:
                format_string += '\n{0}{1}'.format(' ' * indent, line)
        format_string += '\n)'
        return format_string

    @reprlib.recursive_repr()
    def __repr__(self):
        return self._format()

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for k, v in self:
            state_dict[k] = v.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in self:
            v.load_state_dict(state_dict[k])


class MultipleOptimizerHandler(MultipleObjectHandlerBase):

    @typing.overload
    def __new__(cls, other: torch.optim.Optimizer = ...) -> torch.optim.Optimizer: ...

    @typing.overload
    def __new__(cls: _t, other: typing.Mapping = ..., **kw) -> _t: ...

    @typing.overload
    def __new__(cls: _t, other: typing.Iterable = ...) -> _t: ...

    def __new__(cls, other=(), **kw):
        if isinstance(other, torch.optim.Optimizer):
            return other
        return super(MultipleOptimizerHandler, cls).__new__(cls)

    def __init__(self, other=(), **kw):
        super(MultipleOptimizerHandler, self).__init__(other, torch.optim.Optimizer, **kw)

    def __repr__(self):
        return self._format(4)

    def zero_grad(self):
        for opt in self:
            self[opt].zero_grad()

    def step(self):
        for opt in self:
            self[opt].step()


#
# One-time Trainer Class
#

class Trainer(object):
    """

    One-time Pytorch Trainer Utility:
        written by Dong Ha Kim, in Yonsei University.

    Available Parameters:
        :param model: (torch.nn.Module) model object, or list of model objects to use.
        :param optimizer: (torch.optim.Optimizer) optimizer, or list of optimizers.
        :param criterion: (torch.nn.Module) loss function or model object.
            You can also provide string name.
        :param epoch: (int) total epochs.
        :param train_iter: train data loader, or train dataset.
        :param val_iter: validation data loader, or validation dataset.
        :param test_iter: test data loader, or test dataset.
        :param unsupervised: (bool) flag that indicates if task is unsupervised learning or not.
            if true, only prediction will be given as input of criterion.
            otherwise, two arguments (prediction, ground_truth) will be given.
            default is false.
        :param eval_accuracy: (bool) if false, do not calculate accuracy for evaluation metric.
        :param step_task: (callable) task to be run in each epoch.
            no input, or (current_epoch, ), or (current_epoch, current_train_result_list)
            can be given as function input.
            partial is recommended to implement this.
        :param snapshot_dir: (str) provide if you want to use parameter saving and loading.
            in this path name, model's weight parameter at best(least) loss will be temporarily saved.
        :param verbose: (bool) verbosity. with turning it on, you can view learning logs.
            default value is True.
        :param timer: (bool) provide with verbosity, if you want to use time-checking.
            default value is True.
        :param log_interval: (int) provide with verbosity, if you want to set your log interval.
            default value is 20.

    Available Methods:
        (): [call] repeat training and validating for all epochs, followed by testing.
        to(device): apply to(device) in model, criterion, and all tensors.
        train(): run training one time with train dataset.
            returns (train_loss, train_accuracy).
        evaluate(): run validating one time with validation dataset.
            returns (val_loss, val_accuracy).
        step(): run training, followed by validating one time.
            returns (train_loss, train_accuracy, val_loss, val_accuracy).
        run(): repeat training and validating for all epochs.
            returns train result list, which contains each epoch`s
            (train_loss, train_accuracy, val_loss, val_accuracy).
        test(): run testing one time with test dataset.
            returns (test_loss, test_accuracy).
        state_dict(): returns state dictionary of trainer class.
        load_state_dict(): loads state dictionary of trainer class.

    """

    #
    # Constructor
    #

    __initialized: bool = False

    def __init__(self, *args, **kwargs) -> None:

        if args or kwargs:
            self.__real_init(*args, **kwargs)

        self._closed: bool = False
        self._current_epoch: int = 0
        self._best_loss: float = math.inf
        self._time_start: typing.Optional[float] = None
        self._time_stop: typing.Optional[float] = None
        self._processing_fn: _path_type = None
        self._current_run_result: typing.Optional[typing.List] = None

        self.__to_parse: _to_parse_type = (None, None, False, None)

        self.__initialized = True

    def __real_init(
            self,
            model: _model_type,
            criterion: _loss_type,
            optimizer: _optim_type,
            epoch: int,
            train_iter: _data_type = None,
            val_iter: _data_type = None,
            test_iter: _data_type = None,
            step_task: _step_func_type = None,
            snapshot_dir: _path_type = None,
            verbose: bool = True,
            timer: bool = False,
            log_interval: typing.Optional[int] = 20,
    ) -> None:

        _dataset_type = (torch.utils.data.Dataset, torch.utils.data.DataLoader)
        assert train_iter is None or isinstance(train_iter, _dataset_type), \
            "Invalid train_iter type: %s" % train_iter.__class__.__name__
        assert val_iter is None or isinstance(val_iter, _dataset_type), \
            "Invalid val_iter type: %s" % val_iter.__class__.__name__
        assert test_iter is None or isinstance(test_iter, _dataset_type), \
            "Invalid test_iter type: %s" % test_iter.__class__.__name__
        assert isinstance(epoch, int) and epoch > 0, \
            "Epoch is expected to be positive int, got %s" % epoch
        assert isinstance(log_interval, int) and log_interval > 0, \
            "Log Interval is expected to be positive int, got %s" % log_interval

        if step_task:
            assert callable(step_task), \
                "Step Task function is expected to be callable, got %s" % step_task
            try:
                sig = inspect.signature(step_task)
            except ValueError as exc:
                raise TypeError("Invalid Step Task function: %s" % step_task) from exc
            step_task_mode = len(sig.parameters)
            assert step_task_mode in range(3), \
                "Step Task function`s argument length should be 0, 1, or 2."
        else:
            step_task_mode = None

        if not callable(criterion):
            assert isinstance(criterion, str), \
                "Invalid criterion type: %s" % criterion.__class__.__name__
            assert (hasattr(torch.nn, criterion) or hasattr(torch.nn.functional, criterion)), \
                "Invalid criterion string: %s" % criterion
            criterion = getattr(torch.nn.functional, criterion, getattr(torch.nn, criterion)())

        self.model: _model_type = model
        self.criterion: _loss_type = criterion
        self.optimizer: _optim_type = MultipleOptimizerHandler(optimizer)
        self.total_epoch: int = epoch
        self.train_iter: _data_type = train_iter
        self.val_iter: _data_type = val_iter
        self.test_iter: _data_type = test_iter
        self.step_task: _step_func_type = step_task
        self.step_task_mode: typing.Optional[int] = step_task_mode
        self.snapshot_dir: pathlib.Path = pathlib.Path(snapshot_dir).resolve()
        self.verbose: bool = verbose
        self.use_timer: bool = timer
        self.log_interval: int = log_interval
        self.train_batch_size: int = train_iter.batch_size
        self.train_loader_length: int = len(train_iter)
        self.train_dataset_length: int = len(getattr(train_iter, 'dataset', train_iter))
        self.save_and_load: bool = bool(snapshot_dir is not None and val_iter is not None)

    #
    # De-constructor: executed in buffer-cleaning in python exit
    #

    def __del__(self) -> None:
        self._close()

    #
    # Context manager magic methods
    #

    def __enter__(self: _t) -> _t:
        self._open()
        return self

    def __exit__(self, exc_info: ..., exc_class: ..., exc_traceback: ...) -> None:
        try:
            self._close()
        except Exception as exc:
            if (exc_info or exc_class or exc_traceback) is not None:
                pass  # executed in exception handling - just let python raise that exception
            else:
                raise exc

    #
    # Attribute magic methods
    #

    def __setattr__(self, key: str, value: typing.Any):
        if not key.startswith('_') and self.__initialized:
            raise AttributeError('Cannot set attributes after initialized.')
        object.__setattr__(self, key, value)

    def __delattr__(self, key: str):
        if not key.startswith('_') and self.__initialized:
            raise AttributeError('Cannot set attributes after initialized.')
        object.__delattr__(self, key)

    #
    # Call implement: run training, evaluating, followed by testing
    #

    def __call__(self: _t) -> _t:
        with self._with_context():
            self.run()
            if self.test_iter is not None:
                self.test()
            return self

    #
    # Running Methods
    #

    def train(self) -> typing.Sequence[float]:

        result = self._train()
        self._current_epoch += 1
        return result

    def evaluate(self) -> typing.Sequence[float]:

        return self._evaluate(test=False)

    def test(self) -> typing.Sequence[float]:

        return self._evaluate(test=True)

    def step(self) -> typing.Sequence[float]:

        self._log_step(self._current_epoch + 1)

        train_args = self._train()
        self._save()

        if self.val_iter:
            test_loss, *test_args = self._evaluate(test=False)

            # Save the model having the smallest validation loss
            if test_loss < self._best_loss:
                self._best_loss = test_loss
                self._save(self.snapshot_dir / f'best_checkpoint_epoch_{str(self._current_epoch + 1).zfill(3)}.pt')
                for path in sorted(glob.glob(self.snapshot_dir / 'best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

        else:
            test_loss, test_args = None, ()

        self._current_epoch += 1

        self._do_step_task()

        return tuple((*train_args, test_loss, *test_args))

    def run(self) -> typing.List[typing.Sequence[float]]:

        with self._with_context():

            self._current_run_result = result = []
            self._current_epoch = 0
            self._log_start()
            self._timer_start()

            try:
                while self._current_epoch < self.total_epoch:
                    result.append(self.step())

            finally:
                self._timer_stop()
                self._log_stop()
                self._current_run_result = None

                if self.save_and_load and self._current_epoch:
                    self._load()

            return result

    #
    # State dictionary handler: used in saving and loading parameters
    #

    def state_dict(self) -> collections.OrderedDict:

        state_dict = collections.OrderedDict()
        state_dict['epoch'] = self._current_epoch
        state_dict['best_loss'] = self._best_loss
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        if isinstance(self.criterion, torch.nn.Module):
            state_dict['criterion'] = self.criterion.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: collections.OrderedDict) -> None:

        self._current_epoch = state_dict['epoch']
        self._best_loss = state_dict['best_loss']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.load_state_dict(state_dict['criterion'])

    #
    # Device-moving Methods
    #

    def to(self: _t, *args, **kwargs) -> _t:  # overwrite this in subclass, for further features

        self._to_set(*args, **kwargs)
        self._to_apply_inner(self.model)
        if isinstance(self.criterion, torch.nn.Module):
            self._to_apply_inner(self.criterion)
        return self

    # Internal Device-moving Methods

    def _to_set(self, *args, **kwargs) -> None:

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)  # noqa
        device = device or self.__to_parse[0]
        dtype = dtype or self.__to_parse[1]
        non_blocking = non_blocking or self.__to_parse[2]
        convert_to_format = convert_to_format or self.__to_parse[3]
        self.__to_parse = (device, dtype, non_blocking, convert_to_format)

    def _to_apply_tensor(self, v: _v) -> _v:

        device, dtype, _, convert_to_format = self.__to_parse
        return v.to(device, dtype, memory_format=convert_to_format)

    def _to_apply_inner(self, v: _v) -> _v:

        device, dtype, non_blocking, convert_to_format = self.__to_parse
        return v.to(device, dtype, non_blocking, memory_format=convert_to_format)

    def _to_apply_multi_tensor(self, *v: _v) -> typing.Sequence[_v]:

        return tuple(map(self._to_apply_tensor, v))

    # Internal Timing Functions

    def _timer_start(self) -> None:

        self._require_context()

        if self.use_timer:
            self._time_start = time.time()

    def _timer_stop(self) -> None:

        self._require_context()

        if self.use_timer:
            self._time_stop = time.time()

    # Internal Logging Methods

    def _log_start(self) -> None:

        if self.verbose:
            self.log_function(f"\n<Start Learning> \t\t\t\tTotal {self.total_epoch} epochs")

    def _log_step(self, epoch: int) -> None:

        if self.verbose:
            self.log_function(f'\nEpoch {epoch}')

    def _log_train_doing(self, loss: float, iteration: int, whole: int = None) -> None:

        if self.verbose:
            if whole is not None:
                self.log_function(
                    f'\r[Train]\t '
                    f'Progress: {iteration}/{whole} '
                    f'({100. * iteration / whole:05.2f}%), \tLoss: {loss:.6f}',
                    end=' '
                )
            else:
                self.log_function(
                    f'\r[Train]\t '
                    f'Progress: {iteration * self.train_batch_size}/{self.train_dataset_length} '
                    f'({100. * iteration / self.train_loader_length:05.2f}%), \tLoss: {loss:.6f}',
                    end=' '
                )

    def _log_train_done(self, loss: float, accuracy: typing.Optional[float] = None) -> None:

        if self.verbose:
            log = '\r[Train]\t '
            log += f'Average loss: {loss:.5f}, '
            if accuracy is not None:
                log += f'\t\tTotal accuracy: {100. * accuracy:05.2f}% '
            # f'\r[Train]\t '
            # f'Progress: {self.train_dataset_length}/{self.train_dataset_length} (100.00%), '
            # f'\tTotal accuracy: {100. * accuracy:.2f}%'
            self.log_function(log)

    def _log_eval(self, loss: float, accuracy: typing.Optional[float] = None, test: typing.Optional[bool] = False) -> None:

        if self.verbose:
            log = '\n[Test]\t ' if test else '[Eval]\t '
            log += f'Average loss: {loss:.5f}, '
            if accuracy is not None:
                log += f'\t\tTotal accuracy: {100. * accuracy:05.2f}% '
            self.log_function(log)

    def _log_stop(self) -> None:

        if self.verbose:
            log = "\n<Stop Learning> "
            if self.save_and_load:
                log += f"\tLeast loss: {self._best_loss:.4f}"
            if self.use_timer:
                log += "\tDuration: "
                duration = self._time_stop - self._time_start
                if duration > 60:
                    log += f"{int(duration // 60):02}m {duration % 60:05.2f}s"
                else:
                    log += f"{duration:05.2f}s"
            self.log_function(log)

    # Internal Parameter Methods

    def _load(self, fn=None) -> None:

        self._require_context()

        if self.save_and_load:
            self.load_state_dict(torch.load(str(fn) or self._processing_fn))

    def _save(self, fn=None) -> None:

        self._require_context()

        if self.save_and_load:
            torch.save(self.state_dict(), str(fn) or self._processing_fn)

    # Internal Context Methods

    def _open(self) -> bool:

        prev = self._closed

        if prev:
            if self.save_and_load:
                self._processing_fn = str(self.snapshot_dir / f'_processing_{id(self)}.pt')
                self.snapshot_dir.mkdir(exist_ok=True)

        self._closed = False

        return prev

    def _close(self, prev: bool = True) -> None:

        if self._closed:
            return

        if prev:
            if self.save_and_load:
                try:
                    if self._processing_fn is not None:
                        os.remove(self._processing_fn)
                except FileNotFoundError:
                    pass
                self._processing_fn = None

        self._closed = prev

    def _require_context(self) -> None:

        if self._closed:
            raise ValueError('Already closed: %r' % self)

    @contextlib.contextmanager
    def _with_context(self):

        prev = True
        try:
            prev = self._open()
            yield
        finally:
            self._close(prev)

    # Internal Running Methods

    def _train(self) -> typing.Sequence[float]:

        self._require_context()

        data = self.train_iter
        total_loss, total_accuracy = 0., 0.
        total_batch = len(data)
        verbose = self.verbose
        log_interval = self.log_interval

        self.model.train()

        for iteration, (x, y) in enumerate(self.train_iter, 1):
            x, y = self._to_apply_multi_tensor(x, y)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                total_loss += loss.item()
                total_accuracy += torch.eq(torch.argmax(prediction, 1), y).float().mean().item()

            if iteration % log_interval == 0 and verbose:
                self._log_train_doing(loss.item(), iteration)

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch

        self._log_train_done(avg_loss, avg_accuracy)

        return avg_loss, avg_accuracy

    @torch.no_grad()
    def _evaluate(self, *, test: typing.Optional[bool] = False) -> typing.Sequence[float]:

        data = self.test_iter if test else self.val_iter
        assert data is not None, "You must provide dataset for evaluating method."
        total_loss, total_accuracy = 0., 0.
        total_batch = len(data)

        self.model.eval()

        for x, y in data:
            x, y = self._to_apply_multi_tensor(x, y)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)

            total_loss += loss.item()
            total_accuracy += torch.eq(torch.argmax(prediction, 1), y).float().mean().item()

        avg_loss = total_loss / total_batch
        avg_accuracy = total_accuracy / total_batch

        self._log_eval(avg_loss, avg_accuracy, test)

        return avg_loss, avg_accuracy

    def _do_step_task(self) -> None:

        self._require_context()

        if self.step_task:
            args = (self._current_epoch, self._current_run_result)[:self.step_task_mode]
            self.step_task(*args)

    # Log function: overwrite this to use custom logging hook

    log_function = staticmethod(print)


# Clear typing variables from namespace

del typing, PathLike, _t, _v, _step_func_type, _loss_type, _data_type, _path_type, _model_type, _optim_type
