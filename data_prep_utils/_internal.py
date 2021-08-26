import typing
from pathlib import Path
import os


__root: typing.Optional[Path] = None
__initialized: bool = False
__hooks: tuple = ()


def _register_init_hook(func: typing.Callable[..., None]) -> None:
    global __hooks
    if func not in __hooks:
        __hooks = __hooks + (func, )


def lazy_init() -> None:
    global __initialized
    if __initialized is not False:  # True, None
        return
    try:
        __initialized = None  # passing during initialization prevents RecursionError
        if __root is None:
            set_root()
        elif not isinstance(__root, Path):
            raise TypeError("Root path must have set in undesirable way. Please use set_root() function.")
        if not __root.is_dir():
            raise ValueError("Root directory not exists: %s" % __root)
        for hook in __hooks:
            hook()
        __initialized = True
    except Exception:
        __initialized = False
        raise


def set_root(path: typing.Union[str, os.PathLike] = 'data') -> None:
    global __root
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("Bad Path Argument: %s" % path)
    __root = Path(path).resolve()
    lazy_init()


def get_root() -> Path:
    lazy_init()
    return __root
