
requirements = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'pydicom': 'pydicom',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}


def check(raise_exception=False):
    import importlib
    fallback = []
    for key in requirements:
        try:
            importlib.import_module(key)
        except ImportError:
            fallback.append(requirements[key])
            continue
    if fallback:
        exc = ImportError("Please install these libraries by pip: %s" % ", ".join(fallback))
        if raise_exception:
            raise exc
        try:
            from pip._internal.cli.main import main
            for value in fallback:
                main(['pip', 'install', value])
            check(raise_exception=True)
        except Exception as e:
            raise e from exc
    print("All required libraries are installed.")


if __name__ == '__main__':
    check()
