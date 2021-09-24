
requirements = {

    'torch': 'torch',
    'torchvision': 'torchvision',
    'albumentations': 'albumentations',
    'efficientnet_pytorch': 'efficientnet_pytorch',

    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',

    'PIL': 'pillow',
    'pydicom': 'pydicom',
    'torchinfo': 'torchinfo',

    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',

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
            raise exc from e
    print("All required libraries are installed.")


if __name__ == '__main__':
    check()
