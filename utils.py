import hashlib
import os
import platform

import gdown


def get_platform_path():
    system = platform.system()
    data_dir, model_dir, checkpoint_dir, dirs = '', '', '', []
    if system == 'Windows':
        drive, common_dir = 'F', 'cache'
        data_dir = '{}:\\{}\\data'.format(drive, common_dir)
        model_dir = '{}:\\{}\\model'.format(drive, common_dir)
        checkpoint_dir = '{}:\\{}\\checkpoint'.format(drive, common_dir)
        dirs = [data_dir, model_dir, checkpoint_dir]

    elif system == 'Linux':
        common_dir = '/data'
        data_dir = '{}/data'.format(common_dir)
        model_dir = '{}/model'.format(common_dir)
        checkpoint_dir = '{}/checkpoint'.format(common_dir)
        dirs = [data_dir, model_dir, checkpoint_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    return data_dir, model_dir, checkpoint_dir


def _md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            hash.update(block)
    return hash.hexdigest()


def cached_download(url, path, md5=None, quiet=False, postprocess=None):
    def check_md5(path, md5):
        print('[{:s}] Checking md5 ({:s})'.format(path, md5))
        return _md5sum(path) == md5

    if os.path.exists(path) and not md5:
        print('[{:s}] File exists ({:s})'.format(path, _md5sum(path)))
    elif os.path.exists(path) and md5 and check_md5(path, md5):
        pass
    else:
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        gdown.download(url, path, quiet=quiet)

    if postprocess is not None:
        postprocess(path)

    return path
