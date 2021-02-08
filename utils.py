import hashlib
import os
import platform

import gdown
import sys
import time

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


def get_platform_path():
    system = platform.system()
    data_dir, model_dir, checkpoint_dir, dirs = '', '', '', []
    if system == 'Windows':
        drive, common_dir = 'F', 'cache'
        data_dir = '{}:/{}/data'.format(drive, common_dir)
        model_dir = '{}:/{}/model'.format(drive, common_dir)
        checkpoint_dir = '{}:/{}/checkpoint'.format(drive, common_dir)
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


def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpeg', '.jpg', '.bmp'])
