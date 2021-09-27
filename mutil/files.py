import errno
import os
import shutil
from mutil.str_format import get_time_string


def listdir(path, prefix='', postfix='', return_prefix=True,
            return_postfix=True, return_abs=False):
    """
    Lists all files in path that start with prefix and end with postfix.
    By default, this function returns all filenames. If you do not want to
    return the pre- or postfix, set the corresponding parameters to False.
    :param path:
    :param prefix:
    :param postfix:
    :param return_prefix:
    :param return_postfix:
    :return: list(str)
    """
    files = os.listdir(path)
    filtered_files = filter(
        lambda f: f.startswith(prefix) and f.endswith(postfix), files)
    return_files = filtered_files
    if not return_prefix:
        idx_start = len(prefix) - 1
        return_files = [f[idx_start:] for f in filtered_files]
    if not return_postfix:
        idx_end = len(postfix) - 1
        return_files = [f[:-idx_end-1] for f in filtered_files]
    return_files = set(return_files)
    result = list(return_files)
    if return_abs:
        result = [os.path.join(path, r) for r in result]
    return result


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def copy(src, dest, overwrite=False):
    if os.path.exists(dest) and overwrite:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        else:
            os.remove(dest)
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def copy_src(path_from, path_to):
    """
    Make sure to have everything in path_from folder
    There should not be large files (e.g. checkpoints or images)
    Args:
        path_from:
        path_to:

    Returns:

    """
    assert os.path.isdir(path_from)
    # Collect all files and folders that contain python files
    tmp_folder = os.path.join(path_to, 'src/')
    mkdir(tmp_folder)

    from_folder = os.path.basename(path_from)
    copy(path_from, os.path.join(tmp_folder, from_folder), overwrite=True)
    time_str = get_time_string()

    path_archive = os.path.join(path_to, "src_{}".format(time_str))
    shutil.make_archive(path_archive, 'zip', tmp_folder)
    try:
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        # We got a FileNotfound error on the cluster. Maybe some race conditions?
        pass
    print("Copied folder {} to {}".format(path_from, path_archive))