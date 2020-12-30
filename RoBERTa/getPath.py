import os
import platform

def get_local_path():
    """
    Gives the path form the environment to the Paraphraser folder
    (in the form "./path/to/environment/")
    """
    cwd = os.getcwd()
    if platform.system() == 'Windows':
        cwd = cwd[0].lower() + cwd[1:]
    file_name = os.path.join('RoBERTa', 'getPath.py')
    path = os.path.realpath(__file__)

    path = path.replace(cwd, '.', 1)
    path = path.replace(file_name, '')

    return path
