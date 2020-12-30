import os

def get_local_path():
    """
    Gives the path form the environment to the Paraphraser folder
    """
    cwd = os.getcwd()
    file_name = '/RoBERTa/getPath.py'
    path = os.path.realpath(__file__)

    path = path.replace(cwd, '.', 1)
    path = path.replace(file_name, '')

    return path
