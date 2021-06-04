import pathlib
import json
from json.decoder import JSONDecodeError

CONFIG = pathlib.Path() / "config.json"


def read_json(file_path):
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except (FileNotFoundError, JSONDecodeError) as err:
            print(err, "Invalid JSON config")


def get_halos(filename):
    """Generate list of halos from input file

    Args:
        filename (str): list of halos, one per line

    Returns:
        numpy.ndarray: list of halo number integers
    """
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    return [int(h) for h in lines]