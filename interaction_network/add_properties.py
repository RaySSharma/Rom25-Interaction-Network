import argparse
import tqdm

from . import read_json, get_halos, CONFIG
from properties import halos, stars

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append new properties to the halo interaction network for input Romulus25 halo at step 7779"
    )
    parser.add_argument("--halos", help="File containing halo numbers at step 7779")
    parser.add_argument("--verbose", help="Verbose", action="store_true")
    args = parser.parse_args()

    verbose = bool(args.verbose)
    config = read_json(CONFIG)
    args = parser.parse_args()

    halo_numbers = get_halos(args.halos)
    assert len(halo_numbers) > 0
    assert type(halo_numbers[0]) == int

    if verbose:
        halo_numbers = tqdm.tqdm(halo_numbers)

    network_halos = halos.HaloProperties(config_filename=CONFIG, verbose=verbose)
    network_stars = stars.StarProperties(config_filename=CONFIG, verbose=verbose)

    networks = [network_halos, network_stars]
    for network in networks:
        for halo_number in halo_numbers:
            data = network.calculate(halo_number)
            network.save_network(halo_number, data, network.names)
    