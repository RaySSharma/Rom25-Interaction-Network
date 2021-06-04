import tqdm
import tangos as db
import numpy as np
import pathlib
import pynbody
from . import BaseProperties


class StarProperties(BaseProperties):

    names = ["mstar_max_neighbor"]

    def __init__(self, config_filename=None, verbose=False):
        super().__init__(config_filename=config_filename, verbose=verbose)

    def calculate(self):
        """Generate interaction parameters for given halo

        Args:
            halo_num (int): halo number at desired step
        """
        mstar_max = self._calc_halo_mstar_max(self.network)
        v_over_sigma = self._calc_v_over_sigma(self.network)
        return mstar_max

    def _calc_halo_mstar_max(self, data):
        """Iterate over each most significant neighbor, trace each of their progenitors back in time, and find max stellar max from AHF

        Args:
            neighbors (pandas.DataFrame): *AHF_halos DataFrame
        """

        mstar_max_arr = []
        rows = data.iterrows()
        if self.verbose:
            rows = tqdm.tqdm(rows)

        for i, row in rows:
            snapshot_name = (
                str(self.sim_config["TANGOS_DB"]) + "/%00" + str(int(row["timestep"]))
            )
            neighbor_halo = db.get_halo(
                snapshot_name + "/" + str(int(row["neighbor_halo_num"]))
            )

            if neighbor_halo is not None:
                progs = db.relation_finding.MultiHopMajorProgenitorsStrategy(
                    neighbor_halo
                ).all()
                mstar = []
                for prog in progs:
                    prog_halo_num = int(prog.halo_number)
                    prog_timestep = str(self.get_timestep_number(prog))

                    snapshot_name = pathlib.Path(self.sim_config["SIM_DIR"]).glob(
                        "*00" + prog_timestep
                    )
                    snapshot_name = next(snapshot_name).as_posix()
                    prog_halo = (
                        pynbody.load(snapshot_name)
                        .halos(dosort=True)
                        .load_copy(prog_halo_num)
                    )
                    prog_halo.physical_units()

                    mstar.append(prog_halo.s["mass"].sum())

                try:
                    mstar_max = np.hstack(mstar).max()
                except ValueError:
                    mstar_max = 0.0
            else:
                mstar_max = 0.0

            mstar_max_arr.append(mstar_max)
        return np.asarray(mstar_max_arr)

    def _calc_v_over_sigma(self, data):
        """Iterate over each most significant neighbor, trace each of their progenitors back in time, and find max stellar max from AHF

        Args:
            neighbors (pandas.DataFrame): *AHF_halos DataFrame
        """

        v_sig_arr = []
        rows = data.iterrows()
        if self.verbose:
            rows = tqdm.tqdm(rows)

        for i, row in rows:
            snapshot_name = pathlib.Path(self.sim_config["SIM_DIR"]).glob(
                        "*00" + str(int(row["timestep"]))
                    )
            snapshot_name = next(snapshot_name).as_posix()

            neighbor_halo = (
                pynbody.load(snapshot_name)
                .halos(dosort=True)
                .load_copy(int(row["neighbor_halo_num"]))
            )

            neighbor_halo.physical_units()
            pynbody.analysis.angmom.faceon(neighbor_halo)
            vel_tan = neighbor_halo.s["vt"].mean()
            vel_sig = neighbor_halo.s["v_disp"].mean()
            v_sig = vel_tan / vel_sig

            v_sig_arr.append(v_sig)
        return np.asarray(v_sig_arr)