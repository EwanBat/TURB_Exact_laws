import h5py
import logging

logger = logging.getLogger(__name__)


class QuantitiesIOMixIn:

    def quantities_to_h5(self, dic_quant: dict, filename: str):
        with h5py.File(filename, 'w') as f:
            for sat_name, sat_data in dic_quant.items():
                group = f.create_group(sat_name)
                for var_name, data_array in sat_data.items():
                    group.create_dataset(var_name, data=data_array,
                                         compression="gzip", compression_opts=9)
