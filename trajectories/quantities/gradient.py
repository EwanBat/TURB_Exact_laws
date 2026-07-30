import numpy as np
import logging

logger = logging.getLogger(__name__)


class GradientQuantitiesMixIn:

    def _compute_gradient_4satellite(self, dic_quantities: dict, quantities: list):
        for quantity_name in quantities:
            if quantity_name not in self.GRADIENT_QUANTITIES:
                continue
            try:
                result = self._compute_quantity_vectorized(quantity_name, dic_quantities)
                if isinstance(result, dict):
                    dic_quantities['sat_0'].update(result)
                else:
                    dic_quantities['sat_0'][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _get_9satellite_faces_with_sat0(self):
        faces = [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        valid_tuples = []
        for surface in faces:
            for i in range(4):
                t = tuple(surface[:i] + surface[i+1:])
                valid_tuples.append(t)
        return valid_tuples

    def _compute_gradient_9satellite(self, dic_quantities: dict, quantities: list):
        satellite_offsets = self.traj_param.get('satellite_offsets', {})
        if not satellite_offsets:
            raise ValueError("Missing satellite_offsets in traj_param for nbsatellite=9")

        tuples = self._get_9satellite_faces_with_sat0()
        ref_offset = np.asarray(satellite_offsets['sat_0'])

        tetra_setups = []
        for (i, j, k) in tuples:
            source_sat_names = ["sat_0", f"sat_{i}", f"sat_{j}", f"sat_{k}"]
            dR1 = np.asarray(satellite_offsets[f'sat_{i}']) - ref_offset
            dR2 = np.asarray(satellite_offsets[f'sat_{j}']) - ref_offset
            dR3 = np.asarray(satellite_offsets[f'sat_{k}']) - ref_offset
            tetra_setups.append((source_sat_names, dR1, dR2, dR3))

        base_traj_param = dict(self.traj_param)
        base_traj_param['nbsatellite'] = 4

        for quantity_name in quantities:
            if quantity_name not in self.GRADIENT_QUANTITIES:
                continue

            grad_results = {}

            for source_sat_names, dR1, dR2, dR3 in tetra_setups:
                tuple_dic_quant = {
                    f"sat_{local_idx}": dic_quantities[source_sat_names[local_idx]]
                    for local_idx in range(4)
                }

                tuple_traj_param = dict(base_traj_param)
                tuple_traj_param['dR1'] = dR1
                tuple_traj_param['dR2'] = dR2
                tuple_traj_param['dR3'] = dR3

                result = self._compute_quantity_vectorized(
                    quantity_name, tuple_dic_quant, traj_param_override=tuple_traj_param,
                )

                if isinstance(result, dict):
                    for i, (key, value) in enumerate(result.items()):
                        if key not in grad_results:
                            grad_results[key] = []
                        grad_results[key].append(value)
                else:
                    if quantity_name not in grad_results:
                        grad_results[quantity_name] = []
                    grad_results[quantity_name].append(result)

            for key, values in grad_results.items():
                if key not in dic_quantities['sat_0']:
                    dic_quantities['sat_0'][key] = np.mean(values, axis=0)
                else:
                    dic_quantities['sat_0'][key].update(np.mean(values, axis=0))
