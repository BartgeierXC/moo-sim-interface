from pathlib import Path
from typing import Union

import matlab
import numpy as np

from moo_sim_interface.utils.post_simulation_data_processor import PostSimulationDataProcessor
from moo_sim_interface.utils.yaml_config_parser import prepare_simulation_environment


def run_simulation(return_results: bool = False, **args) -> Union[None, list]:
    (model_filename, model_path, input_values, input_names, num_chunks, final_names, sync_execution,
     time_modulo, result_transformation, custom_build_dir) = prepare_simulation_environment(args)

    post_simulation_data_processor = PostSimulationDataProcessor()

    model_file = args.get('model_file')
    model_path = Path(model_file).stem
    model_name = args.get('model_name')

    function_name = Path(model_file).name

    n_outputs = len(args.get('simulation_setup').get('output_configuration').get('parameter_names'))

    eng = matlab.engine.start_matlab()
    path = eng.genpath(model_path)
    eng.addpath(path, nargout=0)

    matlab_function = getattr(eng, function_name)

    indices = list(np.ndindex(input_values[0].shape if len(input_values) > 0 else (1,)))

    combined_results = []
    for i in indices:
        initial_values = [values[i] for values in input_values]  # set the start values

        # TODO: make the call generic
        results = matlab_function(*[matlab.double(val) for val in initial_values], nargout=n_outputs)
        combined_results.append([(i, result_transformation(results))])
        combined_results.append([])  # placeholder for all parameters results

    processed_results = post_simulation_data_processor.do_post_processing(args, input_values, combined_results,
                                                                          model_name, return_results=return_results)

    if return_results:
        return processed_results
