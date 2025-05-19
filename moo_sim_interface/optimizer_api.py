import argparse
import sys
from typing import Union

try:
    from moo_sim_interface.multi_objective_optimization_apis import paref_optimizer, pymoo_optimizer
except ImportError:
    print("The optimization modules are not available. Please use 'pip install moo_sim_interface[optim]', "
          "to install them.")
    sys.exit(1)
from moo_sim_interface.utils.setup_configs import setup_config_dir
from moo_sim_interface.utils.yaml_config_parser import parse_moo_config_file


def moo_env_apis_wrapper(return_results: bool = False, **args) -> Union[None, list]:
    optimizer_environment = args.get('package').lower()
    if optimizer_environment == 'paref':
        results = paref_optimizer.run_optimization(return_results=return_results, **args)
    elif optimizer_environment == 'pymoo':
        results = pymoo_optimizer.run_optimization(return_results=return_results, **args)
    else:
        raise ValueError(f'Unknown Optimization Environment: {optimizer_environment}')

    if return_results:
        return results


def main():
    parser = argparse.ArgumentParser('run_moo')
    parser.add_argument('-f', '--file',
                        help='Provide the filename of your .yml optimization configuration file in the "configs" dir '
                             'or an absolute path (optional)',
                        metavar='',
                        type=str)
    parser.add_argument('-s', '--setup',
                        action='store_true',
                        help='Copy generic config files (e.g., .yml files) to the current working directory.')

    launch_args = parser.parse_args()

    if launch_args.setup:
        setup_config_dir()
        if launch_args.file is None:
            return

    if launch_args.file is not None:
        moo_args = parse_moo_config_file(launch_args.f)
    else:
        print('No config file provided, using default config file from the current working directory.')
        moo_args = parse_moo_config_file()
    moo_env_apis_wrapper(**moo_args)


def run_optimizations(
        moo_config_file: str = None,
        overwrite_config: list[dict] = None,
        return_results: bool = True,
) -> list:
    if moo_config_file is None:
        print('No config file provided, using default config file from the current working directory.')
        moo_args = parse_moo_config_file(overwrite_config=overwrite_config)
    else:
        moo_args = parse_moo_config_file(moo_config_file, overwrite_config=overwrite_config)

    return moo_env_apis_wrapper(return_results=return_results, **moo_args)


if __name__ == '__main__':
    main()
