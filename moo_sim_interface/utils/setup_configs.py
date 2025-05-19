import os
import shutil


def setup_config_dir():
    package_data_dir = os.path.join(os.path.dirname(__file__), '../../configs/generic')
    target_dir = os.getcwd()

    if os.path.exists(package_data_dir):
        for file_name in os.listdir(package_data_dir):
            full_file_name = os.path.join(package_data_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, target_dir)
        print(f'Configuration files copied to {target_dir}')
    else:
        print(f'Package data directory not found: {package_data_dir}')
