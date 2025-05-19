import os
import subprocess

import pytest

skip_docker = pytest.mark.skipif(
    os.getenv('SKIP_DOCKER_TESTS') == 'true',
    reason="Skipping tests that should only run via ci in Docker"
)


@skip_docker
def test_setup_argument_creates_config_files_in_pwd(tmp_path):
    current_dir = tmp_path
    # os.chdir(current_dir)

    result = subprocess.run(
        ["run_sim", "-s"],
        capture_output=True,
        text=True,
    )

    assert (current_dir / 'configs' / 'generic' / 'optimization_config.yml').exists()
    assert (current_dir / 'configs' / 'generic' / 'simulation_config.yml').exists()

    assert f"Configuration files copied to {current_dir}" in result.stdout
