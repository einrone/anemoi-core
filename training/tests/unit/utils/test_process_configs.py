# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
from pytest_mock import MockFixture

from omegaconf import OmegaConf

from anemoi.training.utils.process_configs import ProcessConfigs


class TestProcessConfigs:
    """Test ProcessConfigs instantiation and properties."""

    @pytest.fixture
    def process_configs(self, mocker: MockFixture) -> ProcessConfigs:
        """Fixture to provide a ProcessConfigs instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_base_config = OmegaConf.create("/leonardo_work/DestE_340_26/users/sbuurman/MD-PR/forked_PR/anemoi-core/training/src/anemoi/training/config/hectometric_finetuning_lowres.yaml")
        mock_base_config.dataloader.hectometric = False
        return ProcessConfigs(base_config=mock_base_config)

    def test_process(self, process_configs: ProcessConfigs) -> None:
        """Test that sharding logic correctly partitions the dataset."""
        process_configs.process()

        print(process_configs.TEMPORARY)
    
    def test_update(self, process_configs: ProcessConfigs) -> None:
        """Test that update logic correctly updates the configuration."""
        new_config = process_configs.update()
        print(new_config["dataloader"])
