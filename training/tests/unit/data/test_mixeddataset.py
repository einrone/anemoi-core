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

from anemoi.training.data.dataset import AnemoiDataset


class TestMixedDataset:
    """Test MixedDataset instantiation and properties."""

    @pytest.fixture
    def multi_domain(self, mocker: MockFixture) -> AnemoiDataset:
        """Fixture to provide a AnemoiDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = list(range(10))  # [0, 1, 2, ...7, 8, 9]
        mock_dataset_a.has_trajectories = False
        mock_dataset_a.frequency = "3h"

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = {7, 8, 9}
        mock_dataset_b.dates = list(range(10))  # [0, 1, 2, ...5, 6, 9]
        mock_dataset_b.has_trajectories = False
        mock_dataset_b.frequency = "3h"

        mock_dataset_c = mocker.MagicMock()
        mock_dataset_c.missing = {1, 2}
        mock_dataset_c.dates = list(range(10))  # [0, 3, 4 ...7, 8, 9]
        mock_dataset_c.has_trajectories = False
        mock_dataset_c.frequency = "3h"

        data_readers = {
            "enc_0": {"dataset_a": mock_dataset_a},
            "enc_1": {"dataset_b": mock_dataset_b, "dataset_c": mock_dataset_c},
        }
        relative_date_indices = {
            "dataset_a": [0, 2, 6],
            "dataset_b": [0, 2, 6],
            "dataset_c": [0, 2, 6],
        }  # e.g. f([t, t-6h]) = t+12h
        sample_strategy = "anemoi.training.data.mixed_sampler.MixedSampler"
        return AnemoiDataset(
            data_readers=data_readers,
            relative_date_indices=relative_date_indices,
            sample_strategy=sample_strategy,
        )

    def test_merged_valid_date_indices(self, multi_domain: AnemoiDataset) -> None:
        merged_valid_date_indices = multi_domain.sampler.merged_valid_date_indices
        expected_merged_valid_date_indices = [0, 1, 2, 3]
        assert np.array_equal(merged_valid_date_indices, expected_merged_valid_date_indices)

    def test_available_enc_dict(self, multi_domain: AnemoiDataset) -> None:
        """Test that available_enc_dict returns a dictionary of available encoders and their datasets."""
        available_enc_dict = multi_domain.sampler.available_enc_dict[0]
        expected_available_enc_dict = {
            "enc_0": ["dataset_a"],
            "enc_1": ["dataset_b"],
        }
        assert available_enc_dict == expected_available_enc_dict

    def test_sharding(self, multi_domain: AnemoiDataset) -> None:
        """Test that sharding logic correctly partitions the dataset."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        expected_indices = np.array([0, 1])
        assert np.array_equal(multi_domain.sampler.chunk_index_range, expected_indices)

    def test_valid_date_indices(self, multi_domain: AnemoiDataset) -> None:
        """Test that valid_date_indices returns a dictionary of indices from all datasets."""
        # Test valid_date_indices property
        valid_indices = multi_domain.sampler.valid_date_indices
        expected_indices = {
            "enc_0": {"dataset_a": np.array([0, 1, 2, 3])},
            "enc_1": {"dataset_b": np.array([0]), "dataset_c": np.array([3])},
        }
        for key in expected_indices:
            for subkey in expected_indices[key]:
                assert np.array_equal(valid_indices[key][subkey], expected_indices[key][subkey])

    def test_get_sample(self, multi_domain: AnemoiDataset) -> None:
        """Test that get_sample returns a dictionary of samples from all datasets."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        shuffled_indices = multi_domain.sampler.get_shuffled_chunk_indices()
        sample = multi_domain.sampler.get_sample(shuffled_indices[0])
        assert isinstance(sample, dict)
        assert len(sample) == 2  # should return a sample from two encoders
