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

from anemoi.training.data.dataset import NativeGridDataset


class TestMixedDataset:
    """Test MixedDataset instantiation and properties."""

    @pytest.fixture
    def multi_domain(self, mocker: MockFixture) -> NativeGridDataset:
        """Fixture to provide a NativeGridDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = list(range(10))  # [0, 1, 2, ...7, 8, 9] valid date indices are [0, 1, 2, 3]
        mock_dataset_a.has_trajectories = False
        mock_dataset_a.frequency = "3h"

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = {8, 9}
        mock_dataset_b.dates = list(range(10))  # [0, 1, 2, ...5, 6, 9] valid date indices are [0, 1]
        mock_dataset_b.has_trajectories = False
        mock_dataset_b.frequency = "3h"

        mock_dataset_c = mocker.MagicMock()
        mock_dataset_c.missing = {0, 1}
        mock_dataset_c.dates = list(range(10))  # [0, 3, 4 ...7, 8, 9] valid date indices are [2, 3]
        mock_dataset_c.has_trajectories = False
        mock_dataset_c.frequency = "3h"

        data_readers = {
            "dataset_a": {"encoder": 0, "dataset": mock_dataset_a},
            "dataset_b": {"encoder": 1, "dataset": mock_dataset_b},
            "dataset_c": {"encoder": 1, "dataset": mock_dataset_c},
        }
        relative_date_indices = {
            "dataset_a": [0, 2, 6],
            "dataset_b": [0, 2, 6],
            "dataset_c": [0, 2, 6],
        }  # e.g. f([t, t-6h]) = t+12h
        sample_strategy = "anemoi.training.data.sharded_mixed_sampler.MixedSampler"
        return NativeGridDataset(
            data_readers=data_readers,
            relative_date_indices=relative_date_indices,
            sample_strategy=sample_strategy,
        )

    def test_valid_date_indices(self, multi_domain: NativeGridDataset) -> None:
        merged_valid_date_indices = multi_domain.sampler.merged_valid_date_indices
        expected_merged_valid_date_indices = {"group_0": [0, 1], "group_1": [2, 3]}
        assert np.array_equal(merged_valid_date_indices, expected_merged_valid_date_indices)

    def test_sharding(self, multi_domain: NativeGridDataset) -> None:
        """Test that sharding logic correctly partitions the dataset."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        expected_indices = {"group_0": [0], "group_1": [2]}  # worker 0 gets the first half of the data
        assert np.array_equal(multi_domain.sampler.chunk_index_range, expected_indices)

    def test_get_shuffled_chunk_indices(self, multi_domain: NativeGridDataset) -> None:
        """Test that get_shuffled_chunk_indices returns shuffled indices when shuffle is True."""
        multi_domain.per_worker_init(n_workers=1, worker_id=0)
        shuffled_indices = multi_domain.sampler.get_shuffled_chunk_indices()
        assert isinstance(shuffled_indices, np.ndarray)
        assert len(shuffled_indices) == 4  # should return all indices from both groups
        # Check that the indices are shuffled and are all present
        original_indices = [("group_0", 0), ("group_0", 1), ("group_1", 2), ("group_1", 3)]
        for idx in shuffled_indices:
            assert idx in original_indices

    def test_get_sample(self, multi_domain: NativeGridDataset) -> None:
        """Test that get_sample returns a dictionary of samples from all datasets."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        shuffled_indices = multi_domain.sampler.get_shuffled_chunk_indices()
        sample = multi_domain.sampler.get_sample(shuffled_indices[0])
        assert isinstance(sample, dict)
        assert len(sample) == 2  # should return a sample from two encoders
