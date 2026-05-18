# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import random

import numpy as np
import torch

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_union_valid_data_indices
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)

# it may be that multidomain is not a good name for this, but it is what it is for now


class MixedSampler:
    """Mixing multi dataset and multi domain samplers."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "mixed",
        shard_shapes: dict[str, list[int]] | None = None,
    ) -> None:
        """A dataset that combines multiple data_readers together.

        Args:
            data_readers (dict[str, BaseAnemoiReader]):
                A dictionary mapping domain names to their corresponding data readers for each encoder.
            relative_date_indices (dict[str, TimeIndices]):
                A dictionary mapping domain names to their corresponding relative date indices.
            shuffle (bool, optional):
                Whether to shuffle the data. Defaults to True.
            label (str, optional):
                A label for this sampler. Defaults to "mixed".
        Return:
            None
        """
        self.data_readers = data_readers
        self.shuffle = shuffle
        self.label = label
        self.shard_shapes = shard_shapes
        self.valid_date_indices = {
            encoder_label: {
                name: get_usable_indices(
                    ds.missing,
                    len(ds.dates),
                    relative_date_indices[name],
                    ds.trajectory_ids if ds.has_trajectories else None,
                )
                for name, ds in self.data_readers[encoder_label].items()
            }
            for encoder_label in ["enc_0", "enc_1"]
        }
        merged_data_readers = {
            name: ds for encoder_label in ["enc_0", "enc_1"] for name, ds in self.data_readers[encoder_label].items()
        }
        self.merged_valid_date_indices = compute_union_valid_data_indices(merged_data_readers, relative_date_indices)

        # Normalize the date indices to use slices where possible, which can improve downstream indexing performance.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }
        self.dataset_names = list(merged_data_readers.keys())
        LOGGER.info("valid date indices: %s", self.valid_date_indices)
        self.n_samples_per_worker = {}  # overwrite base to empty dict
        self.chunk_index_range = {}  # overwrite base to empty dict

    def per_worker_init(
        self,
        n_workers: int,
        worker_id: int,
        sample_comm_num_groups: int,
        sample_comm_group_id: int,
        model_comm_group_id: int,
    ) -> None:
        """Initialize a specific worker, based on the valid date indices.

        Args:
            n_workers : int
                The total number of workers.
            worker_id : int
                The ID of the current worker (0-indexed).
            sample_comm_num_groups : int
                The number of sample communication groups.
            sample_comm_group_id : int
                The ID of the sample communication group.
            model_comm_group_id : int
                The ID of the model communication group.

        Returns
        -------
            None
        """
        self.worker_id = worker_id
        shard_size = len(self.available_enc_dict.keys()) // sample_comm_num_groups
        shard_start = sample_comm_group_id * shard_size

        n_samples_per_worker = shard_size // n_workers
        low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()
        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)[0]
        LOGGER.info(
            ("Worker %d (%s, pid %d, base_seed %d, sanity rnd %f)"),
            worker_id,
            self.label,
            os.getpid(),
            base_seed,
            sanity_rnd,
        )
        return n_samples_per_worker

    @property
    def available_enc_dict(self) -> dict[str, list[str]]:
        """Get the available samples per encoder for a given index.

        Returns
        -------
            dict[str, list[str]]:
            A dictionary mapping encoder labels to lists of dataset names that have valid data for the given index.
        """
        encoder_labels = ["enc_0", "enc_1"]
        available_enc_dict = {}
        for i in self.merged_valid_date_indices:
            available_enc = {}
            for encoder_label in encoder_labels:
                available_datasets = [
                    dataset
                    for dataset in self.valid_date_indices[encoder_label]
                    if i in self.valid_date_indices[encoder_label][dataset]
                ]
                if available_datasets:  # only include encoders that have available datasets for this index
                    available_enc[encoder_label] = available_datasets
            if available_enc:  # only include indices that have at least one available encoder
                available_enc_dict[i.item()] = available_enc
        LOGGER.info("Available encoders for each index: %s", available_enc_dict)
        return available_enc_dict

    def get_shuffled_chunk_indices(self) -> list[tuple[str, int]]:
        """Get the shuffled chunk indices from the dataset.

        Returns
        -------
            list[tuple[str, int]]: A list of tuples containing the domain name and index for each shuffled chunk.
        """
        if self.shuffle:
            LOGGER.info(
                "Shuffling chunk indices for chunk index range %s with available_enc_dict %s",
                self.chunk_index_range,
                self.available_enc_dict,
            )
            shuffled_chunk_indices = self.rng.choice(
                list(self.available_enc_dict.keys()),
                size=len(self.available_enc_dict),
                replace=False,
            )[self.chunk_index_range]

        else:
            shuffled_chunk_indices = {list(self.available_enc_dict.keys())[self.chunk_index_range]}
        return shuffled_chunk_indices

    def get_sample(self, index: tuple[str, int]) -> torch.Tensor:
        LOGGER.debug("Getting sample for index %s", index)

        sample = {}
        for encoder in self.available_enc_dict[index]:
            dataset_label = str(
                self.rng.choice(self.available_enc_dict[index][encoder]),
            )  # randomly choose one of the available datasets
            time_step = offset_time_indices(int(index), self.relative_date_indices[dataset_label])
            if self.shard_shapes is not None and self.shard_shapes[dataset_label] is not None:
                start, end = get_partition_range(self.shard_shapes[dataset_label], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            sample[encoder] = {
                dataset_label: self.data_readers[encoder][dataset_label].get_sample(time_step, grid_indices),
            }
        LOGGER.info("Retreived sample for index %s: %s", index, sample)
        return sample
