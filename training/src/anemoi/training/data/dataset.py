# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import itertools
import logging
import os
import random
from abc import ABC
from functools import cached_property

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_data_indices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)


class AnemoiDataset(IterableDataset, ABC):
    """Base Anemoi Datasets torch dataset class."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multi",
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        shuffle : bool, optional
            Shuffle batches, by default True
        """
        self.data_readers = data_readers
        self.shuffle = shuffle
        self.label = label
        self.dataset_names = list(data_readers.keys())
        self._lazy_init_model_and_reader_group_info()
        self.relative_date_indices = relative_date_indices
        self._init_valid_date_indices()

        # Normalize the date indices to use slices where possible, which can improve downstream indexing performance.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in self.relative_date_indices.items()
        }
        self.n_samples_per_worker = {}  # overwrite base to empty dict
        self.chunk_index_range = {}  # overwrite base to empty dict

    def _init_valid_date_indices(self) -> None:
        """Get valid date indices for each dataset."""
        # get dataset labels, encoder labels
        self.dataset_labels = list(self.data_readers.keys())
        encoder_labels = {self.data_readers[dataset_label].encoder for dataset_label in self.dataset_labels}

        # Group datasets by encoder, will look like: {0: [dataset0], 1: [dataset1, dataset2]}
        datasets_per_encoder = {encoder_label: [] for encoder_label in encoder_labels}
        for encoder_label in encoder_labels:
            datasets_per_encoder[encoder_label] = [
                dataset_label
                for dataset_label in self.dataset_labels
                if self.data_readers[dataset_label].encoder == encoder_label
            ]
        # Create groups of datasets that will be sampled together
        groups = list(itertools.product(*datasets_per_encoder.values()))
        self.groups_dict = {"group_" + str(i): group for i, group in enumerate(groups)}
        self.group_labels = list(self.groups_dict.keys())

        # Compute valid date indices for each group of datasets
        self.valid_date_indices = {}
        for group, datasets_in_group in self.groups_dict.items():
            group_valid_date_indices = compute_valid_data_indices(
                {dataset_label: self.data_readers[dataset_label] for dataset_label in datasets_in_group},
                self.relative_date_indices,
            )
            if len(group_valid_date_indices) > 0:
                self.valid_date_indices[group] = group_valid_date_indices

        assert len(self.valid_date_indices) > 0, "No valid date indices found for any group of datasets."

    def _lazy_init_model_and_reader_group_info(self) -> None:
        """Lazy initialize model and reader group info."""
        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

        self.shard_shapes = None

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all data readers."""
        return {name: getattr(dataset, attr_name) for name, dataset in self.data_readers.items()}

    @cached_property
    def statistics(self) -> dict[str, dict]:
        """Return combined statistics from all data readers."""
        return self._collect("statistics")

    @cached_property
    def metadata(self) -> dict[str, dict]:
        """Return combined metadata from all data readers."""
        return self._collect("metadata")

    @cached_property
    def supporting_arrays(self) -> dict[str, dict]:
        """Return combined supporting arrays from all data readers."""
        return self._collect("supporting_arrays")

    @cached_property
    def variables(self) -> dict[str, list[str]]:
        """Return combined variables from all data readers."""
        return self._collect("variables")

    @property
    def data(self) -> dict:
        """Return data from all data readers as dictionary."""
        return self._collect("data")

    @cached_property
    def name_to_index(self) -> dict[str, dict]:
        """Return combined name_to_index mapping from all data readers."""
        return self._collect("name_to_index")

    @cached_property
    def resolution(self) -> dict[str, str]:
        """Return combined resolution from all data readers."""
        return self._collect("resolution")

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return combined frequency from all data readers."""
        freqs = self._collect("frequency")
        freq_ref = None
        for name, freq in freqs.items():
            if freq_ref is None:
                freq_ref = freq
            assert freq == freq_ref, f"Data reader '{name}' has different frequency than other data readers"
        return freq_ref

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
        shard_shapes: dict[str, list[int]],
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        shard_shapes : dict[str, list[int]]
            Shard shapes for all data readers
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        self.shard_shapes = shard_shapes

        assert self.reader_group_size >= 1, f"reader_group_size(={self.reader_group_size}) must be positive"

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d, "
            "sample_comm_group_id %d, sample_comm_num_groups %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
            self.sample_comm_group_id,
            self.sample_comm_num_groups,
        )
        

    def set_ens_comm_group_info(
        self,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
    ) -> None:
        """Set ensemble communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        ens_comm_group_id : int
            Ensemble communication group ID
        ens_comm_group_rank : int
            Ensemble communication group rank
        ens_comm_num_groups : int
            Number of ensemble communication groups
        """
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups

        self.sample_comm_group_id = ens_comm_group_id
        self.sample_comm_num_groups = ens_comm_num_groups

        LOGGER.info(
            "NativeGridDataset.set_ens_comm_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d, "
            "sample_comm_group_id %d, sample_comm_num_groups %d",
            self.global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            self.reader_group_rank,
            self.sample_comm_group_id,
            self.sample_comm_num_groups,
        )
    @property
    def field_shapes(self) -> dict[str, list[int]]:
        """Return field shapes for all data readers."""
        return self._collect("field_shape")
        
    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
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
        self.n_samples_per_worker = {}
        for group in self.group_labels:
            shard_size = len(self.valid_date_indices[group]) // self.sample_comm_num_groups
            shard_start = self.sample_comm_group_id * shard_size

            self.n_samples_per_worker[group] = shard_size // n_workers
            low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)

            self.chunk_index_range[group] = np.arange(low, high, dtype=np.uint32)

            LOGGER.info(
                "Worker %d (pid %d, model comm group %d)  has low/high range %d / %d",
                worker_id,
                os.getpid(),
                self.model_comm_group_id,
                low,
                high,
            )

            base_seed = get_base_seed()
            torch.manual_seed(base_seed)
            random.seed(base_seed)
            self.rng = np.random.default_rng(seed=base_seed)
            sanity_rnd = self.rng.random(1)[0]
            LOGGER.info(
                ("Worker %d (pid %d, base_seed %d, sanity rnd %f)"),
                self.worker_id,
                os.getpid(),
                base_seed,
                sanity_rnd,
            )
        sample = next(self.__iter__())
        # LOGGER.info(f"TEST SAMPLE: {sample.keys()}, {sample[list(sample.keys())[0]].shape}")

    @cached_property
    def shard_shapes(self) -> dict[str, list]:
        """Return shard shapes for all data readers."""
        shard_shapes = {}
        print("Reader group size", self.reader_group_size)
        for name, dataset in self.data_readers.items():
            print("dataset name", name)
            print("grid size", dataset.grid_size)
            shard_shapes[name] = get_balanced_partition_sizes(dataset.grid_size, self.reader_group_size)
        print("shard shapes", shard_shapes)
        return shard_shapes

    def get_shard_slice(self, dataset_name: str, reader_group_rank: int) -> slice:
        """Get the grid shard slice according to the reader rank."""
        start, end = get_partition_range(
            partition_sizes=self.shard_shapes[dataset_name],
            partition_id=reader_group_rank,
        )
        return slice(start, end)

    def get_shuffled_chunk_indices(self) -> list[tuple[str, int]]:
        """Get the shuffled chunk indices from the dataset.

        Returns
        -------
            list[tuple[str, int]]: A list of tuples containing the domain name and index for each shuffled chunk.
        """
        print("shard shapes", self.shard_shapes)
        if self.shuffle:
            shuffled_chunk_indices = {
                group: self.rng.choice(
                    indices,
                    size=len(indices),
                    replace=False,
                )[self.chunk_index_range[group]]
                for group, indices in self.valid_date_indices.items()
            }
            print("shuffled_chunk_indices", shuffled_chunk_indices)

            labeled_samples_and_indexes = [
                (group, i) for group, indices in shuffled_chunk_indices.items() for i in indices
            ]
            print("labeled_samples_and_indexes", labeled_samples_and_indexes)

            labeled_samples = self.rng.choice(
                labeled_samples_and_indexes,
                size=len(labeled_samples_and_indexes),
                replace=False,
            )
            print("labeled_samples ", labeled_samples)
        else:
            shuffled_chunk_indices = {
                group: indices[self.chunk_index_range[group]] for group, indices in self.valid_date_indices.items()
            }
            labeled_samples = [
                (str(group), int(i.item())) for group, inds in shuffled_chunk_indices.items() for i in inds
            ]
        print("samples", labeled_samples)

        return labeled_samples

    def get_sample(self, index: tuple[str, int]) -> torch.Tensor:
        LOGGER.info("Getting sample for index %s", index)
        group_name, i = index
        datasets_in_group = self.groups_dict[group_name]
        x = {}
        LOGGER.info(f"group {group_name}")
        print("index", i)
        print("datasets in group", datasets_in_group)
        print("ALL GROUPS", self.groups_dict)
        for name in datasets_in_group:
            dataset = self.data_readers[name]
            print("dataset", name)
            time_step = offset_time_indices(int(i), self.relative_date_indices[name])
            print("time step", i)
            if self.shard_shapes is not None and self.shard_shapes[name] is not None:
                start, end = get_partition_range(self.shard_shapes[name], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            print("grid indices", grid_indices)
            grid_indices = slice(None)
            x[name] = dataset.get_sample(time_step, grid_indices)
        print('dataset names', x.keys())
        print("dataset shapes", [val.shape for val in x.values()])
        LOGGER.info(f"dataset name {x.keys()} dataset shape {x[list(x.keys())[0]].shape}")
        return x

    def __iter__(self) -> None:
        """Return an iterator that yields a tuple torch.Tensor and its corresponding domain name.

        Returns
        -------
        tuple[torch.Tensor, str]
            A tuple containing the tensor sample and its corresponding domain name
        """
        shuffled_chunk_indices = self.get_shuffled_chunk_indices()
        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d"
            ),
            os.getpid(),
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
        )
        LOGGER.info(f"shuffled_chunk_indices {shuffled_chunk_indices}")

        for i in shuffled_chunk_indices:
            LOGGER.debug(
                (
                    "Worker pid %d yielding sample for index %s, worker id %d, global_rank %d, "
                    "model comm group %d, group_rank %d, seed comm group id %d"
                ),
                os.getpid(),
                i,
                self.worker_id,
                self.global_rank,
                self.model_comm_group_id,
                self.model_comm_group_rank,
                self.sample_comm_group_id,
            )
            yield self.get_sample(i)

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self) -> Tree:
        tree = Tree(f"{self.__class__.__name__}")
        for name, dataset in self.data_readers.items():
            subtree = dataset.tree(prefix=name)
            tree.add(subtree)
        return tree
