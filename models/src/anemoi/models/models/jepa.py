# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import copy
import logging
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class WorldJepa(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality."""

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        
        self.encoder_groups = DotDict(model_config).model.get("encoder_groups", {})
        self.condition_on_residual = DotDict(model_config).model.condition_on_residual
        
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
        )

       

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = {}
        self.num_input_channels_prognostic = {}

        self._internal_input_idx = {}
        self.input_dim = {}
        self.input_dim_latent = self._calculate_input_dim_latent()

        for dataset_name, dataset_indices in data_indices.items():
            encoder_name = self.encoder_groups.get(dataset_name, dataset_name)
            self._internal_input_idx[encoder_name] = (
                dataset_indices.model.input.prognostic
            )

            self.num_input_channels[encoder_name] = len(dataset_indices.model.input)

            self.num_input_channels_prognostic[encoder_name] = len(
                dataset_indices.model.input.prognostic
            )
            self.input_dim[encoder_name] = self._calculate_input_dim(encoder_name)

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""
        # Encoder data -> hidden
        self.encoder_graph_provider = torch.nn.ModuleDict()
        self.encoder = torch.nn.ModuleDict()
        self.ema_encoder = torch.nn.ModuleDict()
        self.dataset_to_encoder = {}

        for dataset_name in self.dataset_names:
            encoder_name = self.encoder_groups.get(dataset_name, dataset_name)
            self.dataset_to_encoder[dataset_name] = encoder_name

            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden)],
                edge_attributes=model_config.model.encoder.get(
                    "sub_graph_edge_attributes"
                ),
                src_size=self.node_attributes.num_nodes[dataset_name],
                dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )
            if encoder_name not in self.encoder:
                self.encoder[encoder_name] = instantiate(
                    model_config.model.encoder,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    in_channels_src=self.input_dim[dataset_name],
                    in_channels_dst=self.input_dim_latent,
                    hidden_dim=self.num_channels,
                    edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
                )
                self.ema_encoder[encoder_name] = copy.deepcopy(
                    self.encoder[encoder_name]
                )
                for p in self.ema_encoder[encoder_name].parameters():
                    p.requires_grad = False

        # Processor hidden -> hidden
        self.processor_graph_provider = create_graph_provider(
            graph=self._graph_data[
                (self._graph_name_hidden, "to", self._graph_name_hidden)
            ],
            edge_attributes=model_config.model.processor.get(
                "sub_graph_edge_attributes"
            ),
            src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            trainable_size=model_config.model.processor.get("trainable_size", 0),
        )

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
            graph_data=self._graph_data,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        base_input_dim += 1  # for forecast step (fcstep)
        if self.condition_on_residual:
            base_input_dim += self.num_input_channels_prognostic[dataset_name]
        return base_input_dim

    def _assemble_input(
        self,
        x: torch.Tensor,
        fcstep: int,
        batch_ens_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ShardSizes]:
        assert (
            dataset_name is not None
        ), "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(
            dataset_name, batch_size=batch_ens_size
        )
        grid_shard_sizes = (
            grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None
        )

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(
                node_attributes_data, 0, grid_shard_sizes, model_comm_group
            )

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(
                    x,
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                node_attributes_data,
                torch.ones(batch_ens_size * x.shape[3], device=x.device).unsqueeze(-1)
                * fcstep,
            ),
            dim=-1,  # feature dimension
        )

        if self.condition_on_residual:
            x_skip_cond = x_skip[:, 0] if x_skip.ndim == 5 else x_skip
            x_data_latent = torch.cat(
                (
                    x_data_latent,
                    einops.rearrange(x_skip_cond, "bse grid vars -> (bse grid) vars"),
                ),
                dim=-1,
            )

        return x_data_latent, x_skip, grid_shard_sizes

    def _fetch_hidden_latent(
        self, batch_ens_size: int, model_comm_group: ProcessGroup
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_hidden_latent = self.node_attributes(
            self._graph_name_hidden, batch_size=batch_ens_size
        )
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(
            x_hidden_latent, 0, shard_sizes_hidden, model_comm_group
        )
        return x_hidden_latent, shard_sizes_hidden

    def _fetch_data_latent(
        self,
        x: torch.Tensor,
        dataset_name: str,
        fcstep: float,
        batch_ens_size: int,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: ProcessGroup,
    ) -> tuple[torch.Tensor, ShardSizes]:
        x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
            x[dataset_name],
            fcstep=fcstep,
            batch_ens_size=batch_ens_size,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
        )
        return x_data_latent, x_skip, shard_sizes_data

    def context_encoder(
        self,
        encoder_name: str,
        x_data_latent: torch.Tensor,
        x_hidden_latent: torch.Tensor,
        batch_ens_size: int,
        enc_shard_info: GraphShardInfo,
        encoder_edge_attr: torch.Tensor,
        encoder_edge_index: torch.Tensor,
        model_comm_group: ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Context encoder for encoding the input data into the latent space."""
        x_data_latent, x_latent = self.encoder[encoder_name](
            (x_data_latent, x_hidden_latent),
            batch_size=batch_ens_size,
            shard_info=enc_shard_info,
            edge_attr=encoder_edge_attr,
            edge_index=encoder_edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )
        return x_data_latent, x_latent

    @torch.no_grad()
    def ema_target_encoder(
        self,
        encoder_name: str,
        x_data_latent: torch.Tensor,
        x_hidden_latent: torch.Tensor,
        batch_ens_size: int,
        enc_shard_info: GraphShardInfo,
        encoder_edge_attr: torch.Tensor,
        encoder_edge_index: torch.Tensor,
        model_comm_group: ProcessGroup,
    ) -> torch.Tensor:
        """EMA (exponential moving average)encoder for encoding the target data into the latent space."""
        x_data_latent, x_latent = self.ema_encoder[encoder_name](
            (x_data_latent, x_hidden_latent),
            batch_size=batch_ens_size,
            shard_info=enc_shard_info,
            edge_attr=encoder_edge_attr,
            edge_index=encoder_edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )
        return x_data_latent, x_latent

    def update_ema_encoder(self, momentum: float = 0.999, inplace: bool = True) -> None:
        """Updates the EMA encoder parameters."""
        for dataset_name in self.encoder:
            for param, ema_param in zip(
                self.encoder[dataset_name].parameters(),
                self.ema_encoder[dataset_name].parameters(),
            ):
                if inplace:
                    ema_param.data.mul_(momentum).add_(param.data, alpha=1 - momentum)
                else:
                    ema_param.data = (
                        momentum * ema_param.data + (1 - momentum) * param.data
                    )

    def inject_noise(
        self,
        x_latent: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        **kwargs,
    ) -> torch.Tensor:
        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=grid_size,
            **kwargs,
        )
        return x_latent_proc, latent_noise

    def predictor(
        self,
        x_latent_proc: torch.Tensor,
        batch_ens_size: int,
        shard_sizes_hidden: list[int],
        proc_edge_shard_sizes: list[int],
        processor_edge_attr: torch.Tensor,
        processor_edge_index: torch.Tensor,
        model_comm_group: ProcessGroup,
        processor_kwargs: dict,
    ) -> torch.Tensor:

        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=batch_ens_size,
            shard_info=GraphShardInfo(
                nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes
            ),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )
        return x_latent_proc

    def forward(
        self,
        x: dict[str, torch.Tensor],
        *,
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward operator.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input tensor, shape (bs, m, e, n, f)
        fcstep : int
            Forecast step
        model_comm_group : ProcessGroup, optional
            Model communication group
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Tensor]
            Output tensor per dataset
        """
        
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        batch_ens_size = (
            batch_size * ensemble_size
        )  # batch and ensemble dimensions are merged
        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(
                batch_size,
                ensemble_size,
                in_out_sharded[dataset_name],
                model_comm_group,
            )

        fcstep = min(1, fcstep)
        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_input_data_dict = {}

        y_data_latent_dict = {}
        target_latents = {}
        shard_sizes_target_data_dict = {}

        x_hidden_latent, shard_sizes_hidden = self._fetch_hidden_latent(
            batch_ens_size, model_comm_group
        )
        
        y_hidden_latent = x_hidden_latent  # For JEPA, the target encoder uses the same hidden latent as the context encoder

        for dataset_name in dataset_names:
            data = x[dataset_name]
            x = data["input"]
            y = data["target"]

            encoder_name = self.dataset_to_encoder[dataset_name]

            x_data_latent, x_skip, shard_sizes_data = self._fetch_data_latent(
                x,
                dataset_name,
                fcstep,
                batch_ens_size,
                grid_shard_sizes,
                model_comm_group,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_input_data_dict[dataset_name] = shard_sizes_data

            (
                encoder_edge_attr,
                encoder_edge_index,
                enc_edge_shard_sizes,
            ) = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_input_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            x_data_latent, x_latent = self.context_encoder(
                encoder_name=encoder_name,
                x_data_latent=x_data_latent,
                x_hidden_latent=x_hidden_latent,
                batch_ens_size=batch_ens_size,
                enc_shard_info=enc_shard_info,
                encoder_edge_attr=encoder_edge_attr,
                encoder_edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
            )
            x_data_latent_dict[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent



            y_data_latent, _, y_shard_sizes_data = self._fetch_data_latent(
                y,
                dataset_name,
                fcstep,
                batch_ens_size,
                grid_shard_sizes,
                model_comm_group,
            )

            shard_sizes_target_data_dict[dataset_name] = y_shard_sizes_data

            (
                encoder_edge_attr,
                encoder_edge_index,
                enc_edge_shard_sizes,
            ) = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_target_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            y_data_latent, y_latent_target = self.ema_target_encoder(
                encoder_name=encoder_name,
                x_data_latent=x_data_latent,
                x_hidden_latent=x_hidden_latent,
                batch_ens_size=batch_ens_size,
                enc_shard_info=enc_shard_info,
                encoder_edge_attr=encoder_edge_attr,
                encoder_edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
            )

            y_data_latent_dict[dataset_name] = y_data_latent
            target_latents[dataset_name] = y_latent_target

        x_latent = sum(dataset_latents.values())
        
        if self.noise_injector is not None:
            x_latent_proc, latent_noise = self.inject_noise(
                x_latent=x_latent,
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                model_comm_group=model_comm_group,
            )
        else:
            x_latent_proc = x_latent
        
        (
            processor_edge_attr,
            processor_edge_index,
            proc_edge_shard_sizes,
        ) = self.processor_graph_provider.get_edges(
            batch_size=batch_ens_size,
            model_comm_group=model_comm_group,
        )
        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        # Processor
        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=batch_ens_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        if self.latent_skip:
            x_latent = x_latent_proc + x_latent



        y_latent_target = sum(target_latents.values())