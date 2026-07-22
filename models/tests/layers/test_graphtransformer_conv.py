import torch

from anemoi.models.layers.conv import GraphTransformerConv


def test_graph_transformer_message_expands_edge_attr_over_heads() -> None:
    conv = GraphTransformerConv(out_channels=4)

    query_i = torch.randn(15, 2, 4)
    key_j = torch.randn(15, 2, 4)
    value_j = torch.randn(15, 2, 4)
    edge_attr = torch.randn(15, 4)

    out = conv.message(
        heads=2,
        query_i=query_i,
        key_j=key_j,
        value_j=value_j,
        edge_attr=edge_attr,
        index=torch.arange(15),
        ptr=None,
        size_i=15,
    )

    assert out.shape == (15, 2, 4)
    assert torch.isfinite(out).all()
