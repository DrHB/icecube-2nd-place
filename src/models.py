import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, trunc_normal_
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.pool import knn_graph
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily
from torch_geometric.typing import Adj
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torch import Tensor, LongTensor
from torch_geometric.nn import EdgeConv
from graphnet.utilities.config import save_model_config

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


# BEiTv2 block
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Attention_rel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, rel_pos_bias=None, key_padding_mask=None):
        # rel_pos_bias: B L L C/h
        # key_padding_mask - float with -inf
        B, N, C = q.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #    qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = F.linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = F.linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(B, k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = F.linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(B, v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            bias = torch.min(key_padding_mask[:, None, :], key_padding_mask[:, :, None])
            bias[
                torch.max(key_padding_mask[:, None, :], key_padding_mask[:, :, None])
                < 0
            ] = 0
            # print(bias.shape,bias.min(),bias.max())
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# BEiTv2 block
class Block_rel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_rel(
            dim, num_heads, attn_drop=attn_drop, qkv_bias=qkv_bias
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, key_padding_mask=None, rel_pos_bias=None, kv=None):
        if self.gamma_1 is None:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    kv,
                    kv,
                    rel_pos_bias=rel_pos_bias,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.gamma_1
                * self.drop_path(
                    self.attn(
                        xn,
                        kv,
                        kv,
                        rel_pos_bias=rel_pos_bias,
                        key_padding_mask=key_padding_mask,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Extractor(nn.Module):
    def __init__(self, dim_base=128, dim=384):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        self.aux_emb = nn.Embedding(2, dim_base // 2)
        self.emb2 = SinusoidalPosEmb(dim=dim_base // 2)
        self.proj = nn.Sequential(
            nn.Linear(6 * dim_base, 6 * dim_base),
            nn.LayerNorm(6 * dim_base),
            nn.GELU(),
            nn.Linear(6 * dim_base, dim),
        )

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        charge = x["charge"] if Lmax is None else x["charge"][:, :Lmax]
        time = x["time"] if Lmax is None else x["time"][:, :Lmax]
        auxiliary = x["auxiliary"] if Lmax is None else x["auxiliary"][:, :Lmax]
        qe = x["qe"] if Lmax is None else x["qe"][:, :Lmax]
        ice_properties = (
            x["ice_properties"] if Lmax is None else x["ice_properties"][:, :Lmax]
        )
        length = torch.log10(x["L0"].to(dtype=pos.dtype))

        x = torch.cat(
            [
                self.emb(4096 * pos).flatten(-2),
                self.emb(1024 * charge),
                self.emb(4096 * time),
                self.aux_emb(auxiliary),
                self.emb2(length).unsqueeze(1).expand(-1, pos.shape[1], -1),
            ],
            -1,
        )
        x = self.proj(x)
        return x


class Rel_ds(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        time = x["time"] if Lmax is None else x["time"][:, :Lmax]
        ds2 = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1) - (
            (time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)
        ).pow(2)
        d = torch.sign(ds2) * torch.sqrt(torch.abs(ds2))
        emb = self.emb(1024 * d.clip(-4, 4))
        rel_attn = self.proj(emb)
        return rel_attn, emb


def get_nbs(x, Lmax=None, K=8):
    pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
    mask = x["mask"][:, :Lmax]
    B = pos.shape[0]

    d = -torch.cdist(pos, pos, p=2)
    d -= 100 * (~torch.min(mask[:, None, :], mask[:, :, None]))
    d -= 200 * torch.eye(Lmax, dtype=pos.dtype, device=pos.device).unsqueeze(0)
    nbs = d.topk(K - 1, dim=-1)[1]
    nbs = torch.cat(
        [
            torch.arange(Lmax, dtype=nbs.dtype, device=nbs.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B, -1, -1),
            nbs,
        ],
        -1,
    )
    return nbs


class LocalBlock(nn.Module):
    def __init__(
        self,
        dim=192,
        num_heads=192 // 64,
        mlp_ratio=4,
        drop_path=0,
        init_values=1,
        **kwargs,
    ):
        super().__init__()
        self.proj_rel_bias = nn.Linear(dim // num_heads, dim // num_heads)
        self.block = Block_rel(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            init_values=init_values,
        )

    def forward(self, x, nbs, key_padding_mask=None, rel_pos_bias=None):
        B, Lmax, C = x.shape
        mask = (
            key_padding_mask
            if not (key_padding_mask is None)
            else torch.ones(B, Lmax, dtype=torch.bool, device=x.device)
        )

        m = torch.gather(mask.unsqueeze(1).expand(-1, Lmax, -1), 2, nbs)
        attn_mask = torch.zeros(m.shape, device=m.device)
        attn_mask[~mask] = -torch.inf
        attn_mask = attn_mask[mask]

        if rel_pos_bias is not None:
            rel_pos_bias = torch.gather(
                rel_pos_bias,
                2,
                nbs.unsqueeze(-1).expand(-1, -1, -1, rel_pos_bias.shape[-1]),
            )
            rel_pos_bias = rel_pos_bias[mask]
            rel_pos_bias = self.proj_rel_bias(rel_pos_bias).unsqueeze(1)

        xl = torch.gather(
            x.unsqueeze(1).expand(-1, Lmax, -1, -1),
            2,
            nbs.unsqueeze(-1).expand(-1, -1, -1, C),
        )
        xl = xl[mask]
        # modify only the node (0th element)
        # print(xl[:,:1].shape,rel_pos_bias.shape,attn_mask[:,:1].shape,xl.shape)
        xl = self.block(
            xl[:, :1],
            rel_pos_bias=rel_pos_bias,
            key_padding_mask=attn_mask[:, :1],
            kv=xl,
        )
        x = torch.zeros(x.shape, device=x.device, dtype=xl.dtype)
        x[mask] = xl.squeeze(1)
        return x


class DeepIceModel(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=12,
        use_checkpoint=False,
        head_size=32,
        depth_rel=4,
        n_rel=1,
        **kwargs,
    ):
        super().__init__()
        self.extractor = Extractor(dim_base, dim)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [Block_rel(dim=dim, num_heads=dim // head_size) for i in range(depth_rel)]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)
        self.n_rel = n_rel

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            if isinstance(blk, LocalBlock):
                x = blk(x, nbs, mask, rel_enc)
            else:
                x = blk(x, attn_mask, rel_pos_bias)
                if i + 1 == self.n_rel:
                    rel_pos_bias = None

        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        x = self.proj_out(x[:, 0])  # cls token
        return x


GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdgeConv(EdgeConv):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.
        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:

        """Forward pass."""
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)
        dev = x.device

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(dev)

        return x, edge_index


class DynEdgeFEXTRACTRO(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
    ):
        """Construct `DynEdge`.
        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
            ]

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes)

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = add_global_variables_after_pooling

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.GELU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(torch.nn.Linear(nb_in, nb_out))
                layers.append(nn.LayerNorm(nb_out))
                layers.append(self._activation)

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="add",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Post-processing operations
        nb_latent_features = (
            sum(sizes[-1] for sizes in self._dynedge_layer_sizes) + nb_input_features
        )

        post_processing_layers = []
        layer_sizes = [nb_latent_features] + list(self._post_processing_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
            post_processing_layers.append(nn.LayerNorm(nb_out))
            post_processing_layers.append(self._activation)

        self._post_processing = torch.nn.Sequential(*post_processing_layers)

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes) if self._global_pooling_schemes else 1
        )
        nb_latent_features = nb_out * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(nn.LayerNorm(nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, x, edge_index, batch, n_pulses) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        global_variables = self._calculate_global_variables(
            x,
            edge_index,
            batch,
            torch.log10(n_pulses),
        )

        # Distribute global variables out to each node
        if not self._add_global_variables_after_pooling:
            distribute = (
                batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
            ).type(torch.float)

            global_variables_distributed = torch.sum(
                distribute.unsqueeze(dim=2) * global_variables.unsqueeze(dim=0),
                dim=1,
            )

            x = torch.cat((x, global_variables_distributed), dim=1)

        # DynEdge-convolutions
        skip_connections = [x]
        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        # Skip-cat
        x = torch.cat(skip_connections, dim=1)
        x = self._post_processing(x)
        return x, edge_index, batch


class ExtractorV0(nn.Module):
    def __init__(self, dim_base=128, dim=384, proj=True):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        self.emb2 = SinusoidalPosEmb(dim=dim_base // 2)
        self.aux_emb = nn.Embedding(2, dim_base // 2)
        self.qe_emb = nn.Embedding(2, dim_base // 2)
        self.proj = nn.Linear(dim_base * 7, dim) if proj else nn.Identity()

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        charge = x["charge"] if Lmax is None else x["charge"][:, :Lmax]
        time = x["time"] if Lmax is None else x["time"][:, :Lmax]
        auxiliary = x["aux"] if Lmax is None else x["auxiliary"][:, :Lmax]
        qe = x["qe"] if Lmax is None else x["qe"][:, :Lmax]
        ice_properties = (
            x["ice_properties"] if Lmax is None else x["ice_properties"][:, :Lmax]
        )

        x = torch.cat(
            [
                self.emb(100 * pos).flatten(-2),
                self.emb(40 * charge),
                self.emb(100 * time),
                self.aux_emb(auxiliary),
                self.qe_emb(qe),
                self.emb2(50 * ice_properties).flatten(-2),
            ],
            -1,
        )
        x = self.proj(x)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim=32, M=10000):
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class ExtractorV11Scaled(nn.Module):
    def __init__(self, dim_base=128, dim=384):
        super().__init__()
        self.pos = ScaledSinusoidalEmbedding(dim=dim_base)
        self.emb_charge = ScaledSinusoidalEmbedding(dim=dim_base)
        self.time = ScaledSinusoidalEmbedding(dim=dim_base)
        self.aux_emb = nn.Embedding(2, dim_base // 2)
        self.emb2 = ScaledSinusoidalEmbedding(dim=dim_base // 2)
        self.proj = nn.Sequential(
            nn.Linear(6 * dim_base, 6 * dim_base),
            nn.LayerNorm(6 * dim_base),
            nn.GELU(),
            nn.Linear(6 * dim_base, dim),
        )

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        charge = x["charge"] if Lmax is None else x["charge"][:, :Lmax]
        time = x["time"] if Lmax is None else x["time"][:, :Lmax]
        auxiliary = x["auxiliary"] if Lmax is None else x["auxiliary"][:, :Lmax]
        length = torch.log10(x["L0"].to(dtype=pos.dtype))

        x = torch.cat(
            [
                self.pos(4096 * pos).flatten(-2),
                self.emb_charge(1024 * charge),
                self.time(4096 * time),
                self.aux_emb(auxiliary),
                self.emb2(length).unsqueeze(1).expand(-1, pos.shape[1], -1),
            ],
            -1,
        )
        x = self.proj(x)
        return x


class EncoderWithDirectionReconstructionV22(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=8,
        use_checkpoint=False,
        head_size=64,
        **kwargs,
    ):
        super().__init__()
        self.extractor = ExtractorV11Scaled(dim_base, dim // 2)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
            ]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.local_root = DynEdgeFEXTRACTRO(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        )
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0["mask"]
        graph_featutre = torch.concat(
            [
                x0["pos"][mask],
                x0["time"][mask].view(-1, 1),
                x0["auxiliary"][mask].view(-1, 1),
                x0["qe"][mask].view(-1, 1),
                x0["charge"][mask].view(-1, 1),
                x0["ice_properties"][mask],
            ],
            dim=1,
        )
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        batch_index = mask.nonzero()[:, 0]
        edge_index = knn_graph(x=graph_featutre[:, :3], k=8, batch=batch_index).to(
            mask.device
        )
        graph_featutre, _, _ = self.local_root(
            graph_featutre, edge_index, batch_index, x0["L0"]
        )
        graph_featutre, _ = to_dense_batch(graph_featutre, batch_index)

        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        x = torch.cat([x, graph_featutre], 2)

        for blk in self.sandwich:
            if isinstance(blk, LocalBlock):
                x = blk(x, nbs, mask, rel_enc)
            else:
                x = blk(x, attn_mask, rel_pos_bias)
                rel_pos_bias = None
        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        x = self.proj_out(x[:, 0])  # cls token
        return x


class EncoderWithDirectionReconstructionV23(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=8,
        use_checkpoint=False,
        head_size=64,
        **kwargs,
    ):
        super().__init__()
        self.extractor = ExtractorV11Scaled(dim_base, dim // 2)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
            ]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.local_root = DynEdgeFEXTRACTRO(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        )
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0["mask"]
        graph_featutre = torch.concat(
            [
                x0["pos"][mask],
                x0["time"][mask].view(-1, 1),
                x0["auxiliary"][mask].view(-1, 1),
                x0["qe"][mask].view(-1, 1),
                x0["charge"][mask].view(-1, 1),
                x0["ice_properties"][mask],
            ],
            dim=1,
        )
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        batch_index = mask.nonzero()[:, 0]
        edge_index = knn_graph(x=graph_featutre[:, :4], k=8, batch=batch_index).to(
            mask.device
        )
        graph_featutre, _, _ = self.local_root(
            graph_featutre, edge_index, batch_index, x0["L0"]
        )
        graph_featutre, _ = to_dense_batch(graph_featutre, batch_index)

        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        x = torch.cat([x, graph_featutre], 2)

        for blk in self.sandwich:
            if isinstance(blk, LocalBlock):
                x = blk(x, nbs, mask, rel_enc)
            else:
                x = blk(x, attn_mask, rel_pos_bias)
                # rel_pos_bias = None
        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        x = self.proj_out(x[:, 0])  # cls token
        return x
