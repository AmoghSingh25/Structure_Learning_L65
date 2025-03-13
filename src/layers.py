import torch as th
from torch import nn

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity

def concat_in_edges(edges):
    return {"m": th.cat([edges.src["h"], edges.dst["h"]], dim=1)}

class RNF_GATConv(nn.Module):
    def __init__(
        self,
        in_rnf_feats,
        intermediate_rnf_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(RNF_GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_rnf_feats)
        self._out_feats = intermediate_rnf_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.fc_attn = nn.Linear(
            self._in_src_feats, intermediate_rnf_feats * num_heads, bias=False
        )
            
        self.fc_tanh = nn.Linear(
            self._in_src_feats, intermediate_rnf_feats, bias=False
        )

        self.fc_project_attn = nn.Linear(
            2*intermediate_rnf_feats, 1, bias=False
        )

        self.fc_project_tanh = nn.Linear(
            2*intermediate_rnf_feats, 1, bias=False
        )

        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, intermediate_rnf_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, intermediate_rnf_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != intermediate_rnf_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * intermediate_rnf_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * intermediate_rnf_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_attn.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_tanh.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, target, edge_weight=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = h_dst = self.feat_drop(feat)


            #### Calculating attention coefficients ####
            feat_src_attn = feat_dst_attn = self.fc_attn(h_src).view(
                *src_prefix_shape, self._num_heads, self._out_feats
            )

            el = (feat_src_attn * self.attn_l)
            er = (feat_dst_attn * self.attn_r)

            src_indices, dst_indices = graph.edges()

            projected_pre_attention = self.fc_project_attn(self.leaky_relu(th.cat([el[src_indices], er[dst_indices]], dim=2)))
            a = th.sum(self.attn_drop(edge_softmax(graph, projected_pre_attention)), dim=1)

            #### Calculating tanh coefficients ####
            feat_src_tanh = feat_dst_tanh = self.fc_tanh(h_src)
            tanh = th.tanh(10000*self.fc_project_tanh(th.cat([feat_src_tanh[src_indices], feat_dst_tanh[dst_indices]], dim=1)))

            #### Calculating final coefficients ####
            graph.edata['coef'] = a * tanh

            #### Assigning original features ####
            graph.srcdata.update({"upd_ft": target})

            # message passing
            graph.update_all(fn.u_mul_e("upd_ft", "coef", "m"), fn.sum("m", "upd_ft"))
            rst = graph.dstdata["upd_ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, a
            else:
                return rst