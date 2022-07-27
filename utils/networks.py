
import torch
import torch.nn as nn

import sys
sys.path.append('../')
sys.path.append('drive/Othercomputers/MacBookPro/')
import utils.embedder as emb
import utils.utils as utils

class MLP(nn.Module):

    def __init__(self, input_ch, hidden_ch, skips=[], activations=None, bias=True):
        super(MLP, self).__init__()

        layers = []
        in_dim = input_ch
        i = 0
        for hidden_dim, activation in zip(hidden_ch[:-1], activations):
            if i in skips:
                layers.append(torch.nn.Linear(in_dim+input_ch, hidden_dim, bias=bias))
            else:
                layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            
            if isinstance(activation, str):
                activation = getattr(nn, activation)()

            layers.append(activation)
            
            in_dim = hidden_dim
            i += 1

        layers.append(torch.nn.Linear(in_dim, hidden_ch[-1], bias=bias))
        self.skips = skips


    def forward(self, x):
        x_ = x
        for i, layer in enumerate(self.layers):
            y = layer(x_)
            if i in self.skips:
                y = torch.cat((y, x), -1)
            x_ = y
        
        return y


class SurfaceRenderingNetwork(nn.Module):
    def __init__(self, input_ch, out_ch, hidden_ch, weight_norm=True, enc_dim=0) -> None:
        super(SurfaceRenderingNetwork, self).__init__()

        in_dim = input_ch # points + normals + view_dir

        self.embed_fn = None
        if enc_dim > 0:
            embed_fn, input_ch = emb.get_embedder(in_dim=3, num_freqs=enc_dim)
            self.embed_fn = embed_fn
            in_dim = in_dim + 2*input_ch - 6 

        layers = []
        for hidden_dim in hidden_ch:
            
            lin = torch.nn.Linear(in_dim, hidden_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            layers.append(lin)
            layers.append(nn.ReLU())
            
            in_dim = hidden_dim

        lin = torch.nn.Linear(in_dim, out_ch)

        if weight_norm:
            lin = nn.utils.weight_norm(lin)

        layers.append(lin)
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, points, normals, view_dirs):
        
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            view_dirs = self.embed_fn(view_dirs)
            
        x = torch.cat([points, view_dirs, normals], dim=-1)
        for layer in self.layers:
            y = layer(x)
            x = y

        return y

class LinearNetwork(nn.Module):
    def __init__(self, linear_mappings, embed_fn):
        super(LinearNetwork, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings)

        self.embed_fn = embed_fn

    def forward(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        rgb_clusters = self.linear_net(encoded_X)

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb