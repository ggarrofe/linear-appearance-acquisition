
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../')
sys.path.append('drive/Othercomputers/MacBookPro/')
import utils.embedder as emb
import utils.utils as utils
import gc
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_ch, out_ch, hidden_ch, weight_norm=True) -> None:
        super(MLP, self).__init__()
        in_dim = input_ch

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
        print("layers", self.layers)

    def forward(self, X):    
        return self.layers(X)

class MLP_skips(nn.Module):

    def __init__(self, input_ch, hidden_ch, skips=[], activations=None, bias=True):
        super(MLP_skips, self).__init__()

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
    def __init__(self, input_ch, out_ch, hidden_ch, weight_norm=True, embed_fn=None) -> None:
        super(SurfaceRenderingNetwork, self).__init__()

        self.embed_fn = embed_fn

        layers = []
        for hidden_dim in hidden_ch:
            layers.append(torch.nn.Linear(input_ch, hidden_dim) if not weight_norm else nn.utils.weight_norm(torch.nn.Linear(input_ch, hidden_dim)))
            layers.append(nn.ReLU())
            
            input_ch = hidden_dim

        layers.append(torch.nn.Linear(input_ch, out_ch) if not weight_norm else nn.utils.weight_norm(torch.nn.Linear(input_ch, out_ch)))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, points, normals, view_dirs):
        
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            view_dirs = self.embed_fn(view_dirs)
            
        return self.layers(torch.cat([points, view_dirs, normals], dim=-1))

    def initialize_first_layer(self, linear_mapping):
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mapping, require_grads=False)

    def first_layer(self, points, normals, view_dirs):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            view_dirs = self.embed_fn(view_dirs)

        return self.layers[0](torch.cat([points, view_dirs, normals], dim=-1))

class VoxelisedLinearNetwork(nn.Module):
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(VoxelisedLinearNetwork, self).__init__()

        # linear_mappings: n_voxels x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings)

        self.embed_fn, self.input_ch = emb.get_posenc_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.num_freqs = num_freqs

    def forward(self, X, row_ids, voxel_ids):
        encoded_X = self.embed_fn(X)
        rgb_clusters = self.linear_net(encoded_X)
        
        rgb = torch.zeros((X.shape[0], 3)).to(X)
        col_indices = torch.stack([3*voxel_ids, 3*voxel_ids+1, 3*voxel_ids+2])
        rgb[row_ids] = rgb_clusters[row_ids, col_indices].T

        return rgb

class ClusterisedLinearNetwork(nn.Module):
    def __init__(self, linear_mappings, **kwargs):
        super(ClusterisedLinearNetwork, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings)

        self.embed_fn = kwargs['embed_fn']
        self.num_freqs = kwargs['num_freqs']
        self.kwargs = kwargs

    def forward(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        rgb_clusters = self.linear_net(encoded_X)

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb

    def specular(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        pos_boundaries = (0, 3*2*self.num_freqs) if 'pos_boundaries' not in self.kwargs else self.kwargs['pos_boundaries']
        spec_boundaries = (4*2*self.num_freqs, encoded_X.shape[-1]) if 'spec_boundaries' not in self.kwargs else self.kwargs['spec_boundaries']
        print("Spec boundaries")
        print("x shape", encoded_X.shape)
        print("pos", pos_boundaries)
        print("spec", spec_boundaries)
        print("boundaries in kwarfs", 'pos_boundaries' in self.kwargs)
        linear_mapping_spec = self.linear_net.weight[..., spec_boundaries[0]:spec_boundaries[1]]

        specular = encoded_X[..., spec_boundaries[0]:spec_boundaries[1]] @ linear_mapping_spec.T

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return specular[row_indices, col_indices].T

    def diffuse(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        pos_boundaries = (0, 3*2*self.num_freqs) if 'pos_boundaries' not in self.kwargs else self.kwargs['pos_boundaries']
        diff_boundaries = (3*2*self.num_freqs, 4*2*self.num_freqs) if 'diff_boundaries' not in self.kwargs is None else self.kwargs['diff_boundaries']
        print("Spec boundaries")
        print("x shape", encoded_X.shape)
        print("pos", pos_boundaries)
        print("diff", diff_boundaries)
        print("boundaries in kwargs", 'pos_boundaries' in self.kwargs)
        linear_mapping_diff = self.linear_net.weight[..., diff_boundaries[0]:diff_boundaries[1]]

        diffuse = encoded_X[..., diff_boundaries[0]:diff_boundaries[1]] @ linear_mapping_diff.T
        
        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return diffuse[row_indices, col_indices].T

    def linear(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        pos_boundaries = (0, 3*2*self.num_freqs) if 'pos_boundaries' not in self.kwargs else self.kwargs['pos_boundaries']
        
        linear = encoded_X[..., pos_boundaries[0]:pos_boundaries[1]] @ self.linear_net.weight[..., pos_boundaries[0]:pos_boundaries[1]].T

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return linear[row_indices, col_indices].T

    def specular_component(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        NdotH_pos = 4*2*self.num_freqs
        encoded_X[..., :NdotH_pos] = 0.

        rgb_clusters = self.linear_net(encoded_X)

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return rgb_clusters[row_indices, col_indices].T
    
    def diffuse_component(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        X_pos = 3*2*self.num_freqs
        NdotL_pos = 4*2*self.num_freqs
        encoded_X[..., :X_pos] = 0.
        encoded_X[..., NdotL_pos:] = 0.

        rgb_clusters = self.linear_net(encoded_X)

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return rgb_clusters[row_indices, col_indices].T

    def ambient_component(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        X_pos = 3*2*self.num_freqs
        encoded_X[..., X_pos:] = 0.

        rgb_clusters = self.linear_net(encoded_X)

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return rgb_clusters[row_indices, col_indices].T


class ClusterisedSelfAttentionLinearNetwork(nn.Module):
    def __init__(self, linear_mappings, **kwargs):
        super(ClusterisedSelfAttentionLinearNetwork, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings)

        self.embed_fn = kwargs['embed_fn']
        self.num_freqs = kwargs['num_freqs']
        self.kwargs = kwargs

    def forward(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        rgb_clusters = self.linear_net(encoded_X)
        print("rgbclusters", rgb_clusters.shape)
        print("centroids", self.kwargs['centroids'].shape)
        print("xyz T", X[..., :3].T.shape)
        scores = self.kwargs['centroids'] @ X[..., :3].T
        print("scores", scores.shape)
        attention_weights = F.softmax(scores, dim=-1)
        print("attention_weights", attention_weights.shape)
        rgb_clusters = torch.reshape(rgb_clusters, [3, -1]) # this reshape may be wrong maybe needs to be transposed before
        print("rgb reshape", rgb_clusters.shape)
        rgb = attention_weights @ rgb_clusters 
        print("rgb out", rgb.shape)
        #row_indices = torch.arange(encoded_X.shape[0])
        #col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        #rgb = rgb_clusters[row_indices, cluster_ids].T
        return rgb

    def specular(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        pos_boundaries = (0, 3*2*self.num_freqs) if 'pos_boundaries' not in self.kwargs else self.kwargs['pos_boundaries']
        spec_boundaries = (4*2*self.num_freqs, encoded_X.shape[-1]) if 'spec_boundaries' not in self.kwargs else self.kwargs['spec_boundaries']
        linear_mapping_spec = self.linear_net.weight[..., spec_boundaries[0]:spec_boundaries[1]]

        specular = encoded_X[..., spec_boundaries[0]:spec_boundaries[1]] @ linear_mapping_spec.T

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return specular[row_indices, col_indices].T

    def diffuse(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        pos_boundaries = (0, 3*2*self.num_freqs) if 'pos_boundaries' not in self.kwargs else self.kwargs['pos_boundaries']
        diff_boundaries = (3*2*self.num_freqs, 4*2*self.num_freqs) if 'diff_boundaries' not in self.kwargs is None else self.kwargs['diff_boundaries']
        linear_mapping_diff = self.linear_net.weight[..., diff_boundaries[0]:diff_boundaries[1]]

        diffuse = encoded_X[..., diff_boundaries[0]:diff_boundaries[1]] @ linear_mapping_diff.T
        
        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return diffuse[row_indices, col_indices].T

    def linear(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        pos_boundaries = (0, 3*2*self.num_freqs) if 'pos_boundaries' not in self.kwargs else self.kwargs['pos_boundaries']
        
        linear = encoded_X[..., pos_boundaries[0]:pos_boundaries[1]] @ self.linear_net.weight[..., pos_boundaries[0]:pos_boundaries[1]].T

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return linear[row_indices, col_indices].T


class EnhancedReflectanceNetwork(nn.Module):
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(EnhancedReflectanceNetwork, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        self.num_clusters = linear_mappings.shape[0]
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings, requires_grad=False)
            
        self.embed_fn_input, self.input_ch_input = emb.get_posenc_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.ReLU = nn.ReLU()

        # X     \ ---------------------\
        # NdotL   > linear layer + relu  >  residual layer
        # NdotH /  
        
        # regularizer: difference between linear layer output and final output 
        self.embed_fn_pos, self.input_ch_pos = emb.get_posenc_embedder(in_dim=3, num_freqs=num_freqs)
        self.cluster_encoder = nn.Sequential(nn.Linear(in_features=self.input_ch_pos, out_features=self.num_clusters),
                                          nn.Tanh())
        self.Tanh = nn.Tanh()
        self.num_freqs = num_freqs

    def forward(self, X_NdotL_NdotH):
        encoded_input = self.embed_fn_input(X_NdotL_NdotH)
        linear_output = self.linear_net(encoded_input)
        linear_output = self.ReLU(linear_output)
        cluster_ids = self.cluster_encoder(self.embed_fn_pos(X_NdotL_NdotH[..., :3]))
        extra_layers_in = torch.cat((linear_output, ), dim=1)
        return self.extra_layers(extra_layers_in)

    def forward_encoded(self, encoded_input, encoded_pos):
        linear_output = self.linear_net(encoded_input)
        linear_output = self.ReLU(linear_output)
        extra_layers_in = torch.cat((linear_output, encoded_pos), dim=1)
        return self.extra_layers(extra_layers_in)

    def specular(self, X_NdotL_NdotH):
        encoded_input = self.embed_fn_input(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        NdotH_pos = 4*2*self.num_freqs
        encoded_input_amb = torch.cat([encoded_input[..., :X_pos], torch.zeros((encoded_input.shape[0], 2*2*self.num_freqs)).to(encoded_input)], dim=-1)
        encoded_input_spec = torch.cat([torch.zeros((encoded_input.shape[0], NdotH_pos)).to(encoded_input), encoded_input[..., NdotH_pos:]], dim=-1)
        
        amb = self.forward_encoded(encoded_input_amb, encoded_input[..., :X_pos])
        spec = self.forward_encoded(encoded_input_spec, encoded_input[..., :X_pos])

        return amb+spec

    def diffuse(self, X_NdotL_NdotH):
        encoded_input = self.embed_fn_input(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        NdotL_pos = 4*2*self.num_freqs
        encoded_input_amb = torch.cat([encoded_input[..., :X_pos], torch.zeros((encoded_input.shape[0], 2*2*self.num_freqs)).to(encoded_input)], dim=-1)
        encoded_input_diff = torch.cat([torch.zeros((encoded_input.shape[0], X_pos)).to(encoded_input), encoded_input[..., X_pos:NdotL_pos], torch.zeros((encoded_input.shape[0], 1*2*self.num_freqs)).to(encoded_input)], dim=-1)
        
        amb = self.forward_encoded(encoded_input_amb, encoded_input[..., :X_pos])
        diff = self.forward_encoded(encoded_input_diff, encoded_input[..., :X_pos])

        return amb+diff
    
    def linear(self, X_NdotL_NdotH, cluster_ids):
        encoded_input = self.embed_fn_input(X_NdotL_NdotH)
        linear = self.linear_net(encoded_input)

        row_indices = torch.arange(encoded_input.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return linear[row_indices, col_indices].T



class LinearMapping(nn.Module):
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(LinearMapping, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        self.linear_net.weight = nn.Parameter(linear_mappings)

        self.embed_fn, self.input_ch = emb.get_posenc_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.num_freqs = num_freqs

    def forward(self, encoded_X):
        return self.linear_net(encoded_X)

    def reflectance(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        rgb_clusters = self.linear_net(encoded_X)
        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb

    def specular(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        X_pos = 3*2*self.num_freqs
        NdotH_pos = 4*2*self.num_freqs
        linear_mapping_spec = torch.cat([self.linear_net.weight[..., :X_pos], 
                                         self.linear_net.weight[..., NdotH_pos:]], dim=-1)

        specular = torch.cat([encoded_X[..., :X_pos], encoded_X[..., NdotH_pos:]], dim=-1) @ linear_mapping_spec.T

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return specular[row_indices, col_indices].T

    def diffuse(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        NdotL_pos = 4*2*self.num_freqs

        diffuse = encoded_X[..., :NdotL_pos] @ self.linear_net.weight[..., :NdotL_pos].T

        row_indices = torch.arange(encoded_X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        return diffuse[row_indices, col_indices].T

class LinearAutoDecoder(nn.Module):
    def __init__(self, pos_size, latent_size, num_clusters):
        super(LinearAutoDecoder, self).__init__()
        self.pos_mapping = nn.Linear(in_features=pos_size, out_features=3*num_clusters, bias=False)
        self.feature_mapping = nn.Linear(in_features=latent_size, out_features=3*num_clusters, bias=False)
        self.latent_size = latent_size
        self.pos_size = pos_size
        
    def set_position_mapping(self, pos_mappings):
        pos_mappings = pos_mappings.reshape(-1, pos_mappings.shape[-1])
        with torch.no_grad():
            self.pos_mapping.weight = nn.Parameter(pos_mappings, requires_grad=False)

    def forward(self, X, cluster_ids, linear_mappings=None):
        # linear_mappings: n_clusters x 3 x encoding_size + latent_size
        if self.training:
            linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])
            self.feature_mapping.weight = nn.Parameter(linear_mappings[..., self.pos_size:])
            self.pos_mapping.weight = nn.Parameter(linear_mappings[..., :self.pos_size])
            #print("latent_mapping shape", self.latent_mapping.weight[:2])

        rgb_clusters = self.pos_mapping(X[..., :self.pos_size]) + self.feature_mapping(X[..., self.pos_size:])
        row_indices = torch.arange(X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb

    def learnable_parameters(self):
        return self.feature_mapping.parameters()

    @staticmethod
    def compute_linear_mappings(X, y, cluster_ids, num_clusters, device=torch.device("cuda"), gpu_limit=1e06, embed_fn=None, input_ch=None):
        if input_ch is None:
            linear_mappings = torch.zeros([num_clusters, 3, X.shape[-1]]).to(device)
        else:
            linear_mappings = torch.zeros([num_clusters, 3, input_ch]).to(device)
        tqdm._instances.clear()
        for cluster_id in tqdm(range(num_clusters), leave=False, unit="linear mapping", desc="Computing linear mappings"):
            X_cluster, y_cluster = X[cluster_ids == cluster_id], y[cluster_ids == cluster_id]

            if X_cluster.shape[0] > gpu_limit:
                X_cluster, indices = utils.filter_duplicates(X_cluster)
                y_cluster = y_cluster[indices]

            if X_cluster.shape[0] < gpu_limit:
                X_inv = torch.linalg.pinv(X_cluster.to(device) if embed_fn is None else embed_fn(X_cluster).to(device))
                linear_mappings[cluster_id] = (X_inv @ y_cluster.to(device)).T
            else:
                X_inv = torch.linalg.pinv(X_cluster if embed_fn is None else embed_fn(X_cluster))
                linear_mappings[cluster_id] = (X_inv @ y_cluster).T.to(device)

        return linear_mappings

    def position_mapping(self, X, cluster_ids):
        rgb_clusters = self.pos_mapping(X)
        row_indices = torch.arange(X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb



class LinearAutoEncoder(nn.Module):
    def __init__(self, pos_size, latent_size, num_clusters):
        super(LinearAutoEncoder, self).__init__()
        self.pos_mapping = nn.Linear(in_features=pos_size, out_features=3*num_clusters, bias=False)
        self.feature_mapping = nn.Linear(in_features=latent_size, out_features=3*num_clusters, bias=False)
        self.latent_size = latent_size
        self.pos_size = pos_size

        self.encoder = nn.Sequential(nn.Linear(pos_size, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, latent_size),
                                     nn.ReLU())
        
    def encode(self, X):
        return self.encoder(X)

    def forward(self, X, cluster_ids, linear_mappings=None):
        # linear_mappings: n_clusters x 3 x encoding_size + latent_size
        if self.training:
            linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])
            self.feature_mapping.weight = nn.Parameter(linear_mappings[..., self.pos_size:])
            self.pos_mapping.weight = nn.Parameter(linear_mappings[..., :self.pos_size])
            #print("latent_mapping shape", self.latent_mapping.weight[:2])

        rgb_clusters = self.pos_mapping(X[..., :self.pos_size]) + self.feature_mapping(X[..., self.pos_size:])
        row_indices = torch.arange(X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb

    def learnable_parameters(self):
        return self.feature_mapping.parameters()

    @staticmethod
    def compute_linear_mappings(X, y, cluster_ids, num_clusters, device=torch.device("cuda"), gpu_limit=1e06, embed_fn=None, input_ch=None):
        if input_ch is None:
            linear_mappings = torch.zeros([num_clusters, 3, X.shape[-1]]).to(device)
        else:
            linear_mappings = torch.zeros([num_clusters, 3, input_ch]).to(device)
        tqdm._instances.clear()
        for cluster_id in tqdm(range(num_clusters), leave=False, unit="linear mapping", desc="Computing linear mappings"):
            X_cluster, y_cluster = X[cluster_ids == cluster_id], y[cluster_ids == cluster_id]

            if X_cluster.shape[0] > gpu_limit:
                X_cluster, indices = utils.filter_duplicates(X_cluster)
                y_cluster = y_cluster[indices]

            if X_cluster.shape[0] < gpu_limit:
                X_inv = torch.linalg.pinv(X_cluster.to(device) if embed_fn is None else embed_fn(X_cluster).to(device))
                linear_mappings[cluster_id] = (X_inv @ y_cluster.to(device)).T
            else:
                X_inv = torch.linalg.pinv(X_cluster if embed_fn is None else embed_fn(X_cluster))
                linear_mappings[cluster_id] = (X_inv @ y_cluster).T.to(device)

        return linear_mappings

    def position_mapping(self, X, cluster_ids):
        rgb_clusters = self.pos_mapping(X)
        row_indices = torch.arange(X.shape[0])
        col_indices = torch.stack([3*cluster_ids, 3*cluster_ids+1, 3*cluster_ids+2])
        rgb = rgb_clusters[row_indices, col_indices].T
        return rgb


class ReflectanceNetwork(nn.Module):
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(ReflectanceNetwork, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        self.linear_mapping = LinearMapping(in_features, linear_mappings, num_freqs)
        

        # X     \ ---------------------\
        # NdotL   > linear layer + relu  >  residual layer
        # NdotH /  
        
        # regularizer: difference between linear layer output and final output 
        '''self.layers = nn.Sequential(self.linear_net, 
                                     nn.ReLU(), 
                                     nn.Linear(in_features=linear_mappings.shape[0], out_features=512),
                                     nn.ReLU(),
                                     nn.Linear(in_features=512, out_features=3),
                                     nn.Tanh())'''
        self.layers = nn.Sequential(self.linear_mapping, 
                                     nn.ReLU(), 
                                     nn.Linear(in_features=linear_mappings.shape[0], out_features=3),
                                     nn.Tanh())
        self.embed_fn, self.input_ch = emb.get_posenc_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.num_freqs = num_freqs

    def forward(self, X_NdotL_NdotH):
        encoded_X = self.embed_fn(X_NdotL_NdotH)
        return self.layers(encoded_X)

    def specular(self, X_NdotL_NdotH):
        encoded_X = self.embed_fn(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        NdotH_pos = 4*2*self.num_freqs
        encoded_X_amb = torch.cat([encoded_X[..., :X_pos], torch.zeros((encoded_X.shape[0], 2*2*self.num_freqs)).to(encoded_X)], dim=-1)
        encoded_X_spec = torch.cat([torch.zeros((encoded_X.shape[0], NdotH_pos)).to(encoded_X), encoded_X[..., NdotH_pos:]], dim=-1)
        
        amb = self.layers(encoded_X_amb)
        spec = self.layers(encoded_X_spec)

        return amb+spec

    def diffuse(self, X_NdotL_NdotH):
        encoded_X = self.embed_fn(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        NdotL_pos = 4*2*self.num_freqs
        encoded_X_amb = torch.cat([encoded_X[..., :X_pos], torch.zeros((encoded_X.shape[0], 2*2*self.num_freqs)).to(encoded_X)], dim=-1)
        encoded_X_diff = torch.cat([torch.zeros((encoded_X.shape[0], X_pos)).to(encoded_X), encoded_X[..., X_pos:NdotL_pos], torch.zeros((encoded_X.shape[0], 1*2*self.num_freqs)).to(encoded_X)], dim=-1)
        
        amb = self.layers(encoded_X_amb)
        diff = self.layers(encoded_X_diff)

        return amb+diff

class ClusterizedReflectance(nn.Module):
    def __init__(self, in_features, linear_mappings, x2cluster, num_freqs):
        super(ClusterizedReflectance, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])
        self.linear_mapping = LinearMapping(in_features, linear_mappings, num_freqs)

        self.num_clusters = x2cluster.shape[0]
        self.x2cluster_net = nn.Linear(in_features=3, out_features=self.num_clusters, bias=False)
        with torch.no_grad():
            self.x2cluster_net.weight = nn.Parameter(x2cluster)
            
        self.embed_fn, self.input_ch = emb.get_posenc_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.num_freqs = num_freqs
        

    def forward(self, X_NdotL_NdotH):
        encoded_X = self.embed_fn(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        cluster_selection = self.x2cluster_net(encoded_X[..., :X_pos]).unsqueeze(-1)

        rgb_clusters = self.linear_mapping(encoded_X)
        rgb_clusters = rgb_clusters.view((-1, self.num_clusters, 3)).transpose(1, 2)

        rgb = rgb_clusters @ cluster_selection
        
        return rgb.squeeze()

    def specular(self, X_NdotL_NdotH):
        encoded_X = self.embed_fn(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        NdotH_pos = 4*2*self.num_freqs
        encoded_X_amb = torch.cat([encoded_X[..., :X_pos], torch.zeros((encoded_X.shape[0], 2*2*self.num_freqs)).to(encoded_X)], dim=-1)
        encoded_X_spec = torch.cat([torch.zeros((encoded_X.shape[0], NdotH_pos)).to(encoded_X), encoded_X[..., NdotH_pos:]], dim=-1)
        
        cluster_selection = self.x2cluster_net(encoded_X[..., :X_pos]).unsqueeze(-1)

        amb_clusters = self.linear_mapping(encoded_X_amb)
        amb = amb_clusters.view((-1, self.num_clusters, 3)).transpose(1, 2) @ cluster_selection
        spec_clusters = self.linear_mapping(encoded_X_spec)
        spec = spec_clusters.view((-1, self.num_clusters, 3)).transpose(1, 2) @ cluster_selection

        return amb.squeeze()+spec.squeeze()

    def diffuse(self, X_NdotL_NdotH):
        encoded_X = self.embed_fn(X_NdotL_NdotH)
        X_pos = 3*2*self.num_freqs
        NdotL_pos = 4*2*self.num_freqs
        encoded_X_amb = torch.cat([encoded_X[..., :X_pos], torch.zeros((encoded_X.shape[0], 2*2*self.num_freqs)).to(encoded_X)], dim=-1)
        encoded_X_diff = torch.cat([torch.zeros((encoded_X.shape[0], X_pos)).to(encoded_X), encoded_X[..., X_pos:NdotL_pos], torch.zeros((encoded_X.shape[0], 1*2*self.num_freqs)).to(encoded_X)], dim=-1)
        
        cluster_selection = self.x2cluster_net(encoded_X[..., :X_pos]).unsqueeze(-1)

        amb_clusters = self.linear_mapping(encoded_X_amb)
        amb = amb_clusters.view((-1, self.num_clusters, 3)).transpose(1, 2) @ cluster_selection
        diff_clusters = self.linear_mapping(encoded_X_diff)
        diff = diff_clusters.view((-1, self.num_clusters, 3)).transpose(1, 2) @ cluster_selection

        return amb.squeeze()+diff.squeeze()

class ResidualReflectance(nn.Module):
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(ResidualReflectance, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        self.linear_mapping = LinearMapping(in_features, linear_mappings, num_freqs)
        self.embed_fn, self.input_ch = emb.get_posenc_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.num_freqs = num_freqs

        self.residual_net = MLP(input_ch=self.input_ch, #X (3) NdotL (1) NdotH (1)
                                out_ch=3,
                                hidden_ch=[512, 512, 512, 512])
        

    def forward(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        return self.linear_mapping.reflectance(X, cluster_ids) + self.residual_net(encoded_X)

    def specular(self, X, cluster_ids):

        encoded_X = self.embed_fn(X)
        X_pos = 3*2*self.num_freqs
        NdotH_pos = 4*2*self.num_freqs
        encoded_X_amb = torch.cat([encoded_X[..., :X_pos], torch.zeros((encoded_X.shape[0], 2*2*self.num_freqs)).to(encoded_X)], dim=-1)
        encoded_X_spec = torch.cat([torch.zeros((encoded_X.shape[0], NdotH_pos)).to(encoded_X), encoded_X[..., NdotH_pos:]], dim=-1)
        
        amb = self.residual_net(encoded_X_amb)
        spec = self.residual_net(encoded_X_spec)

        return amb+spec + self.linear_mapping.specular(X, cluster_ids)

    def diffuse(self, X, cluster_ids):
        encoded_X = self.embed_fn(X)
        X_pos = 3*2*self.num_freqs
        NdotL_pos = 4*2*self.num_freqs
        encoded_X_amb = torch.cat([encoded_X[..., :X_pos], torch.zeros((encoded_X.shape[0], 2*2*self.num_freqs)).to(encoded_X)], dim=-1)
        encoded_X_diff = torch.cat([torch.zeros((encoded_X.shape[0], X_pos)).to(encoded_X), encoded_X[..., X_pos:NdotL_pos], torch.zeros((encoded_X.shape[0], 1*2*self.num_freqs)).to(encoded_X)], dim=-1)
        
        amb = self.residual_net(encoded_X_amb)
        diff = self.residual_net(encoded_X_diff)

        return amb+diff + self.linear_mapping.diffuse(X, cluster_ids)