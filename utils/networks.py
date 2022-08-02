
import torch
import torch.nn as nn

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
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(LinearNetwork, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings)

        self.embed_fn, self.input_ch = emb.get_embedder(in_dim=in_features, num_freqs=num_freqs)
        self.num_freqs = num_freqs

    def forward(self, X, cluster_ids):
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

class LinearMapping(nn.Module):
    def __init__(self, in_features, linear_mappings, num_freqs):
        super(LinearMapping, self).__init__()

        # linear_mappings: n_clusters x 3 x encoding_size
        linear_mappings = linear_mappings.reshape(-1, linear_mappings.shape[-1])

        self.linear_net = nn.Linear(in_features=linear_mappings.shape[-1], out_features=linear_mappings.shape[0], bias=False)
        with torch.no_grad():
            self.linear_net.weight = nn.Parameter(linear_mappings, requires_grad=False)

        self.embed_fn, self.input_ch = emb.get_embedder(in_dim=in_features, num_freqs=num_freqs)
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
    def __init__(self, in_features, latent_size, num_clusters):
        super(LinearAutoDecoder, self).__init__()
        self.pos_mapping = nn.Linear(in_features=in_features, out_features=3*num_clusters, bias=False)
        self.feature_mapping = nn.Linear(in_features=latent_size, out_features=3*num_clusters, bias=False)
        self.latent_size = latent_size
        
    def set_position_mapping(self, pos_mappings):
        pos_mappings = pos_mappings.reshape(-1, pos_mappings.shape[-1])
        with torch.no_grad():
            self.pos_mapping.weight = nn.Parameter(pos_mappings, requires_grad=False)

    def forward(self, X, cluster_ids, feat_mappings=None):
        # linear_mappings: n_clusters x 3 x encoding_size + latent_size
        if self.training:
            feat_mappings = feat_mappings.reshape(-1, feat_mappings.shape[-1])
            self.feature_mapping.weight = nn.Parameter(feat_mappings)
            #print("latent_mapping shape", self.latent_mapping.weight[:2])

        rgb_clusters = self.pos_mapping(X[..., :-self.latent_size]) + self.feature_mapping(X[..., -self.latent_size:])
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
        self.embed_fn, self.input_ch = emb.get_embedder(in_dim=in_features, num_freqs=num_freqs)
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
            
        self.embed_fn, self.input_ch = emb.get_embedder(in_dim=in_features, num_freqs=num_freqs)
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
        self.embed_fn, self.input_ch = emb.get_embedder(in_dim=in_features, num_freqs=num_freqs)
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