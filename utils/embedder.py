import torch

class Embedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_positional_encoding()
        
    def create_positional_encoding(self):
        embed_fns = []
        in_dim = self.kwargs['in_dim']
        out_dim = 0

        max_freq = self.kwargs['num_freqs']-1
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        enc = [input]
        for f in freq_bands:
            for fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, fn=fn, freq=f: fn(x * freq))
                out_dim += in_dim

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(in_dim=3, num_freqs=6):
    embed_kwargs = {
        'in_dim': in_dim,
        'num_freqs': num_freqs,
        'log_sampling': True,
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def positional_encoding(input, L=6, log_sampling=False):
    if log_sampling:
        freq_bands = 2.**torch.linspace(0., L-1, L)
    else:
        freq_bands = torch.linspace(2.**0., 2.**(L-1), L)

    enc = [input]
    for f in freq_bands:
        for fn in [torch.sin, torch.cos]:
            enc.append(fn(f * input))

    return torch.cat(enc, dim=-1)