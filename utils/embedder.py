import torch
import numpy as np

class PositionalEncodingEmbedder:
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


def get_posenc_embedder(in_dim=3, num_freqs=6):
    embed_kwargs = {
        'in_dim': in_dim,
        'num_freqs': num_freqs,
        'log_sampling': True,
    }

    embedder_obj = PositionalEncodingEmbedder(**embed_kwargs)
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

class SphericalHarmonicsEmbedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_sph_harm_encoding()

    def create_sph_harm_encoding(self):
        """Generate integrated directional encoding (IDE) function.
        This function returns a function that computes the integrated directional
        encoding from Equations 6-8 of arxiv.org/abs/2112.03907.
        Args:
        deg_view: number of spherical harmonics degrees to use.
        Returns:
        A function for evaluating integrated directional encoding.
        Raises:
        ValueError: if deg_view is larger than 5.
        """
        deg_view = self.kwargs["deg_view"]
        if deg_view > 5:
            raise ValueError('Only deg_view of at most 5 is numerically stable.')

        self.ml_array = self.get_ml_array(deg_view)
        l_max = 2**(deg_view - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        self.mat = np.zeros((l_max + 1, self.ml_array.shape[1]))
        for i, (m, l) in enumerate(self.ml_array.T):
            for k in range(l - m + 1):
                self.mat[k, i] = self.sph_harm_coeff(l, m, k)

        self.out_dim = self.mat.shape[0]
        print("out_dim", self.out_dim)

    def embed(self, xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).
        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.
        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.concatenate([z**i for i in range(self.mat.shape[0])], axis=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.concatenate(
            [(x + 1j * y)**m for m in self.ml_array[0, :]], axis=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, self.mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * self.ml_array[1, :] * (self.ml_array[1, :] + 1)
        ide = sph_harms * np.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.concatenate([torch.real(ide), torch.imag(ide)], axis=-1)

    def sph_harm_coeff(self, l, m, k):
        """Compute spherical harmonic coefficients."""
        return (np.sqrt(
            (2.0 * l + 1.0) * np.math.factorial(l - m) /
            (4.0 * np.pi * np.math.factorial(l + m))) * self.assoc_legendre_coeff(l, m, k))
        
    def get_ml_array(self, deg_view):
        """Create a list with all pairs of (l, m) values to use in the encoding."""
        ml_list = []
        for i in range(deg_view):
            l = 2**i
            # Only use nonnegative m values, later splitting real and imaginary parts.
            for m in range(l + 1):
                ml_list.append((m, l))

        # Convert list into a numpy array.
        ml_array = np.array(ml_list).T
        return ml_array

    def generalized_binomial_coeff(self, a, k):
        """Compute generalized binomial coefficients."""
        return np.prod(a - np.arange(k)) / np.math.factorial(k)


    def assoc_legendre_coeff(self, l, m, k):
        """Compute associated Legendre polynomial coefficients.
        Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
        (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).
        Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).
        Returns:
        A float, the coefficient of the term corresponding to the inputs.
        """
        return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
                np.math.factorial(l - k - m) *
                self.generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

def get_sph_harm_embedder(in_dim=3, num_freqs=6):
    embed_kwargs = {
        'in_dim': in_dim,
        'num_freqs': num_freqs,
        'log_sampling': True,
    }

    embedder_obj = PositionalEncodingEmbedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim











