import torch

class POD_Basis(nn.Module):
    r_max: float
    r_min: float
    prefactor: float

    def __init__(self, r_max, r_min: float = 0.0, num_basis=10, pdegree=6, kdegree=8, trainable=True):
        r"""Proper Orthogonal Radial Basis, as proposed in POD
        Parameters
        ----------
        r_max : float
            Cutoff radius
        num_basis : int
            Number of Bessel Basis functions
        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.
        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))
