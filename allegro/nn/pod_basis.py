import torch

def eigendecomposition(B, x):

    N = size(B,1)
    M = size(B,2)

    A = (1.0/N)*(B'*B)
    s = eigvals(A)
    U = eigvecs(A)
    ind = argsort(s)
    s = s[ind]
    U = U[:,ind]

    Bnew = B*U;
    sij = x[2:end] - x[1:end-1]
    for i in range(M):
        B2 = Bnew[:,i].*Bnew[:,i]    
        area = 0.5*sum(sij.*(B2[2:end]+B2[1:end-1]))    
        U[:,i] = U[:,i]/sqrt(area);

    return U, s 

def podsnapshots(x, r_min, r_max, pdegree, kdgree, scalefac):
        
    N = length(x)
    M = length(scalefac)
    nbf = pdegree*M + kdegree
    rbf = zeros(N, nbf)

    r = x - r_min
    rmax = r_max - r_min
    for j in range(M):
        alpha = scalefac[j]    
        if (alpha == 0):
            alpha = 1e-3
        y = (1.0 - exp(-alpha*r/rmax))/(1.0-exp(-alpha))

        for i in range(pdegree):                
            rbf[:,i+j*pdegree] = (sqrt(2.0/(rmax))/(i+1))*sin((i+1)*pi*y)/r

    for i in range(kdegree):
        n = pdegree*M+i
        rbf[:,n] = 1.0/(x^(i+1))

    return rbf

def podprojection(x, r_min, r_max, pdegree, kdgree, scalefac):
        
    rbf = podsnapshots(x, r_min, r_max, pdegree, kdgree, scalefac)
    fcut = exponential_cutoff(x, r_max, r_min) 
    
    M = length(scalefac)
    nbf = pdegree*M + kdegree
    for n in range(nbf):
        rbf[:,n] = rbf[:,n] * fcut

    U, s = eigendecomposition(rbf, x)
    
    return U, s

class POD_Basis(nn.Module): 
    r_max: float
    r_min: float

    def __init__(self, r_max, r_min: float = 0.0, num_basis=10, pdegree=6, kdegree=8):
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
        self.pdegree = pdegree
        self.kdegree = kdegree

        self.r_max = float(r_max)
        self.r_min = float(r_min)

        scalefac = [0.0, 2.0, 4.0]
        self.scalefac = scalefac        
        rs = torch.linspace(r_min, r_max, 4001)[1:]

        Phi, Lambda = podprojection(rs, r_min, r_max, pdegree, kdgree, scalefac)
        self.Phi = Phi[:,1:num_basis]
        self.Lambda = Lambda[1:num_basis]

        self.register_buffer("scalefac", scalefac)
        self.register_buffer("Phi", Phi)
        self.register_buffer("Lambda", Lambda)            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.
        Parameters
        ----------
        x : torch.Tensor
            Input
        """

        rbf = podsnapshots(x, self.r_min, self.r_max, self.pdegree, self.kdgree, self.scalefac)

        return rbf*self.Phi

