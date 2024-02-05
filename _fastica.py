import torch
class FastICA():
    def __init__(
        self, 
        n_components = 5,
        max_iter = 500,
        alpha = 0.01,
        tol=1e-4,
        non_linearity = 'tanh'
        ):
        self.n_components = n_components
        self.non_linearity = non_linearity
        self.tol = tol
        
    @staticmethod
    def _logcosh(x):
        """Logcosh function."""
        return torch.log(torch.cosh(x))

    @staticmethod
    def _tanh(x):
        """Hyperbolic tangent function."""
        return torch.tanh(x)
        
    @staticmethod
    def _symmetric_decorrelation(W):
        K = torch.mm(W, W.t())
        s, u = torch.linalg.eigh(K)
        return torch.mm(torch.mm(torch.mm(u, torch.diag(1.0 / torch.sqrt(s))), u.t()),  W)

    
    def fit_transform(self, X):
        n, m = X.shape
        if self.non_linearity == 'logcosh':
            g = FastICA._logcosh
            g_prime = lambda x: torch.tanh(x)
        elif self.non_linearity == 'tanh':
            g = FastICA._tanh
            g_prime = lambda x: 1.0 - torch.tanh(x) ** 2
        else:
            raise ValueError("Invalid nonlinearity. Choose 'logcosh' or 'tanh'.")
        # Whitening
        X -= X.mean(dim=1, keepdim = True)
        U, D, _ = torch.svd(torch.matmul(X, X.T) / n_samples)
        whiteM = torch.matmul(torch.diag(1.0 / D), U.T)
        whiteM = torch.matmul(U,whiteM)
        X_white = torch.matmul(whiteM, X)
        W = torch.randn(X.shape[0],X.shape[0], dtype=x.dtype, device=x.device)

        for _ in range(max_iter):
            W_prev = W.clone()
            # Estimate sources
            Y = torch.matmul(W.T, X_white)
            Y = g(self,Y)
            # Update weights
            W_grad = torch.matmul(X_white, Y.T) / n_samples # X_white 3x10, Y 10x3
            W_grad -= torch.mean(g_prime(Y), dim=1, keepdim=True)*W_grad
            W = (1 - alpha) * W + alpha * W_grad
        
            # Symmetric orthogonalization
            W = FastICA._symmetric_decorrelation(W)
        
            # # Check convergence
            if torch.norm(W - W_prev) < self.tol:
                break
        
        return W, S
              