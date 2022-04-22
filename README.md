## Learning Registration Models with Differentiable Gauss-Newton Optimisation
# MIDL 2022 short paper (under review)

We propose to capture large deformations in few iterations by learning a registration model with differentiable Gauss-Newton and compact CNNs that predict displacement gradients and a suitable residual function. By incorporating a sparse Laplacian regulariser, structural / semantic representations and weak label-supervision we achieve state-of-the-art performance for abdominal CT registration.


 The overall concept includes trainable CNNs that may predict better displacement gradient and residual function values.
 
 ![Concept Figure](https://github.com/mattiaspaul/LearnGN/raw/main/midl2022_shortpaper_concept.png)
 
This key idea is achieved by implementing a second-order Gauss-Newton descent with diffusion regularisation (sparse Laplacian) in a differentiable manner and doing so very efficiently (the below code takes fractions of seconds even for large 3D problems.
```
import cupyx.scipy.sparse.csr_matrix
import cupyx.scipy.sparse.linalg.cg
import cupy as cp
def cupyCG(A2,b2):
    b_val = cp.asarray(b2.data)
    n1 = len(b_val)
    A_ind = cp.asarray(A2._indices().data)
    A_val = cp.asarray(A2._values().data)
    SC = cupyx.scipy.sparse.csr_matrix((A_val,(A_ind[0,:],A_ind[1,:])), shape=(n1,n1))
    solution = cupyx.scipy.sparse.linalg.cg(SC, b_val,tol=1e-3,maxiter=40)[0]
    x = torch.as_tensor(solution, device='cuda')
    return x

from torch.autograd import Function
class LSESolver(Function):

    @staticmethod
    def forward(ctx, A,d,b):
        xy = torch.arange(len(b)).to(b.device)
        A1 = torch.sparse.FloatTensor(torch.cat((xy.view(1,-1),xy.view(1,-1)),0),d,(len(b),len(b)))
        x = cupyCG(A+A1,b).unsqueeze(1)
        ctx.save_for_backward(A+A1,b,x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        AA1,b,x = ctx.saved_tensors
        grad_b = cupyCG(AA1.t(),grad_x).unsqueeze(1)
        grad_A = None
        grad_d = -(x*grad_b)
        return grad_A, grad_d.squeeze(), grad_b
 ```
 
