'''
Created on 22 oct. 2014

@author: mfauvel
'''
import scipy as sp
from scipy import spatial
from parakeet import jit

@jit
def dist(x,z):
        return sp.sum( (x-z)**2 )

@jit
def all_pairdistance(X,Z):
    
    return sp.array([[dist(x,z) for z in Z] for x in X])


@jit
def gram_distance(X):

    n = X.shape[0]
    D = sp.empty((n,n))

    for i in range(n):
        D[i,i]=0.0
        for j in range(i+1,n):
            D[i,j] = dist(X[i,:],X[j,:])
            D[j,i] = D[i,j]
    return D

class KERNEL:
    def __init__(self):
        self.K=0
        
    def compute_kernel(self,x,z=None,kernel='RBF',sig=None,compute_rank=None):
        ''' 
        Compute the kernel matrix
        Input:
            x : the sample matrix nxd (number of samples x number of variables)
            z : idem 
            kernel : the kernel used. Default: RBF.
            sig : the kernel parameter
        Output
            r = the rank of the kernel matrix
        '''
        # Free memory
        self.K=0.0
        
        if (sig is None) and (kernel == 'RBF'):
            print 'Parameters must be selected for the RBF kernel. The program is closed.'
            exit()
            
        if kernel == 'RBF':
            r=x.shape[0] # Get the rank of the matrix
            ## Compute the pairwise distance matrix
            if z is None:
                D = gram_distance(x)
            else:
                D = all_pairdistance(x,z)
                
            ## Compute the Kernel matrix
            self.K = sp.exp(-0.5*D/(sig**2))
            del D
        
        if compute_rank is not None:
            return r
        
    def compute_diag_kernel(self,x,kernel='RBF',sig=None):
        '''
        The function computes the kernel evaluation K(x_i,x_i)
        '''
        if kernel=='RBF':
            self.K= sp.ones((x.shape[0],1))
        
    def scale_kernel(self,s):
        self.K/=s
        
    def center_kernel(self,Ko=None,kd=None):
        '''
        The function center the kernel matrix. If the second argument is provided, it is used as the reference for the centering.
        Input:
            Ko: the reference kernel matrix (for testing)
            Kd: the diagonal kernel matrix (for testing)
        '''
        if Ko is None:
            n = self.K.shape[0]
            s = sp.sum(self.K)/n**2
            ks = sp.sum(self.K,axis=0).reshape(n,1)/n
            self.K -= ks
            self.K -= ks.T
            self.K += s
            del ks, s
        else:
            nt,ni =  self.K.shape
            s = sp.sum(Ko.K)/(ni**2)
            kos = sp.sum(Ko.K,axis=1).reshape(ni,1)/ni
            ks = sp.sum(self.K,axis=1).reshape(nt,1)/ni
            self.K -= kos.T
            self.K -= ks
            self.K += s
                        
            kd.K -= 2*ks
            kd.K += s
            kd.K.shape = (nt,)
            
            del kos,ks,s
