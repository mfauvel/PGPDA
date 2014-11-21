'''
Created on 22 oct. 2014

@author: mfauvel
'''
import scipy as sp
def sq_dist(X,Z=None):
    '''
    The function to computes a matrix of all pairwise squared distances between two sets of vectors
    Substract the mean value for numerical precision
    
    (x-z)^2 = x^2+z^2-2<x,z>
    '''
    x=X.copy()
     
    if Z is None:
        z=x
        nx = x.shape[0]
        mu = sp.mean(x,axis=0)
        x-=mu
        z-=mu
        D = -2*sp.dot(x,z.T)
        x2 = sp.sum(x**2,axis=1)
        D += x2.reshape(nx,1)
        D += x2.T.reshape(1,nx)
    else:
        z=Z.copy() 
        nx,nz = x.shape[0],z.shape[0]
        n = nx+nz        
        mu = (nx*sp.mean(x,axis=0)+nz*sp.mean(x,axis=0))/n
        x -= mu
        z -= mu
        D = -2*sp.dot(x,z.T)
        D += sp.sum(x**2,axis=1).reshape(nx,1)
        D += sp.sum(z**2,axis=1).T.reshape(1,nz)
                
    return D

class KERNEL:
    def __init__(self):
        self.K=0
        self.rank=0
        self.kd=0
        
    def compute_kernel(self,x,z=None,kernel='RBF',sig=None):
        ''' 
        Compute the kernel matrix and the rank of the kernel
        Input:
            x : the sample matrix nxd (number of samples x number of variables)
            z : idem 
            kernel : the kernel used. Default: RBF.
            sig : the kernel parameter

        '''
        # Free memory
        self.K=0.0
        self.rank=0
        self.kd=0
        
        if (sig is None) and (kernel == 'RBF'):
            print 'Parameters must be selected for the RBF kernel.'
            exit()
            
        if kernel == 'RBF':
            self.rank=x.shape[0] # Get the rank of the matrix
            ## Compute the pairwise distance matrix
            D = sq_dist(x,z)
                
            ## Compute the Kernel matrix
            self.K = sp.exp(-0.5*D/(sig**2))
            del D
      
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
