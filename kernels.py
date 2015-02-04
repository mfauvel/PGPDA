'''
Created on 22 oct. 2014

@author: mfauvel
'''
import scipy as sp
from scipy import weave
from scipy.weave import converters
import  multiprocessing as mp

def find_optimal_sig(x,y,sig_r=2.0**sp.arange(-5,5,0.5),ncpus=None):
    '''
    Compute the centered alignement for several value of the kernel parameter 
    '''
    if ncpus is None:
        ncpus=mp.cpu_count()
    
    A =  [compute_alignement(sig,x,y,ncpus) for sig in sig_r]
    A = sp.asarray(A)

    t = A.argmax()
    return sig_r[t],A

def compute_alignement(sig,x,y,ncpus=2):
    # Get some parameters
    n,d = sp.shape(x)
    support_code="""
    #include <math.h>
    #include <stdio.h>
    #include <omp.h>
    """

    code=r"""
    int i,j,k,tm;
    double A=0;
    double kn=0;
    double kin=0;
    double s=0;
    double ki;
    double ktp;
    double *ks = (double*)calloc(n, sizeof(double));
    double **reduc_ks = (double**)calloc(ncpus, sizeof(double*));
    for(i=0;i<ncpus;i++)
        reduc_ks[i]= (double*)calloc(n, sizeof(double));
    double MSIG = -1.0*(double)sig;
     
    # pragma omp parameters for private (i,j,k,ktp,tm) shared(x,y,sig,n,d,reduc_ks)  reduction(+:s) schedule(runtime) num_threads(ncpus)
    for(i=0;i<n;i++){
        tm = omp_get_thread_num();
        reduc_ks[tm][i] += 1;
        s+=1;
        for(j=i+1;j<n;j++){
            ktp = 0;
            for(k=0;k<d;k++)
                ktp += (x(i,k)-x(j,k))*(x(i,k)-x(j,k));
            ktp *= MSIG;
            ktp = exp(ktp);
            reduc_ks[tm][i] += 2*ktp;
            s += 2*ktp;
        }
        reduc_ks[tm][i] /= n;
    }
    s /= (n*n);

    for(i=0;i<ncpus;i++){
        for(j=0;j<n;j++)
            ks[j]+=reduc_ks[i][j];
        free(reduc_ks[i]);
    }
    free(reduc_ks);
        
    # pragma omp parallel for private(i,j,k,ktp,ki) shared(x,y,sig,n,d,s,ks) reduction (+:A,kn,kin) num_threads(ncpus) schedule(runtime)
    for(i=0;i<n;i++){
        A += 1;
        kn += 1;
        kin +=1;
        for(j=i+1;j<n;j++){
            ki = (y(i) == y(j)) ? 1 : 0;
            ktp = 0;
            for(k=0;k<d;k++)
                ktp += (x(i,k)-x(j,k))*(x(i,k)-x(j,k));
            ktp *= MSIG;
            ktp = exp(ktp);
            ktp += s -ks[i]-ks[j];
            A +=2*ktp*ki;
            kn += 2*ktp*ktp;
            kin+=2*ki;
        }
    }
    kn = sqrt(kn);
    kin= sqrt(kin);
    A /=(kn*kin);   
    return_val=A;
    """
    A=weave.inline(code,['sig','n','d','x','y','ncpus'],support_code=support_code,type_converters = converters.blitz,compiler='gcc',extra_compile_args =['-O3 -fopenmp'], extra_link_args=['-lgomp'],)
    return A

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
            D *= (-1.0*sig)

            ## Compute the Kernel matrix
            self.K = sp.exp(D)
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

    
