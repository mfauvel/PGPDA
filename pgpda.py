'''
Created on 18 oct. 2014

@author: mfauvel
'''
import scipy as sp
from scipy import linalg
from kernels import KERNEL

def estim_d(E,threshold):
    ''' The function estimates the intrinsic dimension by looking at the cumulative variance
    Input:
        E: the eigenvalue
        threshold: the percentage of the cumulative variance
    Output:
        d: the intrinsic dimension
    '''
    if E.size == 1:
        d=1
    else:
        d = sp.where(sp.cumsum(E)/sp.sum(E)>threshold)[0][0]+1
    return d
    
def standardize(x,M=None,S=None,REVERSE=None):
    ''' Function that standardize the data
        Input:
            x: the data
            M: the mean vector
            V: the standard deviation vector
        Output:
            x: the standardize data
            M: the mean vector
            V: the standard deviation vector
    '''
    if not sp.issubdtype(x.dtype,float):
        do_convert = 1
    else:
        do_convert = 0
    if REVERSE is None:
        if M is None:
            M = sp.mean(x,axis=0)
            S = sp.std(x,axis=0)
            if do_convert:
                xs = (x.astype('float')-M)/S
            else:
                xs = (x-M)/S
            return xs,M,S
        else:
            if do_convert:
                xs = (x.astype('float')-M)/S
            else:
                xs = (x-M)/S
            xs = (x-M)/S
            return xs
    else:
        return S*x+M

class CV:#Cross_validation 
    def __init__(self):
        self.it=[]
        self.iT=[]
        
    def split_data(self,n,v=5):
        ''' The function split the data into v folds. Whatever the number of sample per class
        Input:
            n : the number of samples
            v : the number of folds
        Output: None        
        '''
        sp.random.seed(1)   # Set the random generator to the same initial state
        step = int(sp.ceil(float(n)/v)) # Compute the number of samples in each fold
        t = sp.random.permutation(n)    # Generate random sampling of the indices
        
        indices=[]
        for i in range(v-1):            # group in v fold
            indices.append(t[i*step:(i+1)*step])
        indices.append(t[(v-1)*step:n])
                
        for i in range(v):
            self.iT.append(sp.asarray(indices[i]))
            l = range(v)
            l.remove(i)
            temp = sp.empty(0,dtype=sp.int64)
            for j in l:            
                temp = sp.concatenate((temp,sp.asarray(indices[j])))
            self.it.append(temp)

class PGPDA: # Parcimonious Gaussian Process Discriminant Analysis
    def __init__(self,model='M0',kernel='RBF',sig=None,dc=None,threshold=None):
        self.model=model
        self.kernel=kernel
        self.sig=sig
        self.dc=dc
        self.threshold=threshold
        self.A=[]
        self.Beta=[]
        self.b = 0.0
        self.ib = 0.0
        self.a = []
        self.prop = []
        self.ni= []
        self.di = []
        self.ri = []
        self.precomputed = None
        
    def train(self,x,y,sig=None,dc=None,threshold=None):
        '''
        The function trains the pgpda model using the training samples
        Input:
            x: the samples matrix of size n x d
            y: the vector with label of size n
            For the precomputed case (self.precomputed==1), x is a KERNEL object.
        Output:
            None; The model is included/updates in the object
        '''
        # Initialization
        n = y.shape[0]
        C = int(y.max())
        eps = sp.finfo(sp.float64).eps      
        list_model_dc = 'M1 M3 M4 M6'
        
        if (sig is None) and (self.sig is None):
            self.sig=0.5
        elif self.sig is None:
            self.sig = sig
        
        if (dc is None) and (self.dc is None):
            self.dc=2
        elif self.dc is None:
            self.dc = dc
            
        if (threshold is None) and (self.threshold is None):
            self.threshold=0.95
        elif self.threshold is None:
            self.threshold = threshold
        
        # Check of consistent dimension
        if (list_model_dc.find(self.model) > -1): 
            for i in range(C):
                ni = sp.size(sp.where(y==(i+1))[0])
                if self.dc > ni:
                    self.dc=ni-1
        
        for i in range(C):
            t = sp.where(y==(i+1))[0]
            self.ni.append(sp.size(t))
            self.prop.append(float(self.ni[i])/n)
            
            # Compute Mi
            Ki= KERNEL()
            if self.precomputed is None:
                Ki.compute_kernel(x[t,:],kernel=self.kernel,sig=self.sig)                
            else:
                Ki.K = x.K[t,:][:,t].copy()
                Ki.rank = Ki.K.shape[0]
                
            self.ri.append(Ki.rank)
            Ki.center_kernel()
            Ki.scale_kernel(self.ni[i])
                        
            # Eigenvalue decomposition TBD 
            E,Beta = linalg.eigh(Ki.K)
            idx = E.argsort()[::-1]
            E = E[idx]
            E[E<eps]=eps
            Beta = Beta[:,idx]
            
            # Parameter estimation
            if list_model_dc.find(self.model) == -1:
                di = estim_d(E[0:self.ri[i]],self.threshold)
            else:
                di = self.dc
            self.di.append(di)
            self.a.append(E[0:di])
            self.b += self.prop[i]*(sp.trace(Ki.K)-sp.sum(self.a[i]))            
            self.Beta.append(Beta[:,0:di])
            del Beta,E
            
        # Last step for the safe estimation of 'b'
        denom = sum(map(lambda p,r,d:p*(r-d),self.prop,self.ri,self.di)) 
        
        if denom <eps:
            self.ib = eps
            self.b/=eps
        elif self.b <eps:
            self.ib = 1.0/eps
            self.b = eps
        else:
            self.ib = denom/self.b
            self.b /=denom
        
        # Finish the estimation for the different models
        if self.model == 'M0' or self.model == 'M1':
            for i in range(C):
                # Compute the value of matrix A
                temp =self.Beta[i]*((1/self.a[i]-self.ib)/self.a[i]).reshape(self.di[i])
                self.A.append(sp.dot(temp,self.Beta[i].T)/self.ni[i])

        elif self.model == 'M2' or self.model == 'M3':
            for i in range(C):
                # Update the value of a
                self.a[i][:]=sp.mean(self.a[i])
                # Compute the value of matrix A
                temp =self.Beta[i]*((1/self.a[i]-self.ib)/self.a[i]).reshape(self.di[i])
                self.A.append(sp.dot(temp,self.Beta[i].T)/self.ni[i])

        elif self.model == 'M4': 
            # Compute the value of a
            al = sp.zeros((self.dc))
            for i in range(self.dc):
                for j in range(C):
                    al[i] += self.prop[j]*self.a[j][i]
            for i in range(C):
                self.a[i]=al.copy()
                temp =self.Beta[i]*((1/self.a[i]-self.ib)/self.a[i]).reshape(self.di[i])
                self.A.append(sp.dot(temp,self.Beta[i].T)/self.ni[i])

        elif self.model == 'M5' or self.model=='M6':
            num = sum(map(lambda p,a:p*sum(a),self.prop,self.a))
            den = sum(map(lambda p,d:p*d,self.prop,self.di))
            ac = num/den
            for i in range(C):
                self.a[i][:]=ac
                temp =self.Beta[i]*((1/self.a[i]-self.ib)/self.a[i]).reshape(self.di[i])
                self.A.append(sp.dot(temp,self.Beta[i].T)/self.ni[i])
                
        self.A = sp.asarray(self.A)   
           
    def predict(self,xt,x,y,out_decision=None,out_proba=None):
        '''
        The function predicts the label for each sample with the learned model
        Input:
            xt: the test samples
            x: the samples matrix of size n x d
            y: the vector with label of size n
        Output
            yp: the label
            D: the discriminant function
            P: the posterior probabilities
        '''
         
        # Initialization
        if isinstance(xt,sp.ndarray):
            nt = xt.shape[0]
        else:
            nt = xt.K.shape[0]
            
        C = int(y.max())
        eps = sp.finfo(sp.float64).eps
        dm = max(self.di)

        D = sp.empty((nt,C))
        Ki = KERNEL()
        Kt = KERNEL()
        kd = KERNEL()
        
        for i in range(C):
            t = sp.where(y==(i+1))[0]
            cst = sp.sum(sp.log(self.a[i])) + (dm-self.di[i])*sp.log(self.b) -2*sp.log(self.prop[i]) 
            if self.precomputed is None:
                Ki.compute_kernel(x[t,:],kernel=self.kernel,sig=self.sig)
                Kt.compute_kernel(xt,z=x[t,:],kernel=self.kernel,sig=self.sig)
                kd.compute_diag_kernel(xt,kernel=self.kernel,sig=self.sig)
            else:
                Ki.K= x.K[t,:][:,t].copy()
                Kt.K= xt.K[:,t].copy()
                kd.K= xt.kd.copy()
            Kt.center_kernel(Ko=Ki, kd=kd)
            Ki.K=None

            #Compute the decision rule
            temp = sp.dot(Kt.K,self.A[i])
            D[:,i] = sp.sum(Kt.K*temp,axis=1)
            D[:,i] += kd.K*self.ib+cst           
            
        # Check if negative value
        if D.min() <0:
            D-=D.min()
        
        yp = D.argmin(1)+1
        yp.shape=(nt,1)

        # Format the output
        if out_proba is None:
            if out_decision is None:
                return yp
            else:
                return yp,D
        else:        
            # Compute posterior !! Should be changed to a safe version
            P = sp.exp(-0.5*D)
            P /= sp.sum(P,axis=1).reshape(nt,1)
            P[P<eps]=0                    
        return yp,D,P
    
    def cross_validation(self,x,y,v=5,sig_r=2.0**sp.arange(-8,0),threshold_r=sp.linspace(0.85,0.9999,10),dc_r=sp.arange(5,50)):
        '''
        To be done and can be changed by using pre-computed kernels
        '''
        # Get parameters
        n=x.shape[0]
        ns = sig_r.size
        nt = threshold_r.size
        nd = dc_r.size
        if self.model == 'M0' or self.model=='M2' or self.model =='M5':
            err = sp.zeros((ns,nt))
        else:
            err = sp.zeros((ns,nd))
            
        # Initialization of the indices for the cross validation
        cv = CV()           
        cv.split_data(n,v=v)
        
        # Start the cross-validation
        if self.model == 'M0' or self.model=='M2' or self.model =='M5':
            for i in range(ns):
                for j in range(nt):
                    for k in range(v):
                        model_temp = PGPDA(model=self.model,kernel=self.kernel)
                        model_temp.train(x[cv.it[k],:],y[cv.it[k]],sig=sig_r[i],threshold=threshold_r[j])
                        yp = model_temp.predict(x[cv.iT[k],:],x[cv.it[k],:],y[cv.it[k]])
                        yp.shape = y[cv.iT[k]].shape
                        t = sp.where(yp!=y[cv.iT[k]])[0]
                        err[i,j]+= float(t.size)/yp.size
                        del model_temp
            err/=v
            t = sp.where(err==err.min())
            self.sig = sig_r[t[0][0]]
            self.threshold = threshold_r[t[1][0]]
            return sig_r[t[0][0]],threshold_r[t[1][0]],err
                        
        else:
            for i in range(ns):
                for j in range(nd):
                    for k in range(v):
                        model_temp = PGPDA(model=self.model,kernel=self.kernel)
                        model_temp.train(x[cv.it[k],:],y[cv.it[k]],sig=sig_r[i],dc=dc_r[j])
                        yp = model_temp.predict(x[cv.iT[k],:],x[cv.it[k],:],y[cv.it[k]])
                        t = sp.where(yp!=y[cv.iT[k]])[0]
                        err[i,j]+= float(t.size)/yp.size
                        del model_temp
            err/=v
            t = sp.where(err==err.min())
            self.sig = sig_r[t[0][0]]
            self.dc = dc_r[t[1][0]]
            return sig_r[t[0][0]],dc_r[t[1][0]],err

class KDA: # Kernel QDA from "Toward an Optimal Supervised Classifier for the Analysis of Hyperspectral Data"
    def __init__(self,mu=None,sig=None):
        self.a = []
        self.A = []
        self.S = []
        self.ni = []
        self.prop=[]
        self.sig=sig
        self.mu=mu
    
    def train(self,x,y,mu=None,sig=None):
        # Initialization
        n = y.shape[0]
        C = int(y.max())
        eps = sp.finfo(sp.float64).eps 
        
        if (mu is None) and (self.mu is None):
            mu=10**(-7)
        elif self.mu is None:
            self.mu =mu
            
        if (sig is None) and (self.sig is None):
            self.sig=0.5
        elif self.sig is None:
            self.sig=sig
        
        # Compute K and 
        K = KERNEL()
        K.compute_kernel(x,sig=self.sig)
        G = KERNEL()
        G.K = self.mu*sp.eye(n)
                    
        for i in range(C):
            t = sp.where(y==(i+1))[0]
            self.ni.append(sp.size(t))
            self.prop.append(float(self.ni[i])/n)
        
            # Compute K_k
            Ki = KERNEL()
            Ki.compute_kernel(x, z=x[t,:],sig=self.sig)
            T = (sp.eye(self.ni[i])-sp.ones((self.ni[i],self.ni[i])))
            Ki.K = sp.dot(Ki.K,T)
            del T
            G.K += sp.dot(Ki.K,Ki.K.T)/self.ni[i]
        G.scale_kernel(C)
        
        # Solve the generalized eigenvalue problem
        a,A = linalg.eigh(G.K,b=K.K)
        idx = a.argsort()[::-1]
        a=a[idx]
        A=A[:,idx]
        
        # Remove negative eigenvalue
        t = sp.where(a>eps)[0]
        a=a[t]
        A=A[:,t]
        
        # Normalize the eigenvalue
        for i in range(a.size):
            A[:,i]/=sp.sqrt(sp.dot(sp.dot(A[:,i].T,K.K),A[:,i]))
        
        # Update model   
        self.a=a
        self.A=A
        self.S= sp.dot(sp.dot(self.A,sp.diag(self.a**(-1))),self.A.T)
        
        # Free memory
        del G,K
    
    def predict(self,xt,x,y,out_decision=None,out_proba=None):
        nt = xt.shape[0]
        C = int(y.max())
        D = sp.empty((nt,C))
        D += self.prop
        eps = sp.finfo(sp.float64).eps
        
        # Pre compute the Gramm kernel matrix
        Kt = KERNEL()
        Kt.compute_kernel(xt,z=x,sig=self.sig)
        Ki = KERNEL()
                
        for i in range(C):
            t = sp.where(y==(i+1))[0]
            Ki.compute_kernel(x,z=x[t,:],sig=self.sig)
            T = Kt.K - sp.dot(Ki.K,sp.ones((self.ni[i]))/self.ni[i])
            temp = sp.dot(T,self.S)
            D[:,i] = sp.sum(T*temp,axis=1)
        
        # Check if negative value
        if D.min() <0:
            D-=D.min()
        
        yp = D.argmin(1)+1
        yp.shape=(nt,1)

        # Format the output
        if out_proba is None:
            if out_decision is None:
                return yp
            else:
                return yp,D
        else:        
            # Compute posterior !! Should be changed to a safe version
            P = sp.exp(-0.5*D)
            P /= sp.sum(P,axis=1).reshape(nt,1)
            P[P<eps]=0                    
        return yp,D,P
    
    def cross_validation(self,x,y,v=5,sig_r=2.0**sp.arange(-8,0),mu_r=10.0**sp.arange(-15,0)):
        # Get parameters
        n=x.shape[0]
        ns = sig_r.size
        nm = mu_r.size
        err = sp.zeros((ns,nm))
        
        # Initialization of the indices for the cross validation
        cv = CV()           
        cv.split_data(n,v=v)
        
        for i in range(ns):
            for j in range(nm):
                for k in range(v):
                    model_temp=KDA()
                    model_temp.train(x[cv.it[k],:],y[cv.it[k]],sig=sig_r[i],mu=mu_r[j])
                    yp = model_temp.predict(x[cv.iT[k],:],x[cv.it[k],:],y[cv.it[k]])
                    yp.shape = y[cv.iT[k]].shape
                    t = sp.where(yp!=y[cv.iT[k]])[0]
                    err[i,j]+= float(t.size)/yp.size
                    del model_temp
        err/=v
        t = sp.where(err==err.min())
        self.sig = sig_r[t[0][0]]
        self.mu = mu_r[t[1][0]]
        return sig_r[t[0][0]],mu_r[t[1][0]],err
                    
                