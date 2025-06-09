# Based on Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
# code based on Dominique Zosso's MATLAB code, available at:
# https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
# Original paper:
# Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
# IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.


import numpy as np

def  VMD_pers(f, alpha, tau, K, DC, init, tol, omega0=None, Niter=500, speed_omegas=1): # By Paolo: added omega0


    if len(f)%2:
        raise ValueError("The length of the signal must be even.")
    

    fs = 1./len(f)
    
    ltemp = len(f)//2 
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)  
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))

    T = len(fMirr)
    t = np.arange(1,T+1)/T  
    
    freqs = t-0.5-(1/T)

    Alpha = alpha*np.ones(K)
    
    f_hat = np.fft.fftshift((np.fft.fft(fMirr))) 
    f_hat_plus = np.copy(f_hat) 
    
    f_hat_plus[:T//2] = 0 

    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    elif init == 3: 
        if omega0 is not None:
            omega_plus[0,:] = omega0[:]
        else:
            omega_plus[0,:] = 0
    elif init == 4:
        omega_plus[0,K//2:K] = np.sort(np.exp(np.log(0.25) + (np.log(0.5)-np.log(0.25))*np.random.rand(1,K//2)))
        for i in range(K//2):
            omega_plus[0,i] = (0.5/K)*(i)
    else:
        omega_plus[0,:] = 0
            
    if DC:
        omega_plus[0,0] = 0
    
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
    
    uDiff = tol+np.spacing(1) # update step
    n = 0 
    sum_uk = 0 # accumulator
       
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)    

    while ( uDiff > tol and  n < Niter-1 ):       
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]
        
        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2)
        
        if not(DC):
            omega_plus[n+1,k] = speed_omegas*np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
        
        for k in np.arange(1,K):
            
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2)
            
            omega_plus[n+1,k] = speed_omegas*np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
           
        
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis = 1) - f_hat_plus)
        
        n = n+1
        
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i]),np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])))

        uDiff = np.abs(uDiff)    

    print(f'The number of iterations is: {n}')    
            
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]
    
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)

    
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])    
    
    u = np.zeros([K,len(t)])
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
        
    u = u[:,T//4:3*T//4]

    u_hat = np.zeros([u.shape[1],K],dtype = complex)
    for k in range(K):
        u_hat[:,k]=np.fft.fftshift(np.fft.fft(u[k,:]))

    return u, u_hat, omega
