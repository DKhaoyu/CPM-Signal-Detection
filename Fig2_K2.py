import numpy as np
import math
import scipy.signal 
from scipy.integrate import quad
import matplotlib.pyplot as plt

def CPM2PAM(alpha_seq,M,h,Lg,BT,K,sps):
    if M == 2:
        sig_len = len(alpha_seq)
        sample_time = sps     #每个符号的采样数
        F_cal = sample_time
        L = Lg
        t = np.linspace(-Lg/2,Lg/2,Lg*sample_time+1)     #为精确重构q，采用比原始采样率的5倍进行重构
        len1 = len(t)
        len2 = 2*len(t)-1
        delta = math.sqrt(math.log(2))/(2*math.pi*BT)
        sigma = delta
        f = np.zeros(len1)
        for k in range(len1):
            kernal = lambda tau:1/(math.sqrt(2*math.pi)*sigma)*np.exp(-((t[k]-tau)**2)/(2*(sigma**2)))
            info = quad(kernal,-1/2,1/2)
            f[k] = 1/2*info[0]
        #plt.figure()
        #plt.stem(f)
        q = np.zeros(len1)
        for i in range(len1-1):
            q[i+1] = q[i] + 1/(F_cal)*f[i+1]
        #plt.figure()
        #plt.stem(q)
        #plt.show()
        u = np.zeros(len2)
        for i in range(len2):
            if i < len1:
                u[i] = math.sin(2*h*math.pi*q[i])/math.sin(h*math.pi)
            else:
                u[i] = u[len2-i-1]
        plt.figure()
        plt.stem(u)
        beta = np.zeros([K,L])
        for k in range(K):
            bin_str = bin(k).replace('0b','')
            for i in range(L-1):
                if i < len(bin_str):
                    beta[k][i+1] = int(bin_str[len(bin_str)-i-1])
        D = np.zeros(K,dtype = int)
        for k in range(K):
            D[k] = 2*L
            for i in range(L):
                D[k] = int(min(D[k],L*(2-beta[k][i])-i))
        ct = [[] for k in range(K)]
        for k in range(K):
            for n in range(D[k]*sample_time):
                temp_product = 1
                for i in range(L):
                    index = int(n+i*sample_time+beta[k][i]*L*sample_time)
                    if index>=len2:
                        temp_product = 0
                        break
                    temp_product *= u[index]
                ct[k].append(temp_product)
        return ct

def Channel(st,N0,sps):
    send_len = len(st)
    w = np.zeros(send_len,dtype=complex)
    sigma = math.sqrt(N0*sps)
    for k in range(send_len):
        w[k] = complex(np.random.normal(0,sigma),np.random.normal(0,sigma))
    theta = np.random.uniform(0,2*math.pi)
    rt = st*complex(math.cos(theta),math.sin(theta))+w
    return rt

def CPM(alpha_seq,BT,h,Lg,sps):
    sig_len = len(alpha_seq)
    sample_time = sps     #每个符号的采样数
    F_cal = sample_time
    L = Lg
    t = np.linspace(-Lg/2,Lg/2,Lg*sample_time+1)     #为精确重构q，采用比原始采样率的5倍进行重构
    len1 = len(t)
    delta = math.sqrt(math.log(2))/(2*math.pi*BT)
    sigma = delta
    f = np.zeros(len1)
    for k in range(len1):
        kernal = lambda tau:1/(math.sqrt(2*math.pi)*sigma)*np.exp(-((t[k]-tau)**2)/(2*(sigma**2)))
        info = quad(kernal,-1/2,1/2)
        f[k] = 1/2*info[0]
    q = np.zeros(len1)
    for i in range(len1-1):
        q[i+1] = q[i] + 1/(F_cal)*f[i+1]
    send_len = int((sig_len + Lg)*F_cal)
    st = np.zeros(send_len,dtype=complex)
    ksai = np.zeros(send_len)
    for n in range(sig_len):
        start = int(n*F_cal)
        ksai[start:start+len1] += 2*h*math.pi*alpha_seq[n]*q
        ksai[start+len1:] += h*math.pi*alpha_seq[n]
    for n in range(send_len):
        st[n] = math.sqrt(2)*complex(math.cos(ksai[n]),math.sin(ksai[n]))
    
    return st

def NonCoherentReceiver(rt,ct,sps,sig_len,L,N,S):
    receive_len = len(rt)
    h_match0 = ct[0][::-1]
    h_match1 = ct[1][::-1]
    xt0 = 1/sps*np.convolve(rt,h_match0)
    xt1 = 1/sps*np.convolve(rt,h_match1)
    x = np.zeros([2,sig_len],dtype = complex)
    for n in range(sig_len):
        x[0,n] = xt0[n*sps+sps-1]
        x[1,n] = xt1[n*sps+sps-1]
    F = np.zeros([2,2,3])
    F[:,:,0] = np.array([0.08024,0.2305,0.0263,0.0421]).reshape(2,2)
    F[:,:,1] = np.array([0.6558,0.45977,0,0]).reshape(2,2)
    F[:,:,2] = np.array([0.4442,0,0,0]).reshape(2,2)
    WF00 = np.array([0.0421])
    WF01 = np.array([0,-0.45977,-0.2305])
    WF10 = np.array([-0.0263])
    WF11 = np.array([0.4442,0.6558,0.0824])
    WF0 = np.array([0.0187,0.0155,-0.0027])
    z0 = scipy.signal.lfilter(WF00,WF0,x[0,...])+scipy.signal.lfilter(WF01,WF0,x[1,...])
    z1 = scipy.signal.lfilter(WF10,WF0,x[0,...])+scipy.signal.lfilter(WF11,WF0,x[1,...])
    z = np.zeros([2,sig_len],dtype = complex)
    z = np.array([z0,z1])
    K_1=  int(np.log2(S))
    path = np.zeros([S,sig_len])
    total_cost = np.zeros(S)
    temp_cost = np.zeros(S)
    alpha_seq = np.zeros([S,2,N+L],dtype = complex)
    temp_alpha = np.zeros([S,2,2,N+L],dtype = complex)
    lamda = np.zeros([S,2])
    y_seq0 = np.zeros([2,N],dtype = complex)
    y_seq1 = np.zeros([2,N],dtype = complex)
    j = complex(0,1)
    for n in range(sig_len):
        if n < K_1:
            if n == 0:
                temp_alpha[0,0,0,0] = -j
                temp_alpha[0,0,1,0] = -j
                temp_alpha[0,1,0,0] = j
                temp_alpha[0,1,1,0] = j
                y_seq0[:,0] = np.matmul(F[:,:,0].T,temp_alpha[0,0,:,0])
                y_seq1[:,1] = np.matmul(F[:,:,1].T,temp_alpha[0,1,:,0])
                lamda[0,0] = abs(z[0][0]*y_seq0[0][0].conjugate()+z[1][0]*y_seq0[1][0].conjugate())
                lamda[0,1] = abs(z[0][0]*y_seq1[0][0].conjugate()+z[1][0]*y_seq1[1][0].conjugate())
                total_cost[0] = lamda[0,0]
                total_cost[1] = lamda[0,1]
                path[0,0] = 0
                path[1,0] = 0
                alpha_seq[0,:,:] = temp_alpha[0,0,:,:]
                alpha_seq[1,:,:] = temp_alpha[0,1,:,:]
            else:
                for s in range(2**(n)):
                    temp_alpha[s,0,0,0] = -j*alpha_seq[s][0][0]
                    temp_alpha[s,1,0,0] = j*alpha_seq[s][0][0]
                    if n == 1:
                        temp_alpha[s,0,1,0] = 1
                        temp_alpha[s,1,1,0] = -1
                    else:
                        temp_alpha[s,0,1,0] = -j*alpha_seq[s][0][1]
                        temp_alpha[s,1,1,0] = j*alpha_seq[s][0][1]
                    temp_alpha[s,0,:,1:N+L] = alpha_seq[s,:,0:N+L-1]
                    temp_alpha[s,1,:,1:N+L] = alpha_seq[s,:,0:N+L-1]
                    for i in range(N):
                        temp0 = np.zeros(2,dtype = complex)
                        temp1 = np.zeros(2,dtype = complex)
                        for l in range(L+1):
                            temp0 += np.matmul(F[:,:,l].T,temp_alpha[s,0,:,i+l])
                            temp1 += np.matmul(F[:,:,l].T,temp_alpha[s,1,:,i+l])
                        y_seq0[:,i] = temp0
                        y_seq1[:,i] = temp1
                    part_sum0 = complex(0,0)
                    part_sum1 = complex(0,0)
                    for k in range(2):
                        for i in range(min(N-1,n-1)):
                            part_sum0 += z[k,n-i-1]*y_seq0[k,i+1].conjugate()
                            part_sum1 += z[k,n-i-1]*y_seq1[k,i+1].conjugate()
                    lamda[s,0] = abs((part_sum0+z[0,n]*y_seq0[0,0].conjugate()+z[1,n]*y_seq0[1,0].conjugate()))-abs(part_sum0)-0.5*(abs(y_seq0[0,0])**2+abs(y_seq1[1,0])**2)
                    lamda[s,1] = abs((part_sum1+z[0,n]*y_seq1[0,0].conjugate()+z[1,n]*y_seq0[1,0].conjugate()))-abs(part_sum1)-0.5*(abs(y_seq1[0,0])**2+abs(y_seq1[1,0])**2)
                for s in range(2**(n+1)):
                    state0 = s//2
                    cost0 = total_cost[state0]+lamda[state0,s%2]
                    temp_cost[s] = cost0
                    path[s,n] = state0
                    alpha_seq[s,...] = temp_alpha[state0,s%2,...]
                total_cost = temp_cost.copy()
                
        else:
            for s in range(S):
                temp_alpha[s,0,0,0] = -j*alpha_seq[s][0][0]
                temp_alpha[s,1,0,0] = j*alpha_seq[s][0][0]
                if n == 1:
                    temp_alpha[s,0,1,0] = 1
                    temp_alpha[s,1,1,0] = -1
                else:
                    temp_alpha[s,0,1,0] = -j*alpha_seq[s][0][1]
                    temp_alpha[s,1,1,0] = j*alpha_seq[s][0][1]
                temp_alpha[s,0,:,1:N+L] = alpha_seq[s,:,0:N+L-1]
                temp_alpha[s,1,:,1:N+L] = alpha_seq[s,:,0:N+L-1]
                for i in range(N):
                    temp0 = np.zeros(2,dtype = complex)
                    temp1 = np.zeros(2,dtype = complex)
                    for l in range(L+1):
                        temp0 += np.matmul(F[:,:,l].T,temp_alpha[s,0,:,i+l])
                        temp1 += np.matmul(F[:,:,l].T,temp_alpha[s,1,:,i+l])
                    y_seq0[:,i] = temp0
                    y_seq1[:,i] = temp1
                part_sum0 = complex(0,0)
                part_sum1 = complex(0,0)
                for k in range(2):
                    for i in range(min(N-1,n-1)):
                        part_sum0 += z[k,n-i-1]*y_seq0[k,i+1].conjugate()
                        part_sum1 += z[k,n-i-1]*y_seq1[k,i+1].conjugate()
                lamda[s,0] = abs((part_sum0+z[0,n]*y_seq0[0,0].conjugate()+z[1,n]*y_seq0[1,0].conjugate()))-abs(part_sum0)-0.5*(abs(y_seq0[0,0])**2+abs(y_seq1[1,0])**2)
                lamda[s,1] = abs((part_sum1+z[0,n]*y_seq1[0,0].conjugate()+z[1,n]*y_seq0[1,0].conjugate()))-abs(part_sum1)-0.5*(abs(y_seq1[0,0])**2+abs(y_seq1[1,0])**2)
            for s in range(S):
                state0 = int(s//2)           #表示首位为0的上一个状态
                state1 = int(s//2+S/2)       #表示首位为1的上一个状态
                cost0 = total_cost[state0]+lamda[state0,s%2]
                cost1 = total_cost[state1]+lamda[state1,s%2]
                if cost0 > cost1:
                    temp_cost[s] = cost0
                    path[s,n] = state0
                    alpha_seq[s,:,:] = temp_alpha[state0,s%2,:,:,]
                else:
                    temp_cost[s] = cost1
                    path[s,n] = state1
                    alpha_seq[s,:,:] = temp_alpha[state1,s%2,:,:]
            total_cost = temp_cost.copy()
    max_id = np.argmax(total_cost)
    cursor = max_id
    decode_seq = np.zeros(sig_len)
    for n in range(sig_len):
        decode_seq[sig_len-1-n] = 2*(cursor%2)-1
        cursor = path[int(cursor),sig_len-1-n]
    return decode_seq

def RunSim():
    N = [2,3,4,5,5,5]
    S = [2,4,4,4,8,32]
    sample_num = [15,13,13,12,12,12]
    BER = [[[] for n in range(sample_num[k])]for k in range(6)]
    N0_seq = [[] for k in range(6)]
    sim_time = 1
    sps = 10
    BT = 0.25
    h = 0.5
    L = 2
    M = 2
    K = 2
    sig_len = 1000000
    ct = CPM2PAM(np.array([1,1]),M,h,L,BT,K,sps)
    for k in range(6):
        N0_seq[k] = [10**(-n/10) for n in range(sample_num[k])]
        for m in range(len(N0_seq[k])):
            temp_BER = 0.0
            for n in range(sim_time):
                alpha = SigGenerate(sig_len)
                st = CPM(alpha,BT,h,L,sps)
                rt = Channel(st,N0_seq[k][m],sps)
                decode_seq = NonCoherentReceiver(rt,ct,sps,sig_len,L,N[k],S[k])
                temp_BER += CalBER(alpha,decode_seq)
            BER[k][m] = temp_BER/sim_time
    file = open('Fig2_K2BER.txt','w')
    for fp in BER:
        file.write(str(fp))
        file.write('\n')
    file.close()
    return

def SigGenerate(sig_len):
    alpha = np.zeros(sig_len)
    for n in range(sig_len):
        alpha[n] = 2*np.random.randint(0,2)-1
    return alpha

def CalBER(send,receive):
    sig_len = len(send)
    error_idx = np.nonzero(receive-send)
    BER = len(error_idx[0])/sig_len
    return BER

if __name__ == '__main__':
    sig_len = 10000
    sps = 10
    alpha = np.zeros(sig_len)
    for n in range(sig_len):
        alpha[n] = 2*(np.random.randint(0,2))-1
        #alpha[n] = 2*(n%2)-1
    st = CPM(alpha,0.25,0.5,2,sps)
    ct = CPM2PAM(alpha,2,0.5,2,0.25,2,sps)
    n0 = 0
    rt = Channel(st,n0,sps)
    decode_seq = NonCoherentReceiver(rt,ct,sps,sig_len,2,2,2)
    diff = decode_seq-alpha
    error_idx = np.nonzero(diff)
    print(len(error_idx[0]))
    #RunSim()
