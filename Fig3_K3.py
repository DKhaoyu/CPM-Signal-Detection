import numpy as np
import math
import scipy.signal 
from scipy.integrate import quad
import matplotlib.pyplot as plt
def CPM(alpha_seq,L,h,sps):
    q = np.zeros(L*sps+1)
    for n in range(L*sps):
        q[n+1] = 1/(2*L)*((n+1)/sps-L/(2*math.pi)*math.sin(2*math.pi*(n+1)/L/sps))
    sig_len = len(alpha_seq)
    send_len = (sig_len+L)*sps
    ksai = np.zeros(send_len)
    st = np.zeros(send_len,dtype = complex)
    for n in range(sig_len):
        ksai[n*sps:(n+L)*sps+1] += 2*h*math.pi*alpha_seq[n]*q
        ksai[(n+L)*sps+1:] += h*math.pi*alpha_seq[n]
    for n in range(send_len):
        st[n] = 2*complex(math.cos(ksai[n]),math.sin(ksai[n]))

    return st

def CPM2PAM(alpha_seq,h,K,sps):
    #默认情况K=3
    sig_len = len(alpha_seq)
    L = 2
    P = 2
    len1 = L*sps+1
    len2 = 2*L*sps+1
    P = 2
    q = np.zeros(len1)
    for n in range(len1):
        t = n/sps
        q[n] = 1/(2*L)*(t-L/(2*math.pi)*math.sin(2*math.pi*t/L))
    alpha_hat = (alpha_seq+3)//2
    gama_hat = np.zeros([sig_len,P])
    for i in range(sig_len):
        bin_str = bin(alpha_hat[i]).replace('0b','')
        bin_num = len(bin_str)
        for l in range(P):
            if l < bin_num:
                gama_hat[i][l] = int(bin_str[bin_num-1-l])
    gama = 2*gama_hat-1
    Q = 2**(L-1)
    beta = np.zeros([Q,L])
    for k in range(Q):
        bin_str = bin(k).replace('0b','')
        for i in range(L-1):
            if i < len(bin_str):
                beta[k][i+1] = int(bin_str[len(bin_str)-i-1])

    u = np.zeros([2,len2])
    for l in range(2):
        hl = 2**(l)*h
        for i in range(len2):
            if i < len1:
                u[l][i] = math.sin(2*hl*math.pi*q[i])/math.sin(hl*math.pi)
            else:
                u[l][i] = u[l][len2-i-1]
    #假设K在0~4之间，即只取表V中的第一部分，因此ct有c00(t)和c01(t)两种组成,由于j=0，所以ct的长度均为3
    ct = np.zeros([L,(L+1)*sps])
    for l in range(2):
        for n in range((L+1)*sps):
            temp = 1.0 
            for i in range(L):
                index = n + i*sps
                if index < len2:
                    temp *= u[l][index]
                else:
                    temp = 0.0
                    break
            ct[l][n] = temp
    gt = np.zeros([K,(L+1)*sps])
    for k in range(K):
        for n in range((L+1)*sps):
            if k == 0:
                gt[k][n] = ct[0][n]*ct[1][n]
            elif (k == 1)and(n<2*sps):
                gt[k][n] = ct[0][n+sps]*ct[1][n]
            elif (k == 2)and(n<2*sps):
                gt[k][n] = ct[0][n]*ct[1][n+sps]
            elif (k == 3)and(n<sps):
                gt[k][n] = ct[0][n+2*sps]*ct[1][n]
            elif (k == 4)and(n<sps):
                gt[k][n] = ct[0][n]*ct[1][n+2*sps]

    aver_gt = np.zeros([3,(L+1)*sps])
    aver_gt[0,...] = gt[0]
    aver_gt[1,...] = 1/2*(gt[1]+gt[2])
    aver_gt[2,...] = 1/2*(gt[3]+gt[4])
    #只是用于调试，验证CPM2PAM波形上的正确性，在实际的通信系统中不需要提供，考虑在K=3的情况下
    F = np.zeros([3,3,2])
    F[:,:,0] = np.array([0.1532,0.20435,0.0053,-0.0302,-0.019193,0.0028,-0.1278995,-0.079034,0.0043]).reshape([3,3])
    F[:,:,1] = np.array([0.6105,0.0379,0,0.2923,0.1778,0,0,0,0]).reshape([3,3])
    F[:,:,0] = F[:,:,0].T
    F[:,:,1] = F[:,:,1].T
    b = np.zeros([sig_len,2],dtype = complex)
    temp_gama_sum = np.zeros([sig_len,2])
    for l in range(2):
        hl = 2**(l)*h
        temp_gama_sum[0][l] = gama[0][l]
        for n in range(sig_len-1):
            temp_gama_sum[n+1][l] = temp_gama_sum[n][l] + gama[n+1][l]
    for l in range(2):
        hl = 2**(l)*h
        for n in range(sig_len):
            theta = hl*math.pi*temp_gama_sum[n][l]
            b[n][l] = complex(math.cos(theta),math.sin(theta))
    pesudo = np.zeros([sig_len,5],dtype = complex)
    for n in range(sig_len):
        if n == 0:
            pesudo[n][0] = b[n][0]*b[n][1]
            pesudo[n][1] = b[n][0]
            pesudo[n][2] = b[n][1]
        elif n == 1:
            pesudo[n][0] = b[n][0]*b[n][1]
            pesudo[n][1] = b[n-1][0]*b[n][1]
            pesudo[n][2] = b[n][0]*b[n-1][1]
        else:
            pesudo[n][0] = b[n][0]*b[n][1]
            pesudo[n][1] = b[n-1][0]*b[n][1]
            pesudo[n][2] = b[n][0]*b[n-1][1]
            pesudo[n][3] = b[n-2][0]*b[n][1]
            pesudo[n][4] = b[n][0]*b[n-2][1]
    send_len = (sig_len + L)*sps
    st1 = np.zeros(send_len,dtype = complex)
    for n in range(sig_len):
        st1[n*sps:(n+L+1)*sps] += pesudo[n][0]*gt[0]
        st1[n*sps:(n+L+1)*sps] += pesudo[n][1]*gt[1]
        st1[n*sps:(n+L+1)*sps] += pesudo[n][1]*gt[2]

    temp_alpha = np.zeros([3,sig_len],dtype = complex)
    temp_alpha[0,:] = pesudo[:,0]
    temp_alpha[1,:] = pesudo[:,1]+pesudo[:,2]
    temp_alpha[2,:] = pesudo[:,3]+pesudo[:,4]
    y = np.zeros([3,sig_len],dtype = complex)
    for n in range(sig_len):
        if n == 0:
            y[:,0] = np.matmul(F[:,:,0].T,temp_alpha[:,0])
        else:
            y[:,n] = np.matmul(F[:,:,0].T,temp_alpha[:,n])+np.matmul(F[:,:,1].T,temp_alpha[:,n-1])

    return aver_gt,y

def SigGenerate(sig_len):
    alpha = np.zeros(sig_len,dtype = int)
    for n in range(sig_len):
        sign = (-1)**(np.random.randint(0,2))
        alpha[n] = sign*(2*np.random.randint(0,2)+1)
    return alpha

def Channel(st,N0,sps):
    send_len = len(st)
    w = np.zeros(send_len,dtype=complex)
    sigma = math.sqrt(N0*sps)
    for k in range(send_len):
        w[k] = complex(np.random.normal(0,sigma),np.random.normal(0,sigma))
    theta = np.random.uniform(0,2*math.pi)
    rt = st*complex(math.cos(theta),math.sin(theta))+w
    return rt

def NonCoherentReceiver(rt,ct,sig_len,L,N,S,h,sps,y):
    receive_len = len(rt)
    filter_len = (L+2)*sps
    h_match0 = ct[0][::-1]
    h_match1 = ct[1][::-1]
    h_match2 = ct[2][::-1]
    xt = np.zeros([3,receive_len+filter_len-1],dtype = complex)
    xt[0,...] = 1/sps*np.convolve(rt,h_match0)
    xt[1,...] = 1/sps*np.convolve(rt,h_match1)
    xt[2,...] = 1/sps*np.convolve(rt,h_match2)
    x = np.zeros([3,sig_len],dtype = complex)
    for n in range(sig_len):
        x[0,n] = xt[0,n*sps+sps-1]
        x[1,n] = xt[1,n*sps+sps-1]
        x[2,n] = xt[2,n*sps+sps-1]

    F = np.zeros([3,3,2])
    F[:,:,0] = np.array([0.1532,0.20435,0.0053,-0.0302,-0.019193,0.0028,-0.1278995,-0.079034,0.0043]).reshape([3,3])
    F[:,:,1] = np.array([0.6105,0.0379,0,0.2923,0.1778,0,0,0,0]).reshape([3,3])
    F[:,:,0] = F[:,:,0].T
    F[:,:,1] = F[:,:,1].T
    WF11 = np.array([-38.767,-7.03616])
    WF12 = np.array([63.719,11.57714])
    WF13 = np.array([18.3094,3.4453])
    WF21 = np.array([8.26484,65.803])
    WF22 = np.array([-133.138,-67.7852])
    WF23 = np.array([-2200.548,711.0978])
    WF31 = np.array([0,42.3998,-34.172])
    WF32 = np.array([0,8.12476,29.8682])
    WF33 = np.array([-4943.14,2183.314,-163.833])
    WF0 = np.array([-21.25,3.325,1.305])
    z0 = scipy.signal.lfilter(WF11,WF0,x[0,...])+scipy.signal.lfilter(WF12,WF0,x[1,...])+scipy.signal.lfilter(WF13,WF0,x[2,...])
    z1 = scipy.signal.lfilter(WF21,WF0,x[0,...])+scipy.signal.lfilter(WF22,WF0,x[1,...])+scipy.signal.lfilter(WF23,WF0,x[2,...])
    z2 = scipy.signal.lfilter(WF31,WF0,x[0,...])+scipy.signal.lfilter(WF32,WF0,x[1,...])+scipy.signal.lfilter(WF33,WF0,x[2,...])
    #z = np.array([z0,z1])
    z = np.zeros([3,sig_len],dtype = complex)
    z[0,0:sig_len-1] = z0[1:sig_len]
    z[1,0:sig_len-1] = z1[1:sig_len]
    z[2,0:sig_len-2] = z2[2:sig_len]
    plt.figure()
    plt.subplot(3,2,1)
    plt.stem(np.real(y[0,:]-0.5*z[0,:]))
    plt.subplot(3,2,2)
    plt.stem(np.imag(y[0,:]-0.5*z[0,:]))
    plt.subplot(3,2,3)
    plt.stem(np.real(y[1,:]-0.5*z[1,:]))
    plt.subplot(3,2,4)
    plt.stem(np.imag(y[1,:]-0.5*z[1,:]))
    plt.subplot(3,2,5)
    plt.stem(np.real(y[2,:]-0.5*z[2,:]))
    plt.subplot(3,2,6)
    plt.stem(np.imag(y[2,:]-0.5*z[2,:]))
    plt.show()
    #维特比部分
    K_1 = int(np.log2(S)/2)
    path = np.zeros([S,sig_len],dtype=int)
    total_cost = np.zeros(S)
    temp_cost = np.zeros(S)
    alpha_seq = np.zeros([S,3,N+L],dtype = complex)
    temp_alpha = np.zeros([S,4,3,N+L],dtype = complex)
    lamda = np.zeros([S,4])
    y_seq = np.zeros([3,N],dtype = complex)
    gama_dict = {0:[-1,-1],1:[1,-1],2:[-1,1],3:[1,1]}
    old_b = np.zeros([2,2,S],dtype = complex)
    new_b = np.zeros([2,2,S,4],dtype = complex)     #第一维表示是b01还是b00，第二位表示是n还是n-1
    for n in range(sig_len):
        if n < K_1:
            if n == 0:
                for next_step in range(4):
                    gama = gama_dict[next_step]
                    new_b[0,0,0,next_step] = complex(math.cos(h*math.pi*gama[0]),math.sin(h*math.pi*gama[0]))
                    new_b[1,0,0,next_step] = complex(math.cos(2*h*math.pi*gama[1]),math.sin(2*h*math.pi*gama[1]))
                    temp_alpha[0,next_step,0,0] = new_b[0,0,0,next_step]*new_b[1,0,0,next_step] 
                    temp_alpha[0,next_step,1,0] = new_b[0,0,0,next_step]+new_b[1,0,0,next_step]
                    temp_alpha[0,next_step,2,0] = new_b[0,0,0,next_step]+new_b[1,0,0,next_step]
                    y_seq[...,0] = np.matmul(F[:,:,0].T,temp_alpha[0,next_step,:,0])
                    lamda[0,next_step] = abs(z[0][n]*y_seq[0][n].conjugate()+z[1][n]*y_seq[1][n].conjugate()+z[2][n]*y_seq[2][n].conjugate())-0.5*np.linalg.norm(y_seq[:,0])**2
                    total_cost[next_step] = lamda[0,next_step]
                    path[next_step,n] = 0
                    alpha_seq[next_step,:,:] = temp_alpha[0,next_step,:,:].copy()
                    old_b[:,:,next_step] = new_b[:,:,0,next_step]
            else:
                for s in range(4**(n)):
                    for next_step in range(4):
                        gama = gama_dict[next_step]
                        new_b[0,:,s,next_step] = np.array([old_b[0,0,s]*complex(math.cos(h*math.pi*gama[0]),math.sin(h*math.pi*gama[0])),old_b[0,0,s]])
                        new_b[1,:,s,next_step] = np.array([old_b[1,0,s]*complex(math.cos(2*h*math.pi*gama[1]),math.sin(2*h*math.pi*gama[1])),old_b[1,0,s]])
                        temp_alpha[s,next_step,0,0] = new_b[0,0,s,next_step]*new_b[1,0,s,next_step]
                        temp_alpha[s,next_step,1,0] = new_b[0,0,s,next_step]*old_b[1,0,s]+new_b[1,0,s,next_step]*old_b[0,0,s]
                        if n == 2:
                            temp_alpha[s,next_step,2,0] = new_b[0,0,s,next_step]+new_b[1,0,s,next_step]
                        else:
                            temp_alpha[s,next_step,2,0] = new_b[0,0,s,next_step]*old_b[1,1,s]+new_b[1,0,s,next_step]*old_b[0,1,s]
                        temp_alpha[s,next_step,:,1:N+L] = alpha_seq[s,:,0:N+L-1].copy()
                        for i in range(N):
                            y_seq[...,i] = np.matmul(F[:,:,0].T,temp_alpha[s,next_step,:,i]) + np.matmul(F[:,:,1].T,temp_alpha[s,next_step,:,i+1])
                        part_sum1 = complex(0,0)
                        for k in range(3):
                            for i in range(min(N-1,n-1)):
                                part_sum1 += z[k][n-i-1]*(y_seq[k][i+1].conjugate())
                        lamda[s,next_step] = abs(part_sum1+z[0][n]*(y_seq[0][0].conjugate())+z[1][n]*(y_seq[1][0].conjugate())+z[2][n]*(y_seq[2][0].conjugate()))-0.5*abs(part_sum1)-0.5*np.linalg.norm(y_seq[:,0])**2
                for s in range(4**(n+1)):
                    state0 = s//4
                    cost0 = total_cost[state0]+lamda[state0,s%4]
                    temp_cost[s] = cost0
                    path[s,n] = state0
                    alpha_seq[s,:,:] = temp_alpha[state0,s%4,:,:].copy()
                    old_b[:,:,s] = new_b[:,:,state0,s%4].copy()
                total_cost = temp_cost.copy()  
        else:
            for s in range(S):
                for next_step in range(4):
                    gama = gama_dict[next_step]
                    new_b[0,:,s,next_step] = np.array([old_b[0,0,s]*complex(math.cos(h*math.pi*gama[0]),math.sin(h*math.pi*gama[0])),old_b[0,0,s]])
                    new_b[1,:,s,next_step] = np.array([old_b[1,0,s]*complex(math.cos(2*h*math.pi*gama[1]),math.sin(2*h*math.pi*gama[1])),old_b[1,0,s]])
                    temp_alpha[s,next_step,0,0] = new_b[0,0,s,next_step]*new_b[1,0,s,next_step]
                    temp_alpha[s,next_step,1,0] = new_b[0,0,s,next_step]*old_b[1,0,s]+new_b[1,0,s,next_step]*old_b[0,0,s]
                    if n == 2:
                        temp_alpha[s,next_step,2,0] = new_b[0,0,s,next_step]+new_b[1,0,s,next_step]
                    else:
                        temp_alpha[s,next_step,2,0] = new_b[0,0,s,next_step]*old_b[1,1,s]+new_b[1,0,s,next_step]*old_b[0,1,s]
                    temp_alpha[s,next_step,:,1:N+L] = alpha_seq[s,:,0:N+L-1].copy()
                    for i in range(N):
                        y_seq[...,i] = np.matmul(F[:,:,0].T,temp_alpha[s,next_step,:,i]) + np.matmul(F[:,:,1].T,temp_alpha[s,next_step,:,i+1])
                    part_sum1 = complex(0,0)
                    for k in range(3):
                        for i in range(min(N-1,n-1)):
                            part_sum1 += z[k][n-i-1]*(y_seq[k][i+1].conjugate())
                    lamda[s,next_step] = abs(part_sum1+z[0][n]*y_seq[0][0].conjugate()+z[1][n]*y_seq[1][0].conjugate()+z[2][n]*y_seq[2][0].conjugate())-0.5*abs(part_sum1)-0.5*np.linalg.norm(y_seq[:,0])**2
            for s in range(S):
                state0 = int(s//4)
                state1 = int(s//4+S/4)
                state2 = int(s//4+2*S/4)
                state3 = int(s//4+3*S/4)
                cost0 = lamda[state0,s%4] + total_cost[state0]
                cost1 = lamda[state1,s%4] + total_cost[state1]
                cost2 = lamda[state2,s%4] + total_cost[state2]
                cost3 = lamda[state3,s%4] + total_cost[state3]
                cost = np.array([cost0,cost1,cost2,cost3])
                state = np.array([state0,state1,state2,state3])
                id = np.argmax(cost)
                temp_cost[s] = cost[id]
                path[s,n] = state[id]
                alpha_seq[s,:,:] = temp_alpha[state[id],s%4,:,:].copy()
                old_b[:,:,s] = new_b[:,:,state[id],s%4].copy()
            total_cost = temp_cost.copy()

    max_id = np.argmax(total_cost)
    cursor = max_id
    code_dict = {0:-3,1:-1,2:1,3:3}
    decode_seq = np.zeros(sig_len)
    for n in range(sig_len):
        decode_seq[sig_len-n-1] = code_dict[cursor%4]
        cursor = path[cursor,sig_len-1-n]
    return decode_seq

def CalBER(send,receive):
    sig_len = len(send)
    error_idx = np.nonzero(receive-send)
    BER = len(error_idx[0])/sig_len
    return BER
def RunSim():
    N = [3,3,4,4]
    S = [4,16,16,64]
    sample_num = [13,13,13,13]
    BER = [[[] for n in range(sample_num[k])]for k in range(4)]
    N0_seq = [[] for k in range(4)]
    sim_time = 1
    sps = 10
    sig_len = 100000
    ct = CPM2PAM(np.array([0]),0.25,5,sps)
    for k in range(4):
        N0_seq[k] = [10**(-n/10) for n in range(sample_num[k])]
        for m in range(len(N0_seq[k])):
            temp_BER = 0.0
            for n in range(sim_time):
                alpha = SigGenerate(sig_len)
                st = CPM(alpha,2,0.25,sps)
                rt = Channel(st,N0_seq[k][m],sps)
                decode_seq = NonCoherentReceiver(rt,ct,sig_len,1,N[k],S[k],0.25,sps)
                temp_BER += CalBER(alpha,decode_seq)
            BER[k][m] = temp_BER/sim_time
    file = open('Fig3_K3BER.txt','w')
    for fp in BER:
        file.write(str(fp))
        file.write('\n')
    file.close()
    return
if __name__ == '__main__':
    '''L = 2
    h = 0.25
    K = 5
    sps = 10
    sig_len = 50
    alpha_seq = SigGenerate(sig_len)
    st = CPM(alpha_seq,L,h,sps)
    gt,y = CPM2PAM(alpha_seq,h,K,sps)
    rt = Channel(st,0.1,sps)
    decode_seq = NonCoherentReceiver(st,gt,sig_len,1,4,16,h,sps,y)
    BER = CalBER(alpha_seq,decode_seq)
    print(BER)'''
    RunSim()




