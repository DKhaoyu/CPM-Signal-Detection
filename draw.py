import numpy as np
import matplotlib.pyplot as plt

def drawFig2():
    BER1 = [0.4078, 0.3913, 0.3727, 0.3382, 0.3173, 0.2634, 0.2268, 0.1859, 0.1542, 0.1219, 0.0937, 0.069, 0.0453, 0.0306, 0.0216]
    BER2 = [0.2562, 0.2257, 0.1782, 0.1436, 0.0957, 0.0715, 0.04079, 0.02477, 0.01175, 0.00475, 0.0018533333333333334, 0.00047, 0.00018]
    BER3 = [0.2573, 0.2082, 0.1848, 0.1228, 0.0868, 0.0555, 0.0307, 0.01664, 0.00757, 0.00274, 0.00103, 0.00030333333333333335, 7.333333333333333e-05]
    BER4 = [0.2465, 0.2064, 0.1628, 0.1193, 0.0761, 0.0427, 0.02468, 0.01172, 0.00551, 0.00186, 0.00047, 0.00013333333333333334]
    BER5 = [0.221, 0.1707, 0.1337, 0.0837, 0.061, 0.0369, 0.01746, 0.00855, 0.00312, 0.00132, 0.00044, 0.00010666666666666667]
    BER6 = [0.2141, 0.1497, 0.1201, 0.077, 0.0455, 0.026, 0.01372, 0.00609, 0.00205, 0.00053, 0.00016, 3e-05]
    SNR = range(15)
    plt.figure()
    plt.plot(SNR,BER1,color='black',linestyle='-',marker = '>',linewidth = 0.5)
    plt.plot(SNR[0:13],BER2,color='black',linestyle='-',marker = '^',linewidth = 0.5)
    plt.plot(SNR[0:13],BER3,color='black',linestyle='-',marker = '<',linewidth = 0.5)
    plt.plot(SNR[0:12],BER4,color='black',linestyle='-',marker = 'd',linewidth = 0.5)
    plt.plot(SNR[0:12],BER5,color='black',linestyle='-',marker = 's',linewidth = 0.5)
    plt.plot(SNR[0:12],BER6,color='black',linestyle='-',marker = 'o',linewidth = 0.5)
    plt.xlabel('Eb/n0   [dB]')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend(["N=2;S=2","N=3;S=4","N=4;S=4","N=5;S=4","N=5;S=8","N=5;S=32"])
    plt.grid(which="both",linestyle = '--',linewidth = 0.2)
    return
def drawFig2_K2():
    BER1 = [0.3205, 0.2874, 0.2588, 0.2257, 0.1887, 0.1515, 0.1316, 0.0923, 0.0719, 0.0543, 0.0384, 0.0231, 0.0135, 0.008, 0.0031]
    BER2 = [0.2784, 0.2229, 0.1927, 0.1507, 0.1167, 0.0825, 0.05043, 0.03202, 0.01712, 0.00864, 0.0036666666666666666, 0.0012266666666666667, 0.00031666666666666665]
    BER3 = [0.2717, 0.2198, 0.1817, 0.1404, 0.096, 0.0604, 0.04212, 0.02071, 0.00954, 0.00462, 0.0014266666666666666, 2.767e-4, 1.233e-4]
    BER4 = [0.2601, 0.2294, 0.1704, 0.1089, 0.0891, 0.0516, 0.03246, 0.01678, 0.00739, 0.00305, 9.667e-4, 2.1333e-4]
    BER5 = [0.2466, 0.1952, 0.1467, 0.1186, 0.0848, 0.0497, 0.0245, 0.01191, 0.00483, 0.0018, 4.133e-4, 1.0667e-4]
    BER6 = [0.2538, 0.1947, 0.1505, 0.1228, 0.0717, 0.0382, 0.02021, 0.00999, 0.00358, 0.00125, 3.67e-4, 2e-5]
    SNR = range(15)
    plt.figure()
    plt.plot(SNR,BER1,color='black',linestyle='-',marker = '>',linewidth = 0.5)
    plt.plot(SNR[0:13],BER2,color='black',linestyle='-',marker = '^',linewidth = 0.5)
    plt.plot(SNR[0:13],BER3,color='black',linestyle='-',marker = '<',linewidth = 0.5)
    plt.plot(SNR[0:12],BER4,color='black',linestyle='-',marker = 'd',linewidth = 0.5)
    plt.plot(SNR[0:12],BER5,color='black',linestyle='-',marker = 's',linewidth = 0.5)
    plt.plot(SNR[0:12],BER6,color='black',linestyle='-',marker = 'o',linewidth = 0.5)
    plt.xlabel('Eb/n0   [dB]')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend(["N=2;S=2","N=3;S=4","N=4;S=4","N=5;S=4","N=5;S=8","N=5;S=32"])
    plt.grid(which="both",linestyle = '--',linewidth = 0.2)
    return
def drawFig3():
    BER1 = 0.5*np.array([0.449, 0.354, 0.28, 0.266, 0.195, 0.145, 0.106, 0.073, 0.044, 0.0182, 0.0107, 0.00309, 0.00078])
    BER2 = 0.5*np.array([0.384, 0.345, 0.282, 0.242, 0.219, 0.125, 0.082, 0.049, 0.024, 0.0109, 0.00335, 0.00081, 0.00031])
    BER3 = 0.5*np.array([0.43, 0.336, 0.262, 0.228, 0.176, 0.144, 0.068, 0.042, 0.0168, 0.0066, 0.0028, 0.00054, 0.00013])
    BER4 = 0.5*np.array([0.408, 0.327, 0.252, 0.216, 0.121, 0.105, 0.061, 0.0366, 0.0162, 0.0065, 0.00165, 0.00032,0.0001])
    SNR = range(13)
    plt.figure()
    plt.plot(SNR,BER1,color='black',linestyle='-',marker = '<',linewidth = 0.5,markerfacecolor = 'white')
    plt.plot(SNR,BER2,color='black',linestyle='-',marker = '^',linewidth = 0.5,markerfacecolor = 'white')
    plt.plot(SNR,BER3,color='black',linestyle='-',marker = 'd',linewidth = 0.5,markerfacecolor = 'white')
    plt.plot(SNR,BER4,color='black',linestyle='-',marker = 's',linewidth = 0.5,markerfacecolor = 'white')
    plt.xlabel('Eb/n0   [dB]')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend(["N=3;S=4","N=3;S=16","N=4;S=16","N=4;S=64"])
    plt.grid(which="both",linestyle = '--',linewidth = 0.2)
    return
def drawFig3_K3():
    BER1 = 0.5*np.array([0.4077, 0.3655, 0.3251, 0.247, 0.2042, 0.1456, 0.1044, 0.0671, 0.0328, 0.0146, 0.0092, 0.0012, 0.0004])
    BER2 = 0.5*np.array([0.3847, 0.3366, 0.2831, 0.2229, 0.1783, 0.1224, 0.0833, 0.0436, 0.0221, 0.00898, 0.00306, 7.867e-4, 2.1e-4])
    BER3 = 0.5*np.array([0.3967, 0.3438, 0.2754, 0.2245, 0.1556, 0.1035, 0.0756, 0.039, 0.0172, 0.00709, 0.00216, 4.6e-4, 1e-4])
    BER4 = 0.5*np.array([0.3883, 0.342, 0.275, 0.201, 0.172, 0.118, 0.056, 0.0332, 0.0142, 0.0048, 0.00185, 0.00025, 8e-5])
    SNR = range(13)
    plt.figure()
    plt.plot(SNR,BER1,color='black',linestyle='-',marker = '<',linewidth = 0.5,markerfacecolor = 'white')
    plt.plot(SNR,BER2,color='black',linestyle='-',marker = '^',linewidth = 0.5,markerfacecolor = 'white')
    plt.plot(SNR,BER3,color='black',linestyle='-',marker = 'd',linewidth = 0.5,markerfacecolor = 'white')
    plt.plot(SNR,BER4,color='black',linestyle='-',marker = 's',linewidth = 0.5,markerfacecolor = 'white')
    plt.xlabel('Eb/n0   [dB]')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend(["N=3;S=4","N=3;S=16","N=4;S=16","N=4;S=64"])
    plt.grid(which="both",linestyle = '--',linewidth = 0.2)
    plt.show()
    return
if __name__  == '__main__':
    drawFig2()
    drawFig2_K2()
    drawFig3()
    drawFig3_K3()