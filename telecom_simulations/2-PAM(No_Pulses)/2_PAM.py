import random
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np

def get_bits(result):
    for i in range(len(result)):
        if random.random() >= 0.5:
            result[i] = 1

def get_symbols(result, b):
    for i in range(len(result)):
        if b[i] == 1:
            result[i] = 1

def get_noise(result, pn):
    q = []
    for i in range(len(result)):
        q.append(np.random.normal())
    for i in range(len(result)):
        result[i] = math.sqrt(pn) * q[i]

def get_signal_and_noise(s, n, r):
    for i in range(len(r)):
        r[i] = s[i] + n[i]

def sign(r, symbols):
    for i in range(len(r)):
        if r[i] >= 0:
            symbols[i] = 1
        else:
            symbols[i] = -1

def get_received_bits(s, bits):
    for i in range(len(s)):
        if s[i] == 1:
            bits[i] = 1
        else:
            bits[i] = 0

def get_energy(s):
    sum = 0
    for i in range(len(s)):
        sum += s[i]**2

def get_power(s):
    sum = 0
    for i in range(len(s)):
        sum += s[i]**2
    return sum / len(s)

def get_mean_noise(s):
    sum = 0
    for i in range(len(s)):
        sum += s[i]
    return sum / len(s)

n = 1000
bits = [0] * n
symbols = [-1] * n
noise = [0] * n
s_plus_n = [0] * n
received_symbols = [-1] * n
received_bits = [0] * n
es = []
ps = []
pn = []
pes = []
peb = []
pbe = []
snr_db = np.arange(-5, 14, 2.5)
signal_power = 0

get_bits(bits)
get_symbols(symbols, bits)

# question 7 starts
# snr = 10**(12.5/10)
# get_noise(noise, 1/snr)
# get_signal_and_noise(symbols, noise, s_plus_n)

# r = np.array(s_plus_n)

# plt.figure()
# plt.scatter(r, np.zeros(len(r)), s=5)

# plt.xlabel("In-Phase")
# plt.ylabel("Quadrature")
# plt.title("Constellation Diagram (SNR = 12.5 dB)")
# plt.grid(True)
# plt.show()
# question 7 ends

#question 1 starts
# signal_power = get_power(symbols)
# print(f"signal power {signal_power}")
#question 1 ends

for i in np.linspace(-5, 12.5, 8):
    snr = 10**(i/10)

    #question 4 starts
    # snr = 10**(0/10)
    # get_noise(noise, 1/snr)
    # noise_power = get_power(noise)
    # print(f"noise power {noise_power}")
    #question 4 ends

    #question 2 starts
    # snr = 10**(1/10)
    # get_noise(noise, 1/snr)
    # mean_noise = get_mean_noise(noise)
    # print(f"mean noise {mean_noise}")
    #question 2 ends

    #question 6 starts you change n to 2**22 and use the snr below
    #snr = 10**(0/10)
    #you have to use all the code that is inside the first for loop
    #question 6 ends  and the result is the first value of this line print(f"mistakes {eb}")
    get_noise(noise, 1/snr)
    get_signal_and_noise(symbols, noise, s_plus_n)
    sign(s_plus_n, received_symbols)
    get_received_bits(received_symbols, received_bits)
    es = 0
    eb = 0
    for j in range(len(symbols)):
        if symbols[j] != received_symbols[j]:
            es += 1
        if bits[j] != received_bits[j]:
            eb += 1
    pes.append(es/n)
    peb.append(eb/n)
    # print(f"mistakes {eb}")
    pbe.append(float(norm.sf(math.sqrt(snr))))

#question 8 starts with all the code inside the first for loop
plt.plot(snr_db, peb, 'o', label="Simulated BER")
plt.plot(snr_db, pbe, '-', label="Theoretical BER")

plt.yscale('log')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid(True, which="both")
plt.show()
#question 8

#question 9 starts with all the code inside the first for loop
# plt.figure()
# plt.semilogy(snr_db, peb, 'o', label="Simulated BER")
# plt.semilogy(snr_db, pbe, '-', label="Theoretical BER")
# plt.xlabel("SNR (dB)")
# plt.ylabel("BER")
# plt.legend()
# plt.grid(True, which="both")
# plt.show()

# plt.figure()
# plt.semilogy(snr_db, pes, 'o', label="Simulated SER")
# plt.semilogy(snr_db, pbe, '-', label="Theoretical SER")
# plt.xlabel("SNR (dB)")
# plt.ylabel("SER")
# plt.legend()
# plt.grid(True, which="both")
# plt.show()
#question 9 ends

#question 5 starts
# Ιστόγραμμα (κανονικοποιημένο ώστε να είναι πυκνότητα)
# samples = np.random.normal(0, 1, n)
# plt.hist(samples, bins=50, density=True)

# Θεωρητική καμπύλη Gaussian
# x = np.linspace(-4, 4, 1000)
# plt.plot(x, norm.pdf(x, 0, 1))

# plt.xlabel("x")
# plt.ylabel("Probability Density")
# plt.title("Standard Gaussian Distribution (μ=0, σ=1)")
# plt.grid(True)

# plt.show()
#question 5 ends