import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import numpy as np

def get_symbols(b):
    b = np.asarray(b)
    val = 1 / np.sqrt(2)

    pairs = b.reshape(-1, 2)
    symbols = np.empty(len(pairs), dtype=complex)

    mask00 = (pairs[:,0] == 0) & (pairs[:,1] == 0)
    mask01 = (pairs[:,0] == 0) & (pairs[:,1] == 1)
    mask10 = (pairs[:,0] == 1) & (pairs[:,1] == 0)
    mask11 = (pairs[:,0] == 1) & (pairs[:,1] == 1)

    symbols[mask00] = complex(+val, +val)
    symbols[mask01] = complex(-val, +val)
    symbols[mask10] = complex(+val, -val)
    symbols[mask11] = complex(-val, -val)

    return symbols

def get_rayleigh(length):
    value = 1/np.sqrt(2)

    re = np.random.normal(0, value, length)
    im = np.random.normal(0, value, length)

    return re + 1j*im

def get_noise(pn, n):
    value = 1/np.sqrt(2)

    re = np.random.normal(0, 1, n)
    im = np.random.normal(0, 1, n)

    noise = (re * value * np.sqrt(pn)) + 1j * (im * value * np.sqrt(pn))
    return noise

def detector(ds):
    ds = np.asarray(ds, dtype=complex)
    val = 1/math.sqrt(2)

    detected = np.zeros_like(ds, dtype=complex)

    mask = (ds.real > 0) & (ds.imag > 0)
    detected[mask] = complex(val, val)

    mask = (ds.real < 0) & (ds.imag >= 0)
    detected[mask] = complex(-val, val)

    mask = (ds.real < 0) & (ds.imag < 0)
    detected[mask] = complex(-val, -val)

    mask = (ds.real >= 0) & (ds.imag < 0)
    detected[mask] = complex(val, -val)

    return detected

def get_received_bits(symbols):
    symbols = np.asarray(symbols, dtype=complex)
    N = len(symbols)
    
    r_bits = np.zeros(2 * N, dtype=int)
    
    mask = (symbols.real > 0) & (symbols.imag > 0)
    r_bits[2*mask.nonzero()[0]] = 0
    r_bits[2*mask.nonzero()[0]+1] = 0

    mask = (symbols.real < 0) & (symbols.imag >= 0)
    r_bits[2*mask.nonzero()[0]] = 0
    r_bits[2*mask.nonzero()[0]+1] = 1

    mask = (symbols.real < 0) & (symbols.imag < 0)
    r_bits[2*mask.nonzero()[0]] = 1
    r_bits[2*mask.nonzero()[0]+1] = 1

    mask = (symbols.real >= 0) & (symbols.imag < 0)
    r_bits[2*mask.nonzero()[0]] = 1
    r_bits[2*mask.nonzero()[0]+1] = 0

    return r_bits
    
def mrcSER(snr, L):
    integrand = lambda theta: (1 + (snr/2) / (np.sin(theta)**2))**(-L)
    ser, _ = quad(integrand,0,3*np.pi/4)
    return ser / np.pi

def mrcBER(snr, L):
    integrand = lambda theta: (1 + snr / (np.sin(theta)**2))**(-L)
    ber, _ = quad(integrand,0,np.pi/2)
    return ber / np.pi

def question_8(ray1, ray2, symb):
    snr = 10**(15/10)
    y1 = symb * ray1 + get_noise(1/snr, len(symb))
    y2 = symb * ray2 + get_noise(1/snr, len(symb))
    numerator = (
        np.conjugate(ray1) * y1 +
        np.conjugate(ray2) * y2
    )

    denominator = (
        np.abs(ray1)**2 +
        np.abs(ray2)**2
    )
    p = numerator / denominator
    
    detected = detector(p)
    ser = np.mean(symb != detected)
    print(f"SER for L=2: {ser}")
    
def question_9(ray1, ray2, ray3, ray4, symb):
    snr = 10**(10/10)
    y1 = symb * ray1 + get_noise(1/snr, len(symb))
    y2 = symb * ray2 + get_noise(1/snr, len(symb))
    y3 = symb * ray3 + get_noise(1/snr, len(symb))
    y4 = symb * ray4 + get_noise(1/snr, len(symb))
    numerator = (
        np.conjugate(ray1) * y1 +
        np.conjugate(ray2) * y2 +
        np.conjugate(ray3) * y3 +
        np.conjugate(ray4) * y4
    )

    denominator = (
        np.abs(ray1)**2 +
        np.abs(ray2)**2 +
        np.abs(ray3)**2 +
        np.abs(ray4)**2
    )
    p = numerator / denominator

    detected = detector(p)
    ser = np.mean(symb != detected)
    print(f"SER for L=4: {ser}")

def question_10(ray1, ray2, ray3, ray4, symb, db):
    N = len(symb)
    simulation_ser1 = np.zeros(len(db))
    simulation_ser2 = np.zeros(len(db))
    simulation_ser4 = np.zeros(len(db))
    theoretical_ser1 = np.zeros(len(db))
    theoretical_ser2 = np.zeros(len(db))
    theoretical_ser4 = np.zeros(len(db))

    for idx, i in enumerate(db):
        snr = 10**(i/10)

        y1 = symb * ray1 + get_noise(1/snr, N)
        y2 = symb * ray2 + get_noise(1/snr, N)
        y3 = symb * ray3 + get_noise(1/snr, N)
        y4 = symb * ray4 + get_noise(1/snr, N)

        p1 = np.conjugate(ray1) * y1 / (np.abs(ray1)**2)
        numerator2 = (
            np.conjugate(ray1) * y1 +
            np.conjugate(ray2) * y2
        )

        denominator2 = (
            np.abs(ray1)**2 +
            np.abs(ray2)**2
        )
        numerator4 = (
            np.conjugate(ray1) * y1 +
            np.conjugate(ray2) * y2 +
            np.conjugate(ray3) * y3 +
            np.conjugate(ray4) * y4
        )

        denominator4 = (
            np.abs(ray1)**2 +
            np.abs(ray2)**2 +
            np.abs(ray3)**2 +
            np.abs(ray4)**2
        )
        p2 = numerator2 / denominator2
        p4 = numerator4 / denominator4

        detected1 = detector(p1)
        detected2 = detector(p2)
        detected4 = detector(p4)

        errors1 = np.count_nonzero(symb != detected1)
        errors2 = np.count_nonzero(symb != detected2)
        errors4 = np.count_nonzero(symb != detected4)

        simulation_ser1[idx] = errors1 / N if errors1 >= 10 else np.nan
        simulation_ser2[idx] = errors2 / N if errors2 >= 10 else np.nan
        simulation_ser4[idx] = errors4 / N if errors4 >= 10 else np.nan

        theoretical_ser1[idx] = mrcSER(snr, 1)
        theoretical_ser2[idx] = mrcSER(snr, 2)
        theoretical_ser4[idx] = mrcSER(snr, 4)

    plt.figure(figsize=(8,5))
    plt.semilogy(db, theoretical_ser1, '-', linewidth=2, label="Theoretical SER 1 Antenna")
    plt.semilogy(db, theoretical_ser2, '-', linewidth=2, label="Theoretical SER 2 Antennas")
    plt.semilogy(db, theoretical_ser4, '-', linewidth=2, label="Theoretical SER 4 Antennas")
    plt.semilogy(db, simulation_ser1, '*', markersize=8, label="Simulated SER 1 Antenna")
    plt.semilogy(db, simulation_ser2, '*', markersize=8, label="Simulated SER 2 Antennas")
    plt.semilogy(db, simulation_ser4, '*', markersize=8, label="Simulated SER 4 Antennas")

    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("SER vs SNR for 1, 2, 4-branch MRC")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
    plt.ylim(1e-8, 1)
    plt.legend()
    plt.show()

def question_11(ray1, ray2, ray3, ray4, symb, b, db):
    N = len(symb)
    simulation_ber1 = np.zeros(len(db))
    simulation_ber2 = np.zeros(len(db))
    simulation_ber4 = np.zeros(len(db))
    theoretical_ber1 = np.zeros(len(db))
    theoretical_ber2 = np.zeros(len(db))
    theoretical_ber4 = np.zeros(len(db))

    for idx, i in enumerate(db):
        snr = 10**(i/10)

        y1 = symb * ray1 + get_noise(1/snr, N)
        y2 = symb * ray2 + get_noise(1/snr, N)
        y3 = symb * ray3 + get_noise(1/snr, N)
        y4 = symb * ray4 + get_noise(1/snr, N)

        p1 = np.conjugate(ray1) * y1 / (np.abs(ray1)**2)
        numerator2 = (
            np.conjugate(ray1) * y1 +
            np.conjugate(ray2) * y2
        )

        denominator2 = (
            np.abs(ray1)**2 +
            np.abs(ray2)**2
        )
        numerator4 = (
            np.conjugate(ray1) * y1 +
            np.conjugate(ray2) * y2 +
            np.conjugate(ray3) * y3 +
            np.conjugate(ray4) * y4
        )

        denominator4 = (
            np.abs(ray1)**2 +
            np.abs(ray2)**2 +
            np.abs(ray3)**2 +
            np.abs(ray4)**2
        )
        p2 = numerator2 / denominator2
        p4 = numerator4 / denominator4

        detected1 = detector(p1)
        detected2 = detector(p2)
        detected4 = detector(p4)

        r1 = get_received_bits(detected1)
        r2 = get_received_bits(detected2)
        r4 = get_received_bits(detected4)

        errors1 = np.count_nonzero(b != r1)
        errors2 = np.count_nonzero(b != r2)
        errors4 = np.count_nonzero(b != r4)

        simulation_ber1[idx] = errors1 / len(b) if errors1 >= 10 else np.nan
        simulation_ber2[idx] = errors2 / len(b) if errors2 >= 10 else np.nan
        simulation_ber4[idx] = errors4 / len(b) if errors4 >= 10 else np.nan

        theoretical_ber1[idx] = mrcBER(snr/2, 1)
        theoretical_ber2[idx] = mrcBER(snr/2, 2)
        theoretical_ber4[idx] = mrcBER(snr/2, 4)

    plt.figure(figsize=(8,5))
    plt.semilogy(db, theoretical_ber1, '-', linewidth=2, label="Theoretical BER 1 Antenna")
    plt.semilogy(db, theoretical_ber2, '-', linewidth=2, label="Theoretical BER 2 Antennas")
    plt.semilogy(db, theoretical_ber4, '-', linewidth=2, label="Theoretical BER 4 Antennas")
    plt.semilogy(db, simulation_ber1, '*', markersize=8, label="Simulated BER 1 Antenna")
    plt.semilogy(db, simulation_ber2, '*', markersize=8, label="Simulated BER 2 Antennas")
    plt.semilogy(db, simulation_ber4, '*', markersize=8, label="Simulated BER 4 Antennas")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR for 1, 2, 4-branch MRC")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
    plt.ylim(1e-8, 1)
    plt.legend()
    plt.show()

n = 2**20
snr_db = np.arange(5, 51, 5)
bits = np.random.randint(0, 2, size=n)
symbols = get_symbols(bits)
rayleigh1 = get_rayleigh(len(symbols))
rayleigh2 = get_rayleigh(len(symbols))
rayleigh3 = get_rayleigh(len(symbols))
rayleigh4 = get_rayleigh(len(symbols))

# question_8(rayleigh1, rayleigh2, symbols)
# question_9(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols)
# question_10(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols, snr_db)
# question_11(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols, bits, snr_db)