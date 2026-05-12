import matplotlib.pyplot as plt
import math
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

def scSER(snr, L): #snr: SNR in linear scale, L: Number of antennas
    ser = 0.0
    for l in range(L):  # 0 to L-1
        denom = l + 1
        mu = math.sqrt((snr / denom) / (2 + (snr / denom)))
        ser += ((-1)**l) * math.comb(L - 1, l) / denom * ((1 - mu) / 2)
    return ser * 2 * L

def question_2(ray1, ray2, symb):
    snr = 10**(15/10)
    y1 = symb * ray1 + get_noise(1/snr, len(symb))
    y2 = symb * ray2 + get_noise(1/snr, len(symb)) 
    max = np.abs(ray1) >= np.abs(ray2)
    selected = np.where(max, y1/ray1, y2/ray2)

    detected = detector(selected)

    ser = np.mean(symb != detected)
    print(f"SER for L=2: {ser}")
    
def question_3(ray1, ray2, ray3, ray4, symb):
    snr = 10**(10/10)
    y1 = symb * ray1 + get_noise(1/snr, len(symb))
    y2 = symb * ray2 + get_noise(1/snr, len(symb))
    y3 = symb * ray3 + get_noise(1/snr, len(symb))
    y4 = symb * ray4 + get_noise(1/snr, len(symb))
    
    m1 = np.abs(ray1)
    m2 = np.abs(ray2)
    m3 = np.abs(ray3)
    m4 = np.abs(ray4)

    max12 = m1 >= m2
    sel12 = np.where(max12, y1/ray1, y2/ray2)

    max34 = m3 >= m4
    sel34 = np.where(max34, y3/ray3, y4/ray4)

    max_final = np.abs(ray1 * max12 + ray2 * (~max12)) >= np.abs(ray3 * max34 + ray4 * (~max34))

    selected = np.where(max_final, sel12, sel34)

    detected = detector(selected)

    ser = np.mean(symb != detected)
    print(f"SER for L=4: {ser}")

def question_4(ray1, ray2, ray3, ray4, symb, db):
    theoretical_ser = np.zeros(len(db))
    simulation_ser = np.zeros(len(db))
    L = 4

    for idx, i in enumerate(db):
        snr = 10**(i/10)
        y1 = symb * ray1 + get_noise(1/snr, len(ray1))
        y2 = symb * ray2 + get_noise(1/snr, len(ray2))
        y3 = symb * ray3 + get_noise(1/snr, len(ray3))
        y4 = symb * ray4 + get_noise(1/snr, len(ray4))

        m1 = np.abs(ray1)
        m2 = np.abs(ray2)
        m3 = np.abs(ray3)
        m4 = np.abs(ray4)

        max12 = m1 >= m2
        sel12 = np.where(max12, y1/ray1, y2/ray2)

        max34 = m3 >= m4
        sel34 = np.where(max34, y3/ray3, y4/ray4)

        max_final = np.abs(ray1 * max12 + ray2 * (~max12)) >= np.abs(ray3 * max34 + ray4 * (~max34))

        selected = np.where(max_final, sel12, sel34)
        
        detected = detector(selected)

        simulation_ser[idx] = np.mean(symb != detected)
        theoretical_ser[idx] = scSER(snr, L)

    simulation_ser = np.maximum(simulation_ser, 1e-10)
    theoretical_ser = np.maximum(theoretical_ser, 1e-10)

    plt.figure(figsize=(8,5))
    plt.semilogy(db, theoretical_ser, '-', linewidth=2, label="Theoretical SER")
    plt.semilogy(db, simulation_ser, '*', markersize=8, label="Simulated SER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title(f"SER vs SNR for {L}-branch SC")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def question_5(ray1, ray2, ray3, ray4, symb, b, db):
    theoretical_ser = np.zeros(len(db))
    simulation_ser = np.zeros(len(db))
    L = 4

    for idx, i in enumerate(db):
        snr = 10**(i/10)
        y1 = symb * ray1 + get_noise(1/snr, len(ray1))
        y2 = symb * ray2 + get_noise(1/snr, len(ray2))
        y3 = symb * ray3 + get_noise(1/snr, len(ray3))
        y4 = symb * ray4 + get_noise(1/snr, len(ray4))

        m1 = np.abs(ray1)
        m2 = np.abs(ray2)
        m3 = np.abs(ray3)
        m4 = np.abs(ray4)

        max12 = m1 >= m2
        sel12 = np.where(max12, y1/ray1, y2/ray2)

        max34 = m3 >= m4
        sel34 = np.where(max34, y3/ray3, y4/ray4)

        max_final = np.abs(ray1 * max12 + ray2 * (~max12)) >= np.abs(ray3 * max34 + ray4 * (~max34))

        selected = np.where(max_final, sel12, sel34)
        
        detected = detector(selected)
        r_b = get_received_bits(detected)

        simulation_ser[idx] = np.mean(b != r_b)
        theoretical_ser[idx] = scSER(snr, L) / 2

    simulation_ser = np.maximum(simulation_ser, 1e-10)
    theoretical_ser = np.maximum(theoretical_ser, 1e-10)

    plt.figure(figsize=(8,5))
    plt.semilogy(db, theoretical_ser, '-', linewidth=2, label="Theoretical BER")
    plt.semilogy(db, simulation_ser, '*', markersize=8, label="Simulated BER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs SNR for {L}-branch SC")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

n = 2**20
snr_db = np.arange(-5, 51, 5)
bits = np.random.randint(0, 2, size=n)
symbols = get_symbols(bits)
rayleigh1 = get_rayleigh(len(symbols))
rayleigh2 = get_rayleigh(len(symbols))
rayleigh3 = get_rayleigh(len(symbols))
rayleigh4 = get_rayleigh(len(symbols))

question_2(rayleigh1, rayleigh2, symbols)
question_3(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols)
question_4(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols, snr_db)
question_5(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols, bits, snr_db)