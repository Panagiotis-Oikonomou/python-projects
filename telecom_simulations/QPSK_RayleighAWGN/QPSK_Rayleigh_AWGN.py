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

def question_1_2(ray):
    amp = np.mean(np.abs(ray))
    power = np.mean(np.abs(ray)**2)
    print(f"Μέση τιμή διαλείψεων πλάτους Rayleigh: {round(amp, 3)}")
    print(f"Ισχύς των διαλείψεων Rayleigh: {round(power, 1)}")

def question_3(ray):
    r = np.abs(ray)
    plt.figure(figsize=(7,5))
    plt.hist(r, bins=100, density=True, label="p(x) - generated", alpha=0.6,
         color='blue', edgecolor='black', linewidth=0.5)

    x = np.linspace(0, np.max(r), 1000)
    pdf = 2 * x * np.exp(-x**2)

    plt.plot(x, pdf, 'orange', linewidth=2, label="p(x) - ideal")

    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.grid()
    plt.legend()

    plt.show()

def question_4(ray):
    c = np.abs(ray[:2000])
    c_db = 20 * np.log10(c)

    plt.figure(figsize=(10,4))
    plt.plot(c_db, linewidth=1)
    
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (dB)")
    
    plt.grid()
    plt.show()
    
def question_6(ray):
    p = np.abs(ray)**2

    plt.figure(figsize=(7,5))
    plt.hist(p, bins=50, density=True, alpha=0.6,
             color='blue', edgecolor='black', label="Simulation")

    x = np.linspace(0, np.max(p), 1000)
    pdf = np.exp(-x)

    plt.plot(x, pdf, 'r', linewidth=2, label="Theoretical")

    plt.xlabel("Power")
    plt.ylabel("PDF")
    plt.title("Exponential Distribution (Rayleigh Power)")
    plt.grid()
    plt.legend()

    plt.show()

def question_7(ray, symb):
    s_r = ray * symb
    snr = 10**(10/10)
    noise = get_noise(1/snr, len(ray))
    s_r_n = s_r + noise

    phase = np.angle(ray)
    phase_revert = np.exp(-1j * phase)

    pre_detector = s_r_n * phase_revert
    detected = detector(pre_detector)

    symbol_errors = symb != detected
    num = np.sum(symbol_errors)
    print(f"Symbol errors {num}")

def question_8(ray, symb):
    s_r = ray * symb
    snr = 10**(20/10)
    noise = get_noise(1/snr, len(ray))
    s_r_n = s_r + noise

    phase = np.angle(ray)
    phase_revert = np.exp(-1j * phase)

    pre_detector = s_r_n * phase_revert
    detected = detector(pre_detector)

    symbol_errors = symb != detected
    num = np.sum(symbol_errors)
    ser = num / len(symb)
    print(f"Symbol error rate {ser}")

def question_9(ray, symb):
    s_r = ray * symb
    snr = 10**(30/10)
    noise = get_noise(1/snr, len(ray))
    s_r_n = s_r + noise

    phase = np.angle(ray)
    phase_revert = np.exp(-1j * phase)

    pre_detector = s_r_n * phase_revert
    pre_detector = pre_detector[:200]

    plt.figure(figsize=(6,6))
    plt.scatter(pre_detector.real, pre_detector.imag, s=5, alpha=0.5)

    # Ideal points (QPSK)
    val = 1/np.sqrt(2)
    ideal = np.array([
        complex(+val, +val),
        complex(-val, +val),
        complex(+val, -val),
        complex(-val, -val)
    ])

    plt.scatter(ideal.real, ideal.imag, color='red', s=100, marker='x')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.title("Constellation Diagram (SNR = 30 dB)")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.axis('equal')

    plt.show()

def question_10(ray, symb, snr_db):
    s_r = ray * symb

    phase = np.angle(ray)
    phase_revert = np.exp(-1j * phase)

    theoretical_ser = np.zeros(len(snr_db))
    simulation_ser = np.zeros(len(snr_db))

    for idx, i in enumerate(snr_db):
        snr = 10**(i/10)
        noise = get_noise(1/snr, len(ray))
        s_r_n = s_r + noise

        pre_detector = s_r_n * phase_revert
        detected = detector(pre_detector)

        symbol_errors = symb != detected
        simulation_ser[idx] = np.mean(symbol_errors)
        theoretical_ser[idx] = 0.5 * (1 - np.sqrt(snr/(1 + snr)))

    simulation_ser = np.maximum(simulation_ser, 1e-6)
    theoretical_ser = np.maximum(theoretical_ser, 1e-6)

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, theoretical_ser, '-', label="Theoretical SER")
    plt.semilogy(snr_db, simulation_ser, '*', label="Simulated SER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("SER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def question_11(ray, symb, b, snr_db):
    s_r = ray * symb

    phase = np.angle(ray)
    phase_revert = np.exp(-1j * phase)

    theoretical_ber = np.zeros(len(snr_db))
    simulation_ber = np.zeros(len(snr_db))

    for idx, i in enumerate(snr_db):
        snr = 10**(i/10)
        noise = get_noise(1/snr, len(ray))
        s_r_n = s_r + noise

        pre_detector = s_r_n * phase_revert
        detected = detector(pre_detector)

        received_bits = get_received_bits(detected)
        bit_errors = b != received_bits
        simulation_ber[idx] = np.mean(bit_errors)
        theoretical_ber[idx] = (0.5 * (1 - np.sqrt(snr/(1 + snr))))/2

    simulation_ber = np.maximum(simulation_ber, 1e-6)
    theoretical_ber = np.maximum(theoretical_ber, 1e-6)

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, theoretical_ber, '-', label="Theoretical BER")
    plt.semilogy(snr_db, simulation_ber, '*', label="Simulated BER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("BER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

n = 2**22
snr_db = np.arange(-5, 31, 5)
bits = np.random.randint(0, 2, size=n)
symbols = get_symbols(bits)
rayleigh = get_rayleigh(len(symbols))

# question_1_2(rayleigh)
# question_3(rayleigh)
# question_4(rayleigh)
# question_6(rayleigh)
# question_7(rayleigh, symbols)
# question_8(rayleigh, symbols)
# question_9(rayleigh, symbols)
# question_10(rayleigh, symbols, snr_db)
# question_11(rayleigh, symbols, bits, snr_db)