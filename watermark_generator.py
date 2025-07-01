#!/usr/bin/env python
# coding: utf-8
# StopSec System, July 2025
# PURPOSE: To generate code modulation (CM) watermarking signal on the pseudonym subcarrier.
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal as signal

class WatermarkGenerator:
    def __init__(self, text_message=None):
        
        # Basic OFDM and system parameters
        self.FFT = 128  # Number of FFT points
        self.OFDM_size = 144  # Total size including cyclic prefix
        self.data_size = 100  # Number of data subcarriers
        # Watermarking Code. This is an 15 length m-sequence
        self.pn_sequence = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        self.packet = 9000  # Length of packet
        self.chip_samp = self.packet // len(self.pn_sequence)  # Samples per chip
        self.CP = 16  # Cyclic prefix length
        self.mod_index = 1.0  # Modulation index for amplitude watermarking
        self.pilotValue = 1.4142 + 1.4142j  # Pilot symbol
        self.pseudonymValue = 1.4142 + 1.4142j # Pseudonym carrier value

        # Default text to encode if none is given
        self.text_message = text_message or (
            'The StopSec protocol was implemented and validated at the POWDER wireless testbed in 2025.'
        )

        # Subcarrier allocations
        self.allCarriers = np.arange(-self.FFT//2, self.FFT//2) 
        self.pilotCarriers = np.array([-36, -12, 12, 36])
        self.pseudonymCarrier = np.array([0])

        guardLower = np.arange(-64, -52)
        guardUpper = np.arange(52, 63)
        guardBands = np.concatenate((guardLower, guardUpper))
        self.reserved = np.concatenate((guardBands, self.pilotCarriers, self.pseudonymCarrier))
        self.dataCarriers = np.array([sc for sc in self.allCarriers if sc not in self.reserved])

    def text2bits(self, message):
        # Convert text message to list of bits (ASCII 8-bit binary per character)
        return [int(bit) for char in message for bit in format(ord(char), '08b')]

    def lut(self, data, inputVec, outputVec):
        # Lookup table for modulation mapping
        output = np.zeros(data.shape)
        eps = np.finfo('float').eps
        for i in range(len(inputVec)):
            for k in range(len(data)):
                if abs(data[k] - inputVec[i]) < eps:
                    output[k] = outputVec[i]
        return output

    def binary2mary(self, data, M):
        # Convert binary data to M-ary symbols
        log2M = round(math.log2(M))
        if len(data) % log2M != 0:
            raise ValueError("Input to binary2mary must be divisible by log2(M).")
        binvalues = 2 ** np.arange(log2M - 1, -1, -1)
        reshaped_data = np.reshape(data, (-1, log2M))
        return reshaped_data.dot(binvalues)

    def generate_data_frame(self):
        # Generate modulated data symbols using QPSK
        A = math.sqrt(9/2)  # Amplitude scaling
        data_sequence = self.text2bits(self.text_message)  # Convert message to bits
        data_bits = np.tile(data_sequence, 25)  # Repeat bits to fill frame
        data = self.binary2mary(data_bits, 4)  # Convert to 4-ary symbols (QPSK)

        # QPSK mapping
        inputVec = [0, 1, 2, 3]
        outputVecI = [A, -A, A, -A]
        outputVecQ = [A, A, -A, -A]

        # Map symbols to complex IQ samples
        xI = self.lut(data, inputVec, outputVecI).reshape((1, len(data)))
        xQ = self.lut(data, inputVec, outputVecQ).reshape((1, len(data)))
        qpsk_IQ = (xI.flatten() + 1j * xQ.flatten()).astype(np.complex64)
        return qpsk_IQ

    def _generate_ofdm(self, data, amplitude_mode='high'):
        # Generate OFDM symbols from data
        result = []
        for i in range(len(data) // self.data_size):
            payload = data[i * self.data_size: (i + 1) * self.data_size]
            symbol = np.zeros(self.FFT, dtype=complex)

            # Insert pilot and pseudonym symbols
            symbol[self.pilotCarriers] = self.pilotValue
            mod_factor = 1 + self.mod_index if amplitude_mode == 'high' else 1 - self.mod_index
            symbol[self.pseudonymCarrier] = self.pseudonymValue * mod_factor

            # Insert data symbols
            symbol[self.dataCarriers] = payload

            # IFFT and add cyclic prefix
            ofdm_time = np.fft.ifft(symbol,n=self.FFT) #np.fft.ifft(np.fft.ifftshift(symbol), n=self.FFT)
            cp = ofdm_time[-self.CP:]
            result.extend(np.hstack([cp, ofdm_time]))

        return np.array(result)

    def watermark_bit(self, data, bit_value):
        # Encode watermark using chip-level amplitude modulation
        watermark_signal = None  
    
        for i, bit in enumerate(self.pn_sequence):
            chip_data = data[i * self.chip_samp:(i + 1) * self.chip_samp]
            segment = self._generate_ofdm(chip_data, amplitude_mode='high' if bit == bit_value else 'low')
            if watermark_signal is None:
                watermark_signal = segment
            else:
                watermark_signal = np.concatenate([watermark_signal, segment])
    
        return watermark_signal


    def generate_htstf(self):
        # Load and repeat training sequence from HTSTF.mat file
        mat = scipy.io.loadmat('HTSTF.mat')
        stf = mat['stf'].flatten()
        return np.tile(stf, 8)

    def generate_watermark(self, pseudonym_packet):
        # Combine all steps to create a complete watermark signal
        signal = self.generate_data_frame()
        preamble = self.generate_htstf()
        wm_one = self.watermark_bit(signal, bit_value=1)
        wm_zero = self.watermark_bit(signal, bit_value=0)
    
        watermark = None
    
        # Generate watermark from pseudonym bit sequence
        for bit in pseudonym_packet:
            segment = wm_one if bit == 1 else wm_zero
    
            if watermark is None:
                watermark = segment
            else:
                watermark = np.concatenate([watermark, segment])
    
        return np.concatenate([preamble, 2.5*watermark])

