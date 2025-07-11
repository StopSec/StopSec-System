#!/usr/bin/env python
# coding: utf-8
# StopSec System, July 2025
# PURPOSE: This is a real-time pseudonym receiver for the StopSec System

import time
import uhd
import numpy as np
from numba import njit, prange
import argparse
from scipy.signal import correlate, correlation_lags, fftconvolve
from scipy.spatial.distance import hamming
from collections import deque
import scipy.io
import requests
from requests.auth import HTTPBasicAuth
import matplotlib.pyplot as plt


class RealTimePseudonymDetector:
    """
    Real-time receiver and decoder for watermark-based pseudonym transmission.
    Captures signal using USRP, synchronizes using HT-STF preamble, extracts
    pseudonyms, decodes them using Hamming(31,26), and pushes to remote DB.
    """

    def adjust_parameters_by_sample_rate(self):
        """Adjust OFDM and chip timing parameters based on sampling rate."""
        base_rate = 2e6  # Default reference rate
        scale = self.sample_rate / base_rate
        self.chip_samples = int(864 * scale)
        self.FFT = int(64 * scale)
        self.CP = int(16 * scale)
        self.OFDM_size = self.FFT + self.CP
        print(f"[Adjusted Parameters] chip_samples: {self.chip_samples}, FFT: {self.FFT}, CP: {self.CP}, OFDM_size: {self.OFDM_size}")

    def __init__(self, device_addr="addr=192.168.40.2", center_freq=3.385e9,
                 sample_rate=2e6, gain=30, cp=16, slide_step=1,
                 pseudonym_length=38, packet=12960,
                 OFDM_size=144, chip_samples=864, FFT=128, num_chips=15):
        """
        Initialize system parameters for USRP signal acquisition and decoding.
        """

        # RF and signal parameters
        self.device_addr = device_addr
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.adjust_parameters_by_sample_rate()
        self.gain = gain
        self.detection_times = []

        # Pseudonym decoding configuration
        self.pseudonym_length = pseudonym_length
        self.OFDM_size = OFDM_size
        self.packet_length = packet
        self.chip_samples = chip_samples
        self.FFT = FFT
        self.num_chips = num_chips
        self.CP = cp
        self.slide_step = slide_step
        self.pseudonym_samples = self.pseudonym_length * self.packet_length
        self.buffer_length = int(2 * self.pseudonym_samples)

        # Power tracking
        self.avg_signal_power = []
        self.avg_noise_power = []

        # Buffers and matching filters
        self.big_buffer = deque(maxlen=self.buffer_length)
        self.matching_filter0 = np.array([-1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1])
        self.matching_filter1 = -self.matching_filter0
        self.pseudonym_preamble = np.array([0, 0, 1, 1, 1, 0, 1])
        self.htstf_preamble = self.generate_htstf()

        # USRP stream handlers
        self.usrp = None
        self.rx_streamer = None

        # Hamming decoder constants
        self.PARITY_POSITIONS = np.array([2**i for i in range(5)])
        self.ALL_POSITIONS = np.arange(1, 32)
        self.DATA_POSITIONS = np.array([p for p in self.ALL_POSITIONS if p not in self.PARITY_POSITIONS])
        self.PARITY_MASKS = [
            np.where((self.ALL_POSITIONS & parity_pos) > 0)[0]
            for parity_pos in self.PARITY_POSITIONS
        ]

    def decode_hamming_31_26(self, received):
        """Decode Hamming(31,26) and correct a single-bit error if present."""
        received = np.asarray(received, dtype=np.uint8)
        if received.shape[0] != 31:
            raise ValueError("Input must be 31 bits.")

        syndrome = 0
        for i, pos in enumerate(self.PARITY_POSITIONS):
            mask = self.PARITY_MASKS[i]
            parity = np.bitwise_xor.reduce(received[mask])
            if parity:
                syndrome += pos

        corrected = received.copy()
        if syndrome == 0:
            print('Pseudonym correctly detected!')
            status = "correct"
        elif 1 <= syndrome <= 31:
            corrected[syndrome - 1] ^= 1
            print(f'One error corrected at bit {syndrome}')
            status = "corrected"
        else:
            print('Uncorrectable error')
            return None, "uncorrectable"

        data_bits = corrected[self.DATA_POSITIONS - 1]
        return data_bits, status

    def setup_usrp(self):
        """Initialize USRP device and stream settings."""
        self.usrp = uhd.usrp.MultiUSRP(self.device_addr)
        self.usrp.set_rx_rate(self.sample_rate)
        tune_request = uhd.types.TuneRequest(self.center_freq, 1.5e6)
        self.usrp.set_rx_freq(tune_request)
        self.usrp.set_rx_gain(self.gain)
        self.usrp.set_rx_antenna("TX/RX")
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.rx_streamer = self.usrp.get_rx_stream(stream_args)
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(cmd)

    def stop_usrp(self):
        """Stop USRP streaming."""
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(cmd)

    def generate_htstf(self):
        """Load HT-STF reference sequence from file."""
        mat = scipy.io.loadmat('HTSTF.mat')
        return np.tile(mat['stf'].flatten(), 8)

    def cross_correlation_max(self, signal, pattern):
        """Perform correlation with preamble to detect synchronization point."""
        flipped_pattern = pattern.real[::-1]
        corr = fftconvolve(signal.real, flipped_pattern, mode='valid')
        lag = np.argmax(np.abs(corr))
        return lag, corr

    def average_subcarrier_power_48(self, signal):
        """Estimate power on the 48th subcarrier (used for watermark energy)."""
        try:
            reshaped = signal.reshape(-1, self.OFDM_size)
        except ValueError:
            remainder = len(signal) % self.OFDM_size
            padding = self.OFDM_size - remainder
            signal = np.pad(signal, (0, padding), mode='constant')
            reshaped = signal.reshape(-1, self.OFDM_size)

        data = reshaped[:, self.CP:]
        fft_data = np.fft.fftshift(np.fft.fft(data, n=self.FFT, axis=1), axes=1)
        power_96 = np.abs(fft_data[:, 64]) ** 2
        return np.mean(power_96)

    def matched_filter_detection(self, signal):
        """Perform chip-level detection using matched filters."""
        total_bits = self.pseudonym_length
        chip_len = self.chip_samples
        required_len = total_bits * self.packet_length

        if len(signal) < required_len:
            repeats = -(-required_len // len(signal))
            signal = np.tile(signal, repeats)[:required_len]

        p_bits = np.zeros(total_bits, dtype=np.uint8)
        for bit_idx in range(total_bits):
            bit_start = bit_idx * self.packet_length
            bit_end = bit_start + self.num_chips * chip_len
            bit_samples = signal[bit_start:bit_end]
            if len(bit_samples) != self.num_chips * chip_len:
                repeats = -(-self.num_chips * chip_len // len(bit_samples))
                bit_samples = np.tile(bit_samples, repeats)[:self.num_chips * chip_len]

            chips = bit_samples.reshape((self.num_chips, chip_len))
            chip_power = np.array([self.average_subcarrier_power_48(chip) for chip in chips])
            dot0 = np.dot(chip_power, self.matching_filter0)
            dot1 = np.dot(chip_power, self.matching_filter1)
            p_bits[bit_idx] = 1 if dot1 > dot0 else 0

        return p_bits

    def refill_big_buffer(self):
        """Fill the sample buffer to full capacity."""
        recv_buffer = np.zeros((1, self.rx_streamer.get_max_num_samps() - 4), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        while len(self.big_buffer) < self.buffer_length:
            num_rx = self.rx_streamer.recv(recv_buffer, metadata)
            if num_rx == 0 or metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
            self.big_buffer.extend(recv_buffer.flatten()[:num_rx])

    def read_samples(self, num_needed):
        """Read and append new samples from USRP to buffer."""
        recv_buffer = np.zeros((1, self.rx_streamer.get_max_num_samps() + 3596), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        while num_needed > 0:
            num_rx = self.rx_streamer.recv(recv_buffer, metadata)
            if num_rx == 0 or metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
            flat = recv_buffer.flatten()[:num_rx]
            self.big_buffer.extend(flat)
            num_needed -= len(flat)

    def detect_pseudonym(self):
        """Main detection loop: synchronize, decode, validate and post pseudonym."""
        self.recent_ratios = []
        threshold_factor = 4.5
        corr_window_len = self.OFDM_size

        self.setup_usrp()
        self.refill_big_buffer()

        # Measure noise floor
        for i in range(100):
            self.avg_noise_power.append(self.average_subcarrier_power_48(np.array(self.big_buffer, dtype=np.complex64)))
            self.refill_big_buffer()
        rx_noise = np.mean(self.avg_noise_power)

        start_time = time.time()
        try:
            while True:
                samples = np.array(self.big_buffer, dtype=np.complex64)
                lag, corr = self.cross_correlation_max(samples[:-self.pseudonym_samples + len(self.htstf_preamble)], self.htstf_preamble)
                start = lag + len(self.htstf_preamble)

                # Correlation thresholding
                corr_peak = np.max(np.abs(corr[lag:lag + corr_window_len]))
                corr_median = np.median(np.abs(corr))
                ratio = min(corr_peak / (corr_median + 1e-6), 10.0)
                self.recent_ratios.append(ratio)
                if len(self.recent_ratios) > 10:
                    self.recent_ratios.pop(0)

                if corr_peak < threshold_factor * corr_median:
                    self.read_samples(self.pseudonym_samples)
                    continue

                # Fallback strategy if HT-STF is misaligned
                if start + self.pseudonym_samples > len(samples):
                    search_range = corr[:lag - self.pseudonym_samples]
                    if len(search_range) > 0:
                        fallback_local_idx = np.argmax(search_range)
                        fallback_value = search_range[fallback_local_idx]
                        fallback_global_idx = fallback_local_idx + corr_window_len

                        if fallback_value > threshold_factor * corr_median:
                            lag = fallback_global_idx
                            start = lag + len(self.htstf_preamble)
                        else:
                            self.read_samples(self.pseudonym_samples)
                            continue
                    else:
                        self.read_samples(self.pseudonym_samples)
                        continue

                # Extract and decode pseudonym
                rx_signal = samples[start:start + self.pseudonym_samples]
                rx_pseudonym = self.matched_filter_detection(rx_signal)

                if hamming(rx_pseudonym[:len(self.pseudonym_preamble)], self.pseudonym_preamble) == 0:
                    pseudonym, status = self.decode_hamming_31_26(rx_pseudonym[len(self.pseudonym_preamble):])
                    if status in {"correct", "corrected"}:
                        signal_estimate = self.average_subcarrier_power_48(rx_signal) - rx_noise
                        self.avg_signal_power.append(max(signal_estimate, self.avg_signal_power[-1] if self.avg_signal_power else 0))
                        if status == "corrected":
                            print('ONE BIT ERROR CORRECTED!')

                        # Dynamic threshold update
                        if len(self.recent_ratios) >= 2:
                            threshold_factor = max(8.5, min(threshold_factor, 12.5))

                        print("✅ Pseudonym Detected!")
                        end_time = time.time()
                        self.detection_times.append(end_time)
                        start_time = time.time()

                        # Post to DB
                        url = "http://192.168.1.1:8080/write/"
                        data = {
                            "string_value": ''.join(map(str, pseudonym)),
                            "timestamp": time.time()
                        }
                        try:
                            response = requests.post(url, json=data, auth=HTTPBasicAuth('primary_usr', 'pwd'), timeout=3)
                            if response.status_code == 200:
                                print('Pseudonym written to remote database')
                            else:
                                print('⚠️ Error in database post')
                        except Exception as e:
                            print("DB Post Error:", e)
                    else:
                        print('Pseudonym detected in error!')
                else:
                    print("🚫 No valid pseudonym found.")
                self.read_samples(self.pseudonym_samples)

        except KeyboardInterrupt:
            print("Detection stopped.")

        finally:
            self.stop_usrp()
            if self.avg_signal_power:
                snr = 10 * np.log10(0.5 * np.mean(self.avg_signal_power) / rx_noise)
                print(f"SNR Estimate (48th subcarrier): {snr:.2f} dB")
            else:
                print('SNR Undefined!')

            if len(self.detection_times) > 1:
                intervals = np.diff(self.detection_times)
                avg_interval = np.mean(intervals[1:])
                print(f"Average detection interval: {avg_interval:.2f} sec")
            else:
                print("Not enough detections to compute interval.")


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Watermark RX RealTime")
    parser.add_argument("-f", "--freq", type=float, default=3385e6, help="Center frequency in Hz (default: 3385e6)")
    parser.add_argument("-g", "--gain", type=float, default=30, help="Transmit gain in dB (default: 30)")
    parser.add_argument("-r", "--rate", type=float, default=2e6, help="Sample rate in samples/sec (default: 2e6)")
    args = parser.parse_args()

    # Instantiate and run detector
    detector = RealTimePseudonymDetector(center_freq=args.freq, sample_rate=args.rate, gain=args.gain)
    detector.detect_pseudonym()
