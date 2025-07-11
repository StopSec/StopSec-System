#!/usr/bin/env python
# coding: utf-8
# StopSec System, July 2025
# PURPOSE: This is a real-time pseudonym transmitter for the StopSec System

import uhd
import numpy as np
import time
import datetime
import sys
import signal
import argparse
import importlib
import requests
from requests.auth import HTTPBasicAuth
import csv

# Reload the watermark generator module
import watermark_generator
importlib.reload(watermark_generator)
from watermark_generator import WatermarkGenerator

class Watermark:
    def adjust_parameters_by_sample_rate(self):
        base_rate = 5e6  # reference sampling rate
        scale = self.samp_rate / base_rate
        self.chip_samples = int(640 * scale)
        self.FFT = int(64 * scale)
        self.CP = int(16 * scale)
        self.OFDM_size = self.FFT + self.CP
        # print(f"[Adjusted TX Parameters] chip_samples: {self.chip_samples}, FFT: {self.FFT}, CP: {self.CP}, OFDM_size: {self.OFDM_size}")

    lo_adjust = 1.5e6  # LO offset adjustment
    master_clock = 200e6  # Master clock rate

    def __init__(self, addr="192.168.40.2", external_clock=False, chan=0,
                 center_freq=3385e6, gain=30, samp_rate=5e6, pseudonym_bits=26, start_epoch=None):
        # Initialize SDR and transmission parameters
        self.addr = addr
        self.external_clock = external_clock
        self.channel = chan
        self.center_freq = center_freq
        self.gain = gain
        self.samp_rate = samp_rate
        self.adjust_parameters_by_sample_rate()
        self.pseudo = pseudonym_bits
        self.usrp = None
        self.txstreamer = None
        self.start_epoch = start_epoch
        self.keep_running = True
        self.generator = WatermarkGenerator()
        # self.preamble = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])  # Fixed preamble
        self.preamble = np.array([0, 0, 1, 1, 1, 0, 1])
        # Define Hamming (31,26) code positions
        self.PARITY_POSITIONS = np.array([2**i for i in range(5)])
        self.ALL_POSITIONS = np.arange(1, 32)
        self.DATA_POSITIONS = np.array([p for p in self.ALL_POSITIONS if p not in self.PARITY_POSITIONS])
        self.PARITY_MASKS = np.array([(self.ALL_POSITIONS & p) > 0 for p in self.PARITY_POSITIONS])

    def encode_hamming_31_26(self, data_bits):
        # Encode data bits using Hamming (31,26)
        data_bits = np.asarray(data_bits, dtype=np.uint8)
        if data_bits.shape[0] != 26:
            raise ValueError("Input must be 26 bits.")
        codeword = np.zeros(31, dtype=np.uint8)
        codeword[self.DATA_POSITIONS - 1] = data_bits
        for i, pos in enumerate(self.PARITY_POSITIONS):
            mask = self.PARITY_MASKS[i]
            parity = np.bitwise_xor.reduce(codeword[mask])
            codeword[pos - 1] = parity
        return codeword

    def get_80211_transmission_gap(self, cw_min=659, difs=34e-6, slot_time=9e-6):
        backoff_slots = np.random.randint(0, cw_min + 1)  # Uniform random in [0, CWmin]
        backoff_time = backoff_slots * slot_time
        total_gap = difs + backoff_time
        return 10*total_gap

    
    def init_radio(self):
        # Initialize USRP device
        self.usrp = uhd.usrp.MultiUSRP(f"addr={self.addr}")
        if self.external_clock:
            self.usrp.set_time_source("external")
            self.usrp.set_clock_source("external")
        self.usrp.set_master_clock_rate(self.master_clock)
        self.usrp.set_tx_antenna("TX/RX", self.channel)

    def setup_streamers(self):
        # Setup TX streamer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.channel]
        self.txstreamer = self.usrp.get_tx_stream(st_args)

    def tune(self, freq, gain, rate, use_lo_offset=True):
        # Configure radio parameters
        self.currate = rate
        self.usrp.set_tx_rate(rate, self.channel)
        if use_lo_offset:
            lo_off = rate / 2 + self.lo_adjust
            tune_req = uhd.types.TuneRequest(freq, lo_off)
        else:
            tune_req = uhd.types.TuneRequest(freq)
        self.usrp.set_tx_freq(tune_req, self.channel)
        self.usrp.set_tx_gain(gain, self.channel)

    def Set_all_params(self):
        # Apply all configuration steps
        self.init_radio()
        self.setup_streamers()
        self.tune(self.center_freq, self.gain, self.samp_rate)

    def Generate_pseudonym(self, pseudonym_bits):
        # Generate random pseudonym
        return np.random.binomial(n=1, p=0.5, size=pseudonym_bits)

    def send_samples(self, samples):
        # Transmit samples over SDR
        meta = uhd.types.TXMetadata()
        meta.start_of_burst = True
        meta.end_of_burst = False

        max_samps = self.txstreamer.get_max_num_samps() 
        total = samples.size
        idx = 0

        total_requested = 0
        total_sent = 0
        drop_events = 0

        while idx < total:
            nsamps = min(total - idx, max_samps)
            buf = np.zeros((1, max_samps), dtype=np.complex64)
            buf[0, :nsamps] = samples[idx:idx + nsamps]

            if idx + nsamps >= total:
                meta.end_of_burst = True

            sent = self.txstreamer.send(buf, meta)

            total_requested += nsamps
            total_sent += sent

            if sent == 0:
                drop_events += 1
                print(f"[WARNING] {time.strftime('%Y-%m-%d %H:%M:%S')} - Drop detected.")
            elif sent < nsamps:
                print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - Partial send: {sent}/{nsamps}")

            idx += sent

        print(f"[TX SUMMARY] Sent {total_sent}/{total_requested} samples. Drop events: {drop_events}")

    def run(self):
        if self.start_epoch is not None:
            print(f"[INFO] Waiting until {self.start_epoch} to start transmission...")
            while time.time() < self.start_epoch:
                if not self.keep_running:
                    print("[INFO] Transmission aborted during wait.")
                    return
                time.sleep(0.01)
            print("[INFO] Starting transmission.")
        
        # Main loop to transmit pseudonyms and log detection intervals
        elapsed_times = []
        last_detection_time = None
        no_tx = 0
        while self.keep_running:
            pseudonym_packet = self.Generate_pseudonym(self.pseudo)
            pseudonym_mess = ''.join(map(str, pseudonym_packet))
            timestamp_generated = time.time()
            
            # Write pseudonym to local DB
            write_url = "http://127.0.0.1:8081/write/"
            data = {
                "string_value": pseudonym_mess,
                "timestamp": timestamp_generated
            }
            try:
                response = requests.post(write_url, json=data, timeout=3)
                if response.status_code == 200:
                    print('[INFO] Pseudonym written to local DB with timestamp')
                else:
                    print('[ERROR] Could not write to local DB:', response.status_code)
            except Exception as e:
                print('[ERROR] Exception writing to local DB:', e)
            
            # Check for matches in remote DB
            try:
                local_db = requests.get("http://127.0.0.1:8081/read_all/").json()
                remote_db = requests.get("http://192.168.1.1:8080/read_all/", auth=HTTPBasicAuth('secondary_usr', 'pwd')).json()
                local_entries = {entry['string_value']: entry['timestamp'] for entry in local_db.get('entries', [])}
                remote_entries = {entry['string_value'] for entry in remote_db.get('entries', [])}

                for string_value in list(remote_entries):
                    if string_value in local_entries:
                        current_detection_time = time.time()
                        if last_detection_time is not None:
                            elapsed = current_detection_time - last_detection_time
                            elapsed_times.append(elapsed)
                            print(f"[INFO] Time between detections = {elapsed:.2f} seconds")
                        last_detection_time = current_detection_time

                        # Delete matched pseudonym from local DB
                        try:
                            delete_url = f"http://127.0.0.1:8081/delete_by_string/{string_value}"
                            del_response = requests.delete(delete_url)
                            if del_response.status_code == 200:
                                print(f"[INFO] Deleted matched entry '{string_value}' from local DB.")
                            else:
                                print(f"[WARN] Could not delete entry '{string_value}': Status {del_response.status_code}")
                        except Exception as e:
                            print(f"[ERROR] Exception during deletion from local DB: {e}")

                        # Stop after 100 detections and save to file
                        if len(elapsed_times) >= 100:
                            print('Mean time between detections:', np.mean(elapsed_times))
                            with open("Elapsed_Time.csv", "w", newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(["Elapsed Time (s)"])
                                writer.writerows([[et] for et in elapsed_times])
                            print("[INFO] Logged 100 elapsed times to Elapsed_Time.csv. Stopping transmission.")
                            self.keep_running = False
                            return
            except Exception as e:
                print("[ERROR] Exception checking databases:", e)
                
            send_start_time = time.time()
            # Encode pseudonym and generate watermarked signal
            pseudonym_codeword = self.encode_hamming_31_26(pseudonym_packet)
            
            final_packet = np.concatenate((self.preamble, pseudonym_codeword))
           
            watermarked_signal = self.generator.generate_watermark(final_packet)
            
            # send_start_time = time.time()
            self.send_samples(watermarked_signal)
           
            # time.sleep(self.get_80211_transmission_gap())
           
            send_end_time = time.time()
            
            send_duration = send_end_time - send_start_time
            print(f"Time taken to generate and send samples: {send_duration:.6f} seconds")

def handle_interrupt(sig, frame):
    print("\n[INFO] Graceful shutdown triggered.")
    tx.keep_running = False

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watermark TX RealTime")
    parser.add_argument("-f", "--freq", type=float, default=3385e6, help="Center frequency in Hz (default: 3385e6)")
    parser.add_argument("-g", "--gain", type=float, default=30, help="Transmit gain in dB (default: 30)")
    parser.add_argument("-r", "--rate", type=float, default=5e6, help="Sample rate in samples/sec (default: 2e6)")
    parser.add_argument("--start_time", type=str, help="Start time in HH:MM format (24-hour)")
    args = parser.parse_args()

    start_epoch = None
    if args.start_time:
        now = datetime.datetime.now()
        hh, mm = map(int, args.start_time.split(":"))
        start_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if start_dt < now:
            start_dt += datetime.timedelta(days=1)
        start_epoch = start_dt.timestamp()
    
    tx = Watermark(center_freq=args.freq, gain=args.gain, samp_rate=args.rate, start_epoch=start_epoch)
    signal.signal(signal.SIGINT, handle_interrupt)
    tx.Set_all_params()
    tx.run()
