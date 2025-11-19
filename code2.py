import os
import csv
import threading
from collections import deque, defaultdict
from datetime import datetime
from pythonosc import dispatcher, osc_server
from scipy.signal import butter, filtfilt
import numpy as np
import time

# --- CONFIG ---
SAMPLE_RATE = 256           # Hz (change to your Muse's sample rate)
WINDOW_SECONDS = 1.0        # how many seconds to accumulate before computing band power
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLE_RATE)
EEG_CHANNELS = 4            # number of EEG channels your Muse provides (change if needed)

ALPHA_BAND = (8.0, 13.0)    # Hz
BETA_BAND  = (13.0, 30.0)   # Hz

CSV_PATH = "muse_stream_with_bands.csv"
# ---------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpower_from_signal(signal):
    # simple band power = mean(square(signal))
    return float(np.mean(np.square(signal)))

class MuseOSCLoggerWithBands:
    def __init__(self, ip="127.0.0.1", port=5000, csv_path=CSV_PATH):
        self.ip = ip
        self.port = port
        self.csv_path = csv_path

        # Buffers per EEG channel
        self.buffers = [deque(maxlen=WINDOW_SIZE) for _ in range(EEG_CHANNELS)]
        self.lock = threading.Lock()  # protect buffers when server thread writes

        # Precompute filters
        self.b_alpha, self.a_alpha = butter_bandpass(ALPHA_BAND[0], ALPHA_BAND[1], SAMPLE_RATE, order=4)
        self.b_beta,  self.a_beta  = butter_bandpass(BETA_BAND[0],  BETA_BAND[1],  SAMPLE_RATE, order=4)

        # Prepare CSV with header
        header = ["timestamp", "address"]
        # add raw channel columns
        for ch in range(EEG_CHANNELS):
            header.append(f"raw_ch{ch+1}")
        # add alpha/beta power columns (aggregated per-window)
        for ch in range(EEG_CHANNELS):
            header.append(f"alpha_power_ch{ch+1}")
        for ch in range(EEG_CHANNELS):
            header.append(f"beta_power_ch{ch+1}")

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # OSC dispatcher
        self.dispatcher = dispatcher.Dispatcher()
        # map only EEG; but you can map "*" to log everything raw too
        self.dispatcher.map("/muse/eeg", self.eeg_handler)
        # optional: map other addresses to generic printer
        self.dispatcher.map("*", self.generic_handler)

        # Start server thread
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()
        print(f"Listening for Muse OSC on {self.ip}:{self.port} — logging to {self.csv_path}")

    def start_server(self):
        server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        server.serve_forever()

    def generic_handler(self, address, *args):
        # This will also trigger for /muse/eeg unless a specific mapping exists; we keep it for other streams
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {address}: {args}")
        # You can write these other streams to CSV or separate file if you wish.

    def eeg_handler(self, address, *args):
        """
        Expected args: a sequence of EEG channel values (one sample per channel),
        e.g. (ch1_val, ch2_val, ch3_val, ch4_val)
        """
        timestamp = datetime.now().isoformat()
        vals = list(map(float, args))  # ensure float

        # If number of channels differs, pad/truncate
        if len(vals) < EEG_CHANNELS:
            vals += [0.0] * (EEG_CHANNELS - len(vals))
        elif len(vals) > EEG_CHANNELS:
            vals = vals[:EEG_CHANNELS]

        # Print raw sample
        print(f"[{timestamp}] {address}: {vals}")

        # Append raw sample to buffers
        with self.lock:
            for ch in range(EEG_CHANNELS):
                self.buffers[ch].append(vals[ch])

            # If buffers are filled up to WINDOW_SIZE, compute band power
            if len(self.buffers[0]) >= WINDOW_SIZE:
                # convert buffers to numpy arrays
                windowed = np.stack([np.array(self.buffers[ch]) for ch in range(EEG_CHANNELS)])
                alpha_powers = []
                beta_powers  = []

                # For each channel, filter the window and compute power
                for ch in range(EEG_CHANNELS):
                    sig = windowed[ch, :]

                    # apply zero-phase bandpass filtering
                    try:
                        alpha_filt = filtfilt(self.b_alpha, self.a_alpha, sig)
                        beta_filt  = filtfilt(self.b_beta,  self.a_beta,  sig)
                    except Exception as e:
                        # If filtfilt fails (e.g. edge cases), fallback to raw segment
                        print("Filter error:", e)
                        alpha_filt = sig
                        beta_filt = sig

                    alpha_power = bandpower_from_signal(alpha_filt)
                    beta_power  = bandpower_from_signal(beta_filt)

                    alpha_powers.append(alpha_power)
                    beta_powers.append(beta_power)

                # Write a CSV row: timestamp, address, raw_ch1..raw_chN, alpha_power_ch1.., beta_power_ch1..
                row = [timestamp, address] + vals + alpha_powers + beta_powers
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                # Optionally: print summary of band powers
                avg_alpha = float(np.mean(alpha_powers))
                avg_beta  = float(np.mean(beta_powers))
                print(f"  Window computed — avg alpha power: {avg_alpha:.6f}, avg beta power: {avg_beta:.6f}")

if __name__ == "__main__":
    logger = MuseOSCLoggerWithBands(port=5000)

    try:
        while True:
            time.sleep(0.2)  # sleep to keep main thread responsive/cheap CPU
    except KeyboardInterrupt:
        print("\nStopping logger...")