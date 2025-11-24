"""
Real-time continuous MP3 pitch shifter controlled by EEG using librosa (no external Rubber Band CLI required).
- Continuous playback.
- Smooth real-time pitch changes triggered by EEG spikes (blinks).
- Handles OSC EEG messages where each channel is a separate float argument.
- Includes debounce logic to prevent spamming rapid pitch shifts.
"""

import argparse
import threading
import numpy as np
import sounddevice as sd
import librosa
from pythonosc import dispatcher, osc_server
from datetime import datetime
import time

# ---------------------- Configuration ----------------------
BLOCKSIZE = 4096  # increased to reduce underflow
SR = 16000
NUM_SEMITONES = [-4, -2, 0, 2, 4]
LOOP = True
BLINK_THRESHOLD = 1100  # EEG value threshold to detect a blink
BLINK_COOLDOWN = 0.3   # seconds

# ---------------------- Playback engine (continuous + real-time pitch) ----------------------
class ContinuousPlayer:
    def __init__(self, audio, sr, semitone_list, blocksize=BLOCKSIZE):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        self.audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.sr = sr
        self.semitone_list = semitone_list
        self.blocksize = blocksize

        self.current_idx = semitone_list.index(0) if 0 in semitone_list else 0
        self.target_idx = self.current_idx
        self.pos = 0
        self.lock = threading.Lock()

        self.current_chunk = np.zeros(self.blocksize, dtype=np.float32)
        self.chunk_pos = 0

        self.stream = sd.OutputStream(channels=1, samplerate=self.sr, blocksize=self.blocksize,
                                      callback=self.callback, dtype='float32')

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def advance_index(self, delta=1):
        with self.lock:
            self.target_idx = (self.target_idx + delta) % len(self.semitone_list)
            print(f"{datetime.now().isoformat()} - target pitch -> {self.semitone_list[self.target_idx]} semitones")

    def callback(self, outdata, frames, time_info, status):
        if status:
            print("Stream status:", status)

        out = np.zeros((frames,), dtype=np.float32)
        filled = 0

        while filled < frames:
            if self.chunk_pos >= len(self.current_chunk):
                end_pos = self.pos + self.blocksize
                if end_pos > len(self.audio):
                    if LOOP:
                        chunk = np.concatenate((self.audio[self.pos:], self.audio[:end_pos - len(self.audio)]))
                        self.pos = end_pos - len(self.audio)
                    else:
                        chunk = self.audio[self.pos:]
                        chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                        self.pos = len(self.audio)
                else:
                    chunk = self.audio[self.pos:end_pos]
                    self.pos = end_pos

                with self.lock:
                    st = self.semitone_list[self.target_idx]

                # Use librosa.effects.pitch_shift instead of pyrubberband
                self.current_chunk = librosa.effects.pitch_shift(y=chunk, sr=self.sr, n_steps=st).astype(np.float32)
                self.chunk_pos = 0

            remaining = frames - filled
            available = len(self.current_chunk) - self.chunk_pos
            to_copy = min(remaining, available)

            out[filled:filled + to_copy] = self.current_chunk[self.chunk_pos:self.chunk_pos + to_copy]
            self.chunk_pos += to_copy
            filled += to_copy

        outdata[:, 0] = out

# ---------------------- OSC handling with debounce ----------------------
class OSCController:
    def __init__(self, player: ContinuousPlayer, ip, port):
        self.player = player
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/muse/eeg", self.eeg_handler)
        self.dispatcher.map("*", self.generic_handler)
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        print(f"OSC server listening on {ip}:{port}")
        self.last_blink_time = 0  # timestamp of last triggered blink

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def eeg_handler(self, addr, *args):
        now = time.time()
        if now - self.last_blink_time < BLINK_COOLDOWN:
            return
        if any(a > BLINK_THRESHOLD for a in args):
            self.player.advance_index(1)
            self.last_blink_time = now

    def generic_handler(self, addr, *args):
        return

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser(description='Real-time continuous pitch-shifter controlled by EEG.')
    parser.add_argument('--file', '-f', required=True, help='Path to MP3 file')
    parser.add_argument('--port', type=int, default=3001, help='OSC listen port')
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='OSC listen IP')
    args = parser.parse_args()

    print('Loading audio...')
    audio, sr = librosa.load(args.file, sr=SR, mono=False)
    print('audio shape:', audio.shape, 'sample rate:', sr)

    player = ContinuousPlayer(audio, SR, NUM_SEMITONES)
    osc = OSCController(player, ip=args.ip, port=args.port)
    osc.start()

    print('Starting audio...')
    player.start()

    print('Running. Send EEG data to /muse/eeg to trigger pitch shifts.')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Stopping...')
        player.stop()

if __name__ == '__main__':
    main()
