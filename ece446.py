from pythonosc import dispatcher, osc_server
from datetime import datetime
import threading
import csv
import os

class MuseOSCLogger:
    def __init__(self, ip="127.0.0.1", port=5000, csv_path="muse_stream.csv"):
        self.ip = ip
        self.port = port
        self.csv_path = csv_path

        # If the file doesn't exist yet, create it with a header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "address", "values"])

        # Set up dispatcher
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("*", self.generic_handler)

        # Start server thread
        self.server_thread = threading.Thread(
            target=self.start_server, daemon=True
        )
        self.server_thread.start()

        print(f"🎧 Logging Muse OSC data to {self.csv_path}")
        print(f"📡 Listening on {self.ip}:{self.port} ...")

    # Start OSC server
    def start_server(self):
        server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self.dispatcher
        )
        server.serve_forever()

    # Handle all OSC messages
    def generic_handler(self, address: str, *args):
        timestamp = datetime.now().isoformat()

        # Print incoming data
        print(f"[{timestamp}] {address}: {args}")

        # Append to CSV instantly
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, address, list(args)])

if __name__ == "__main__":
    logger = MuseOSCLogger(port=5000)

    # Keep main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping logger...")