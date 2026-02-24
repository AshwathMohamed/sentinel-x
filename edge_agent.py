import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import os

# --- 1. PQC-Inspired Signature Simulator ---
class PostQuantumPlutonSimulator:
    def __init__(self):
        # Placeholder for future Pluton ML-DSA firmware support.
        # We simulate asymmetric keys using a hash-bound flow for this PoC.
        self.mock_private_key = os.urandom(32) 
        self.mock_public_key = hashlib.sha256(self.mock_private_key).digest()

    def sign_pqc(self, data_bytes):
        """Simulates ML-DSA signature using HMAC-like flow bound to the public key."""
        h = hashlib.sha3_512()
        h.update(self.mock_public_key + data_bytes)
        return h.digest()

# --- 2. The Edge Agent ---
class SentinelEdgeAgent:
    def __init__(self, node_id="Unknown-Node"):
        self.node_id = node_id
        # Simulation: A lightweight PyTorch Autoencoder representing the NPU payload
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.hsm = PostQuantumPlutonSimulator()
        self.ema_score = 0.0

    def train_on_normal_data(self, normal_data, epochs=15):
        """Pre-training phase to establish a reliable baseline."""
        data_tensor = torch.FloatTensor(normal_data)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            recon = self.model(data_tensor)
            loss = self.criterion(recon, data_tensor)
            loss.backward()
            self.optimizer.step()

    def detect_anomaly(self, input_data):
        """Calculates smoothed anomaly score based on reconstruction error."""
        data_tensor = torch.FloatTensor(input_data)
        with torch.no_grad():
            reconstruction = self.model(data_tensor).numpy()
        raw_score = np.mean(np.square(input_data - reconstruction))
        # Exponential Moving Average (EMA) to smooth out fluctuations
        self.ema_score = (0.3 * float(raw_score)) + (0.7 * self.ema_score)
        return self.ema_score, reconstruction

    def apply_differential_privacy(self, delta, clip_val=0.1, noise_std=0.01):
        """Enforces (epsilon, delta) Local Differential Privacy (LDP)."""
        clipped_delta = np.clip(delta, -clip_val, clip_val)
        noise = np.random.normal(0, noise_std, clipped_delta.shape)
        return (clipped_delta + noise).astype(np.float32)

    def quantize_to_int4(self, delta):
        """Compresses 32-bit floats into 4-bit integers (Quanta Optimization)."""
        max_val = np.max(np.abs(delta))
        scale = max_val / 7.0 if max_val > 0 else 1.0
        quantized_delta = np.clip(np.round(delta / scale), -8, 7).astype(np.int8)
        return quantized_delta, float(scale)

    def step(self, raw_telemetry, clip_val=0.1, noise_std=0.01):
        """Main execution loop for telemetry processing."""
        processed = np.array(raw_telemetry, dtype=np.float32).reshape(1, -1)
        score, recon = self.detect_anomaly(processed)
        
        # Calculate Update Delta (the signal sent to the server)
        raw_delta = (processed - recon).astype(np.float32)
        
        # Apply Privacy and Compression
        dp_delta = self.apply_differential_privacy(raw_delta, clip_val, noise_std)
        q_delta, scale = self.quantize_to_int4(dp_delta)
        
        # Post-Quantum Signature Simulation
        payload = q_delta.tobytes() + np.array(scale, dtype=np.float32).tobytes()
        pqc_signature = self.hsm.sign_pqc(payload)
        
        return {
            "node_id": self.node_id,
            "anomaly_score": score,
            "quantized_delta": q_delta,
            "scale_factor": scale,
            "pqc_signature": pqc_signature,
            "public_key": self.hsm.mock_public_key,
            "raw_size_bytes": raw_delta.nbytes,
            "quantized_size_bytes": q_delta.nbytes
        }