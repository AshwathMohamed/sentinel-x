Project Sentinel-X

AMD SLINGSHOT 2026 Submission

Sentinel-X is a decentralized, privacy-preserving threat detection framework using Asynchronous Federated Learning (AFL), architected for the AMD hardware ecosystem.

Live Visuals

View Live Architecture Infographic

Architecture Overview

Edge Inference (Target: AMD Ryzen AI NPU): Network monitoring is offloaded to the NPU via the Ryzen AI SDK to maintain host CPU performance.

Data Minimization (INT4 + DP): Local Differential Privacy is applied via clipping and Gaussian noise, followed by INT4 Quantization, reducing data egress by ~99%.

Hardware Security (Target: AMD Pluton): A PQC-Inspired signature flow (simulated via SHA3-512-based MAC) is used to preserve pipeline bandwidth for future NIST ML-DSA firmware updates.

Cloud Aggregation (Target: AMD Instinct MI300X): Uses the ROCm-accelerated Multi-Krum Byzantine-robust algorithm to identify and reject poisoned update deltas.

System Schematic

[Edge: Ryzen AI NPU] → [DP + INT4 Quant + PQC Sign] → [MI300X ROCm Multi-Krum] → Global Model Update


AMD Integration Benefits

Ryzen AI NPU: Targeted for <50ms local inference latency for real-time packet inspection.

Instinct MI300X: High-bandwidth HBM3 memory allows for millions-scale client aggregation.

Quantization: Quark-compatible INT4 mapping ensures optimization for AMD specialized compute units.

Quick Start

Install Requirements:

pip install -r requirements.txt


Launch Dashboard:

streamlit run dashboard.py


Run Simulation:
Click "Train on Normal Data" to establish a baseline, then "Trigger Global Aggregation Round."