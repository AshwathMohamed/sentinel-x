import torch
import numpy as np
import hashlib

class MultiKrumServerAggregator:
    def __init__(self, n_clients, f_byzantine, k_select):
        self.n = n_clients
        self.f = f_byzantine
        self.k = k_select
        self.device = torch.device("cpu") 

    def verify_pqc_signature(self, signature, payload_bytes, public_key):
        if not signature or not public_key:
            return False
            
        h = hashlib.sha3_512()
        h.update(public_key + payload_bytes)
        expected_signature = h.digest()
        
        return signature == expected_signature

    def dequantize(self, quantized_delta, scale_factor):
        return quantized_delta.astype(np.float32) * scale_factor

    def aggregate(self, client_payloads):
        valid_updates = []
        node_ids = []
        
        for payload in client_payloads:
            raw_bytes = payload['quantized_delta'].tobytes() + np.array(payload['scale_factor'], dtype=np.float32).tobytes()
            
            if not self.verify_pqc_signature(payload.get('pqc_signature', b''), raw_bytes, payload.get('public_key', b'')):
                continue
                
            restored_delta = self.dequantize(payload['quantized_delta'], payload['scale_factor'])
            valid_updates.append(torch.tensor(restored_delta, device=self.device))
            node_ids.append(payload.get('node_id', 'Unknown-Node'))
            
        if len(valid_updates) < self.k:
            return np.array([]), [], [], [0.0] * len(client_payloads)

        updates_tensor = torch.stack(valid_updates)
        num_updates = updates_tensor.size(0)
        
        dist_matrix = torch.cdist(updates_tensor, updates_tensor, p=2)
        
        krum_scores = []
        num_neighbors = max(1, num_updates - self.f - 2) 
        
        for i in range(num_updates):
            sorted_distances, _ = torch.sort(dist_matrix[i])
            score = torch.sum(sorted_distances[1 : num_neighbors + 1])
            krum_scores.append(score.item())
            
        scores_tensor = torch.tensor(krum_scores)
        
        _, top_k_indices = torch.topk(scores_tensor, self.k, largest=False)
        selected_deltas = updates_tensor[top_k_indices]
        
        accepted_nodes = [node_ids[i] for i in top_k_indices.tolist()]
        rejected_nodes = [n for n in node_ids if n not in accepted_nodes]
        
        global_update = torch.mean(selected_deltas, dim=0)
        
        return global_update.numpy(), accepted_nodes, rejected_nodes, krum_scores