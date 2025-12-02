#!/usr/bin/env python3
"""
cassia_scheduler.py

Real Docker-backed scheduler for CASSIA:
- Starts function containers on cold starts
- Reuses warm containers per node
- Measures real latency via HTTP /invoke
- Runs light SSL pretrain, policy head, and REINFORCE fine-tuning

Prerequisites (run on host with Docker or run scheduler as container with --network cassia-net):
- Docker images built: cassia-login, cassia-upload, cassia-analytics, cassia-batch
- Docker network created: docker network create cassia-net
- Python packages: install via requirements.txt (numpy, torch, scikit-learn, matplotlib, docker, requests)
"""

import os
import time
import random
import math
from uuid import uuid4
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import docker
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------- CONFIG --------------------
NUM_NODES = 5
FUNCTION_IMAGES = ["cassia-login", "cassia-upload", "cassia-analytics", "cassia-batch"]
FUNCTION_PORT = 8000
DOCKER_NETWORK = "cassia-net"

WARM_POOL_LIMIT = 20  # per node
TRAIN_SAMPLES = 1000  # reduce for real container runs
TEST_SAMPLES  = 400
REINFORCE_EPOCHS = 3  # keep small initially (real containers are slow)
BATCH_SIZE = 64

LOAD_PENALTY = 0.10  # optional additional synthetic load factor

# Docker client
client = docker.from_env()

# warm_pool: list of dicts per node
# keying strategy: use (func_type, user_id) or just func_type to reduce #containers.
# Here we key by func_type to keep warm pool manageable for demo.
warm_pool = [OrderedDict() for _ in range(NUM_NODES)]

# -------------------- HELPERS: Workload & Invocations --------------------

def generate_bursty_workload(num_nodes, steps, base_rate=30, burst_prob=0.25, skew_strength=0.7):
    workload = []
    skewed_probs = np.array([0.6] + [(1 - 0.6) / (num_nodes - 1)] * (num_nodes - 1))
    for t in range(steps):
        if random.random() < 0.4:
            node_loads = [0] * num_nodes
            for _ in range(num_nodes * base_rate):
                node_choice = np.random.choice(range(num_nodes), p=skewed_probs)
                node_loads[node_choice] += 1
        else:
            node_loads = [np.random.poisson(base_rate) for _ in range(num_nodes)]
        if random.random() < burst_prob:
            target_node = random.randint(0, num_nodes - 1)
            burst_load = int(base_rate * (1 + skew_strength * 10))
            node_loads[target_node] += burst_load
        if random.random() < 0.5:
            skew_node = random.randint(0, num_nodes - 1)
            node_loads[skew_node] += int(base_rate * skew_strength)
        if random.random() < 0.2:
            spike_node = random.randint(0, num_nodes - 1)
            node_loads[spike_node] += random.randint(200, 500)
        workload.append(node_loads)
    return workload

def generate_function_invocations(n=10000):
    types = ['login', 'upload', 'analytics', 'batch']
    user_ids = list(range(10))
    data = []
    for _ in range(n):
        function_type = random.choice(types)
        user_id = random.choice(user_ids)
        arrival_time = random.randint(0, 86400)
        payload = random.uniform(0.1, 5.0)
        cpu = random.uniform(0.1, 2.0)
        memory = random.uniform(128, 1024)
        cold_start = random.choice([0, 1])
        data.append([function_type, user_id, arrival_time, payload, cpu, memory, cold_start])
    return data

# -------------------- STEP 1: Generate / Preprocess Data --------------------

print("Preparing synthetic invocation dataset (smaller for Docker runs)...")
raw_data = generate_function_invocations(n=TRAIN_SAMPLES + TEST_SAMPLES)
encoder = OneHotEncoder(sparse_output=False)
types_encoded = encoder.fit_transform([[row[0]] for row in raw_data])

scaler = StandardScaler()
numeric = scaler.fit_transform([[row[1], row[2], row[3], row[4], row[5]] for row in raw_data])
y = [row[6] for row in raw_data]
X = np.hstack((types_encoded, numeric))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SAMPLES / (TRAIN_SAMPLES + TEST_SAMPLES), random_state=42)

# workloads: one step per sample
workload_train = generate_bursty_workload(NUM_NODES, steps=len(X_train), base_rate=30, burst_prob=0.25, skew_strength=0.7)
workload_test  = generate_bursty_workload(NUM_NODES, steps=len(X_test), base_rate=30, burst_prob=0.25, skew_strength=0.7)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------- STEP 2: Self-Supervised Context Encoder --------------------

class ContextDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        mask = np.random.randint(0, len(x))
        label = x[mask]
        x[mask] = 0.0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.predictor(h)

train_ds = ContextDataset(X_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContextEncoder(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Pretraining context encoder (brief)...")
for epoch in range(3):  # keep short for real runs
    total_loss = 0.0
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x).squeeze()
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Context pretrain Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------- STEP 3: Policy Head (inference-only) --------------------

class PolicyHead(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_nodes=NUM_NODES):
        super().__init__()
        self.encoder = encoder.encoder  # reuse encoder part only
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

scheduler = PolicyHead(model, num_nodes=NUM_NODES).to(device)

# -------------------- DOCKER-BASED INVOCATION UTILITIES --------------------

def safe_container_remove(container):
    try:
        container.stop(timeout=1)
    except Exception:
        pass
    try:
        container.remove()
    except Exception:
        pass

def wait_for_http(container_name, port=FUNCTION_PORT, timeout=8.0, interval=0.25):
    """Wait until container HTTP endpoint is responding (within same docker network by name)."""
    deadline = time.time() + timeout
    url = f"http://{container_name}:{port}/invoke"
    while time.time() < deadline:
        try:
            r = requests.post(url, timeout=1)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

def ensure_container_for_function(node, func_type_idx, key_by='func_type'):
    """
    Ensure a warm container exists on given node for func_type_idx.
    Keying currently uses func_type only (keeps warm containers small).
    Returns (container_name, cold_flag)
    """
    pool = warm_pool[node]
    key = int(func_type_idx)  # if key_by == 'func_id', use tuple (func_type_idx,user)
    if key in pool:
        # Move used key to end to implement LRU-ish behavior
        pool.move_to_end(key)
        return pool[key]["name"], 0

    # Cold start: start a new container with deterministic name within network
    safe_suffix = uuid4().hex[:6]
    container_name = f"node{node}-f{func_type_idx}-{safe_suffix}"

    image = FUNCTION_IMAGES[func_type_idx]
    try:
        container = client.containers.run(
            image,
            name=container_name,
            detach=True,
            network=DOCKER_NETWORK,
            # inside network we can reach container by name:port
            # avoid mapping host ports here to keep network-local addressing clean
        )
    except docker.errors.APIError as e:
        # If name conflict or other issue: try a random name
        container = client.containers.run(image, detach=True, network=DOCKER_NETWORK)

    # Wait until endpoint is ready (or timeout)
    ready = wait_for_http(container_name, port=FUNCTION_PORT, timeout=6.0)
    if not ready:
        # still accept it but warn
        print(f"[WARN] container {container_name} might not be healthy/ready.")

    pool[key] = {"name": container_name, "container": container, "started_at": time.time()}
    # eviction if needed (LRU)
    while len(pool) > WARM_POOL_LIMIT:
        old_key, old_val = pool.popitem(last=False)  # pop oldest
        try:
            safe_container_remove(old_val["container"])
        except Exception:
            pass

    return container_name, 1

def invoke_function_docker(node, func_type_idx, extra_load=0.0):
    """
    Ensure container and invoke /invoke endpoint. Returns (latency_ms, reward, cold_flag).
    """
    container_name, cold = ensure_container_for_function(node, func_type_idx)
    url = f"http://{container_name}:{FUNCTION_PORT}/invoke"
    t0 = time.monotonic()
    try:
        r = requests.post(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        # if request fails, give very large penalty and attempt cleanup of container
        print(f"[ERROR] Invocation failed for {container_name}: {e}")
        # try to remove the container to avoid stuck containers
        try:
            cont = client.containers.get(container_name)
            safe_container_remove(cont)
        except Exception:
            pass
        # big penalty
        latency_ms = 5000.0 + extra_load * 0.1
        reward = -latency_ms - (100.0 if cold else 0.0)
        return latency_ms, reward, cold

    t1 = time.monotonic()
    latency_ms = (t1 - t0) * 1000.0
    # optionally add synthetic extra load penalty
    latency_ms += extra_load * LOAD_PENALTY

    # cold penalty (we still count cold separately)
    if cold:
        latency_ms += 100.0

    reward = -latency_ms
    return latency_ms, reward, cold

# -------------------- STEP 4: Evaluation / CASSIA Inference (real containers) --------------------

def run_cassia_inference_on_test(scheduler_model, X_test_local, workload_pattern):
    cold_start_events = [0] * NUM_NODES
    response_times = [[] for _ in range(NUM_NODES)]
    total_requests = [0] * NUM_NODES

    scheduler_model.eval()
    with torch.no_grad():
        for i in range(len(X_test_local)):
            x_np = X_test_local[i].astype(np.float32)
            x = torch.tensor(x_np, dtype=torch.float32).to(device)
            func_type = int(np.argmax(x_np[:4]))
            # derive a simple func_id (func_type, user)
            user_id = int(scaler.inverse_transform([x_np[4:]])[0][0])
            func_id = (func_type, user_id)

            scores = scheduler_model(x)
            probs = F.softmax(scores.cpu(), dim=-1).squeeze()
            node = int(torch.multinomial(probs, num_samples=1).item())

            current_load = workload_pattern[i][node]
            latency, _, cold = invoke_function_docker(node, func_type, extra_load=current_load)
            if cold:
                cold_start_events[node] += 1
            response_times[node].append(latency)
            total_requests[node] += 1

            # safety small sleep to avoid Docker overload
            time.sleep(0.01)

    return response_times, cold_start_events, total_requests

# -------------------- STEP 5: Baselines (FIFO & Round-Robin) --------------------

def run_baseline(strategy="fifo", workload_pattern=None):
    num_nodes_local = NUM_NODES
    rr_counter = 0
    node_free_time = [0.0] * num_nodes_local
    baseline_cold = [0] * num_nodes_local
    baseline_latency = [[] for _ in range(num_nodes_local)]
    baseline_requests = [0] * num_nodes_local
    current_time = 0.0
    if workload_pattern is None:
        workload_pattern = [[0]*num_nodes_local for _ in range(len(X_test))]

    for i in range(len(X_test)):
        true_cold = y_test[i]
        base_latency = random.uniform(50, 150)
        if strategy == "fifo":
            node = min(range(num_nodes_local), key=lambda n: node_free_time[n])
        elif strategy == "round_robin":
            node = rr_counter % num_nodes_local
            rr_counter += 1
        else:
            node = 0
        current_load = workload_pattern[i][node]
        latency = base_latency + current_load * LOAD_PENALTY
        if true_cold:
            latency += 100
            baseline_cold[node] += 1
        finish_time = max(current_time, node_free_time[node]) + latency
        node_free_time[node] = finish_time
        baseline_latency[node].append(latency)
        baseline_requests[node] += 1
    all_latencies = [lat for lst in baseline_latency for lat in lst]
    avg_latency = np.mean(all_latencies) if all_latencies else 0
    total_cold = sum(baseline_cold)
    total_time = len(X_test) * 0.1
    throughput_per_node = [reqs / total_time for reqs in baseline_requests]
    return baseline_latency, baseline_cold, avg_latency, total_cold, throughput_per_node

# -------------------- STEP 6: REINFORCE TRAINING (fine-tune on real containers) --------------------

class ReinforceScheduler(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_nodes=NUM_NODES):
        super().__init__()
        self.encoder = encoder.encoder
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, x):
        h = self.encoder(x)
        logits = self.policy(h)
        probs = F.softmax(logits, dim=-1)
        return probs

reinforce_model = ReinforceScheduler(model).to(device)
reinforce_optimizer = optim.Adam(reinforce_model.parameters(), lr=1e-3)

def train_reinforce_real(model, X_data, epochs=REINFORCE_EPOCHS):
    lambda_balance = 0.05
    entropy_coef = 0.02

    for epoch in range(epochs):
        # Reset or keep warm pool between epochs as design choice. We'll keep across epoch to simulate warm carry-over.
        node_counts = [0] * NUM_NODES
        log_probs, rewards, entropies = [], [], []

        model.train()
        for i in range(len(X_data)):
            x_np = X_data[i].astype(np.float32)
            x = torch.tensor(x_np, dtype=torch.float32).to(device)
            func_type = int(np.argmax(x_np[:4]))
            user_id = int(scaler.inverse_transform([x_np[4:]])[0][0])
            func_id = (func_type, user_id)

            probs = model(x)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            node = int(action.item())
            node_counts[node] += 1

            # Real invocation — this is the slow part
            extra_load = workload_train[i][node] if i < len(workload_train) else 0
            latency, reward, cold = invoke_function_docker(node, func_type, extra_load=extra_load)
            # amplify reward differences
            reward *= 1.0

            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            entropies.append(entropy)

            # small throttle
            time.sleep(0.01)

        # Normalize rewards
        rewards = torch.stack(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        usage_dist = torch.tensor(node_counts, dtype=torch.float32).to(device)
        if usage_dist.sum() > 0:
            usage_dist /= usage_dist.sum()
        load_variance = torch.var(usage_dist)

        # REINFORCE loss
        loss = 0.0
        for log_p, r, ent in zip(log_probs, rewards, entropies):
            loss = loss + (-log_p * r - entropy_coef * ent)

        loss = loss + lambda_balance * load_variance

        reinforce_optimizer.zero_grad()
        loss.backward()
        reinforce_optimizer.step()
        print(f"[REINFORCE] Epoch {epoch+1} Loss: {loss.item():.2f} LoadVar: {load_variance.item():.4f} NodeCounts: {node_counts}")

# -------------------- STEP 7: Run everything end-to-end --------------------

def save_bar_plot(labels, values, title, ylabel, fname):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def cleanup_all_warm_containers():
    print("Cleaning up warm pool containers...")
    for node_pool in warm_pool:
        for k, v in list(node_pool.items()):
            try:
                cont = client.containers.get(v["name"])
                safe_container_remove(cont)
            except Exception:
                pass
        node_pool.clear()

def main():
    try:
        # Inference with initial (untrained) scheduler
        print("Running CASSIA inference (initial policy)...")
        responses, cold_events, totals = run_cassia_inference_on_test(scheduler, X_test, workload_test)

        # Baselines
        fifo_latency, fifo_cold, fifo_avg, fifo_total, fifo_throughput = run_baseline("fifo", workload_pattern=workload_test)
        rr_latency, rr_cold, rr_avg, rr_total, rr_throughput = run_baseline("round_robin", workload_pattern=workload_test)

        cassia_all = [lat for lst in responses for lat in lst]
        cassia_avg = np.mean(cassia_all) if cassia_all else 0
        cassia_total_cold = sum(cold_events)

        print("\nInitial Results:")
        print(f"FIFO avg: {fifo_avg:.2f} ms, RR avg: {rr_avg:.2f} ms, CASSIA avg: {cassia_avg:.2f} ms")
        print(f"Cold starts FIFO: {fifo_total}, RR: {rr_total}, CASSIA: {cassia_total_cold}")

        # Save initial plots
        labels = ['FIFO', 'RoundRobin', 'CASSIA']
        avg_latencies = [fifo_avg, rr_avg, cassia_avg]
        total_cold_starts = [fifo_total, rr_total, cassia_total_cold]
        os.makedirs("outputs", exist_ok=True)
        save_bar_plot(labels, avg_latencies, "Average Latency per Strategy (initial)", "Latency (ms)", "outputs/avg_latency_initial.png")
        save_bar_plot(labels, total_cold_starts, "Total Cold Starts per Strategy (initial)", "Cold Starts", "outputs/cold_starts_initial.png")
        print("Saved initial plots to outputs/")

        # Fine-tune with REINFORCE on real containers (small epochs!)
        print("\nStarting REINFORCE fine-tuning on real containers (this will spawn containers)...")
        train_reinforce_real(reinforce_model, X_train)

        # Evaluate REINFORCE
        print("Evaluating REINFORCE policy on test set...")
        reinforce_responses, reinforce_colds, reinforce_totals = run_cassia_inference_on_test(reinforce_model, X_test, workload_test)
        reinforce_all = [lat for lst in reinforce_responses for lat in lst]
        reinforce_avg = np.mean(reinforce_all) if reinforce_all else 0
        reinforce_total_cold = sum(reinforce_colds)

        print("\nREINFORCE Results:")
        print(f"REINFORCE avg latency: {reinforce_avg:.2f} ms, total cold starts: {reinforce_total_cold}")

        labels = ['FIFO', 'RoundRobin', 'CASSIA (initial)', 'REINFORCE']
        avg_latencies = [fifo_avg, rr_avg, cassia_avg, reinforce_avg]
        total_cold_starts = [fifo_total, rr_total, cassia_total_cold, reinforce_total_cold]
        save_bar_plot(labels, avg_latencies, "Average Latency per Strategy (final)", "Latency (ms)", "outputs/avg_latency_final.png")
        save_bar_plot(labels, total_cold_starts, "Total Cold Starts per Strategy (final)", "Cold Starts", "outputs/cold_starts_final.png")
        print("Saved final plots to outputs/")

    finally:
        # attempt cleanup to avoid leftover containers
        cleanup_all_warm_containers()
        print("Done. Check outputs/ for graphs.")

if __name__ == "__main__":
    main()
