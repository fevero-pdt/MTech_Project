#!/usr/bin/env python3
"""
scheduler/cassia_demo.py

Pool-based warm-pool scheduler + hybrid capacity-aware selection + active concurrency tracking.

Requirements:
- Docker running and function images available: cassia-login, cassia-upload, cassia-analytics, cassia-batch
- Docker network cassia-net (or change DOCKER_NETWORK)
"""

import os
import time
import random
from uuid import uuid4
from collections import deque

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

# Pool settings: POOL_K warm containers per (node, func_type)
POOL_K = 2
WARM_POOL_LIMIT = 1000

TRAIN_SAMPLES = 400
TEST_SAMPLES = 160
REINFORCE_EPOCHS = 2
BATCH_SIZE = 64
LOAD_PENALTY = 0.10

# Hybrid load-balancing parameter
LAMBDA_LOAD = 2.0  # tune: higher -> stronger avoidance of busy nodes

# Docker client
client = docker.from_env()

# warm_pool is list indexed by node; each entry is dict: func_type -> deque(records)
warm_pool = [dict() for _ in range(NUM_NODES)]
# rr_index per node to round-robin among pool entries for a func
rr_index = [dict() for _ in range(NUM_NODES)]

# Active and total request trackers (global across runs)
total_requests = [0 for _ in range(NUM_NODES)]   # cumulative assigned
active_requests = [0 for _ in range(NUM_NODES)]  # in-flight

# -------------------- WORKLOAD & INVOCATIONS --------------------

def generate_bursty_workload(num_nodes, steps, base_rate=30, burst_prob=0.25, skew_strength=0.7):
    workload = []
    skewed_probs = np.array([0.6] + [(1 - 0.6) / (num_nodes - 1)] * (num_nodes - 1))
    for t in range(steps):
        if random.random() < 0.4:
            node_loads = [0] * num_nodes
            for _ in range(max(1, num_nodes * base_rate)):
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

# -------------------- DATA PREP --------------------
print("Preparing data...")
raw_data = generate_function_invocations(n=TRAIN_SAMPLES + TEST_SAMPLES)
encoder = OneHotEncoder(sparse_output=False)
types_encoded = encoder.fit_transform([[row[0]] for row in raw_data])

scaler = StandardScaler()
numeric = scaler.fit_transform([[row[1], row[2], row[3], row[4], row[5]] for row in raw_data])
y = [row[6] for row in raw_data]
X = np.hstack((types_encoded, numeric))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SAMPLES / (TRAIN_SAMPLES + TEST_SAMPLES), random_state=42)
workload_train = generate_bursty_workload(NUM_NODES, steps=len(X_train), base_rate=30, burst_prob=0.25, skew_strength=0.7)
workload_test  = generate_bursty_workload(NUM_NODES, steps=len(X_test), base_rate=30, burst_prob=0.25, skew_strength=0.7)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# -------------------- SELF-SUPERVISED MODEL --------------------
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
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

print("Pretraining encoder (brief)...")
for epoch in range(2):
    tot = 0.0
    model.train()
    for bx, by in train_loader:
        bx = bx.to(device); by = by.to(device)
        opt.zero_grad()
        out = model(bx).squeeze()
        loss = crit(out, by)
        loss.backward()
        opt.step()
        tot += loss.item()
    print(f"Epoch {epoch+1}, Loss {tot:.4f}")

# -------------------- POLICY HEAD --------------------
class PolicyHead(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_nodes=NUM_NODES):
        super().__init__()
        self.encoder = encoder.encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )
    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

scheduler = PolicyHead(model).to(device)

# -------------------- DOCKER HELPERS --------------------
def safe_container_remove(container):
    try:
        container.stop(timeout=1)
    except Exception:
        pass
    try:
        container.remove()
    except Exception:
        pass

def wait_for_http(container_name, port=FUNCTION_PORT, timeout=6.0, interval=0.25):
    url = f"http://{container_name}:{port}/invoke"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.post(url, timeout=1)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

# -------------------- POOL-BASED ensure_container_for_function --------------------
def ensure_container_for_function(node, func_type_idx, user_id=None):
    """
    Pool-based: keep up to POOL_K containers per (node, func_type).
    Round-robin among them. Return (container_name, cold_flag, startup_ms).
    """
    node_pool = warm_pool[node]
    key = int(func_type_idx)
    if key not in node_pool:
        node_pool[key] = deque()

    pool = node_pool[key]

    # if pool has entries, choose next via round-robin
    if len(pool) > 0:
        idx = rr_index[node].get(key, 0) % len(pool)
        rr_index[node][key] = (idx + 1) % max(1, len(pool))
        rec = pool[idx]
        return rec["name"], 0, 0.0

    # else create new container (cold)
    safe_suffix = uuid4().hex[:6]
    container_name = f"node{node}-f{func_type_idx}-p{len(pool)}-{safe_suffix}"
    image = FUNCTION_IMAGES[func_type_idx]
    try:
        container = client.containers.run(image, name=container_name, detach=True, network=DOCKER_NETWORK)
    except docker.errors.APIError:
        container = client.containers.run(image, detach=True, network=DOCKER_NETWORK)
        container_name = container.name

    t0 = time.monotonic()
    ready = wait_for_http(container_name, port=FUNCTION_PORT, timeout=6.0)
    t1 = time.monotonic()
    startup_time_ms = (t1 - t0) * 1000.0
    pool.append({"name": container_name, "container": container, "started_at": time.time()})
    # if pool bigger than POOL_K, evict oldest
    while len(pool) > POOL_K:
        old = pool.popleft()
        try:
            safe_container_remove(old["container"])
        except Exception:
            pass
    print(f"[COLD START-POOL] node={node} func={func_type_idx} container={container_name} startup_ms={startup_time_ms:.1f}")
    return container_name, 1, startup_time_ms

def invoke_function_docker(node, func_type_idx, extra_load=0.0, user_id=None):
    """
    Returns: total_latency_ms, startup_time_ms, invoke_time_ms, cold_flag
    (Does NOT modify active_requests; caller must increment/decrement around this call.)
    """
    container_name, cold_flag, startup_ms = ensure_container_for_function(node, func_type_idx, user_id=user_id)
    url = f"http://{container_name}:{FUNCTION_PORT}/invoke"
    t0 = time.monotonic()
    try:
        r = requests.post(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] invocation failed for {container_name}: {e}")
        try:
            cont = client.containers.get(container_name)
            safe_container_remove(cont)
        except Exception:
            pass
        invoke_ms = 5000.0
    else:
        t1 = time.monotonic()
        invoke_ms = (t1 - t0) * 1000.0
    total_latency = startup_ms + invoke_ms + extra_load * LOAD_PENALTY
    return total_latency, startup_ms, invoke_ms, cold_flag

# -------------------- RESET & CLEANUP --------------------
def reset_warm_pool():
    print("[ACTION] Resetting warm pool (stopping and removing existing warm containers)...")
    for node_idx in range(NUM_NODES):
        node_pool = warm_pool[node_idx]
        for func_k, dq in list(node_pool.items()):
            while dq:
                rec = dq.popleft()
                try:
                    cont = client.containers.get(rec["name"])
                    safe_container_remove(cont)
                except Exception:
                    pass
        node_pool.clear()
        rr_index[node_idx].clear()
    # reset trackers
    for i in range(NUM_NODES):
        total_requests[i] = 0
        active_requests[i] = 0

def cleanup_all_warm_containers():
    print("Final cleanup of warm pool containers...")
    reset_warm_pool()

# -------------------- REAL BASELINE (FIFO/RR using real containers) --------------------
def run_baseline_real(strategy="fifo", workload_pattern=None):
    """
    Run FIFO or RoundRobin but using real container invocations (invoke_function_docker).
    Returns: response_times, startup_times, invoke_times, cold_counts, requests_per_node
    """
    if workload_pattern is None:
        workload_pattern = [[0]*NUM_NODES for _ in range(len(X_test))]

    response_times = [[] for _ in range(NUM_NODES)]
    startup_times = [[] for _ in range(NUM_NODES)]
    invoke_times = [[] for _ in range(NUM_NODES)]
    cold_counts = [0] * NUM_NODES
    requests = [0] * NUM_NODES

    rr_counter = 0
    node_free_time = [0.0] * NUM_NODES  # used for FIFO selection as before

    for i in range(len(X_test)):
        # choose node exactly like simulated baseline to be comparable
        if strategy == "fifo":
            node = min(range(NUM_NODES), key=lambda n: node_free_time[n])
        elif strategy == "round_robin":
            node = rr_counter % NUM_NODES
            rr_counter += 1
        else:
            node = 0

        x_np = X_test[i].astype(np.float32)
        func_type = int(np.argmax(x_np[:4]))
        user_id = int(scaler.inverse_transform([x_np[4:]])[0][0])
        extra_load = workload_pattern[i][node]

        # mark assigned and active
        total_requests[node] += 1
        active_requests[node] += 1

        total_latency, startup_ms, invoke_ms, cold = invoke_function_docker(node, func_type, extra_load=extra_load, user_id=user_id)

        # decrement active
        active_requests[node] = max(0, active_requests[node] - 1)

        response_times[node].append(total_latency)
        startup_times[node].append(startup_ms)
        invoke_times[node].append(invoke_ms)
        if cold:
            cold_counts[node] += 1
        requests[node] += 1

        # approximate finish_time by adding measured latency (ms -> seconds)
        finish_time = node_free_time[node] + (total_latency / 1000.0)
        node_free_time[node] = finish_time

        # throttle
        time.sleep(0.005)

    return response_times, startup_times, invoke_times, cold_counts, requests

# -------------------- CASSIA/REINFORCE inference function (hybrid selection) --------------------
def run_cassia_inference_on_test(scheduler_model, X_test_local, workload_pattern):
    """
    Runs the learned policy (CASSIA or REINFORCE) on real containers using hybrid selection.
    Returns: response_times, startup_times, invoke_times, cold_start_events, total_requests
    """
    response_times = [[] for _ in range(NUM_NODES)]
    startup_times = [[] for _ in range(NUM_NODES)]
    invoke_times = [[] for _ in range(NUM_NODES)]
    cold_start_events = [0] * NUM_NODES
    total_reqs_local = [0] * NUM_NODES

    scheduler_model.eval()
    with torch.no_grad():
        for i in range(len(X_test_local)):
            x_np = X_test_local[i].astype(np.float32)
            x = torch.tensor(x_np, dtype=torch.float32).to(device)

            func_type = int(np.argmax(x_np[:4]))
            user_id  = int(scaler.inverse_transform([x_np[4:]])[0][0])

            # model scores
            scores_tensor = scheduler_model(x).cpu().detach()
            scores = scores_tensor.numpy().ravel()  # shape (NUM_NODES,)

            # compute normalized current load from active_requests
            loads = np.array(active_requests, dtype=float) + 1e-8
            norm_loads = loads / max(1.0, loads.mean())

            # hybrid combined score: higher is better
            combined = scores - LAMBDA_LOAD * norm_loads

            # pick best node (deterministic). alternative: sample from adjusted softmax if you want exploration.
            node = int(np.argmax(combined))

            # book-keep: assign
            total_requests[node] += 1
            active_requests[node] += 1
            total_reqs_local[node] += 1

            extra_load = workload_pattern[i][node] if i < len(workload_pattern) else 0

            total_latency, startup_ms, invoke_ms, cold = invoke_function_docker(node, func_type, extra_load=extra_load, user_id=user_id)

            # done
            active_requests[node] = max(0, active_requests[node] - 1)

            response_times[node].append(total_latency)
            startup_times[node].append(startup_ms)
            invoke_times[node].append(invoke_ms)
            if cold:
                cold_start_events[node] += 1

            time.sleep(0.005)

    return response_times, startup_times, invoke_times, cold_start_events, total_reqs_local

# -------------------- SIMULATED BASELINE (kept for reference) --------------------
def run_baseline_sim(strategy="fifo", workload_pattern=None):
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
    return baseline_latency, baseline_cold, avg_latency, total_cold, throughput_per_node, baseline_requests

# -------------------- REINFORCE (with load-aware logits) --------------------
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

def train_reinforce_real(model, X_data, epochs=REINFORCE_EPOCHS, max_samples_for_real=120):
    """
    Train with load-aware logits: subtract a load term from logits before sampling.
    Use a small subset (max_samples_for_real) for real invocations to avoid host overload.
    """
    lambda_balance = 0.05
    entropy_coef = 0.02
    N = min(len(X_data), max_samples_for_real)
    for epoch in range(epochs):
        node_counts = [0] * NUM_NODES
        log_probs, rewards, entropies = [], [], []

        model.train()
        for i in range(N):
            x_np = X_data[i].astype(np.float32)
            x = torch.tensor(x_np, dtype=torch.float32).to(device)
            func_type = int(np.argmax(x_np[:4]))
            user_id = int(scaler.inverse_transform([x_np[4:]])[0][0])

            logits = model(x)  # shape (num_nodes,)
            # compute current normalized loads (use active_requests snapshot)
            loads = np.array(active_requests, dtype=float) + 1e-8
            norm_loads = loads / max(1.0, loads.mean())
            load_tensor = torch.tensor(norm_loads, dtype=torch.float32).to(device)

            # subtract load penalty from logits (capacity-aware)
            adjusted_logits = logits - (LAMBDA_LOAD * load_tensor)

            probs = F.softmax(adjusted_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            node = int(action.item())
            node_counts[node] += 1

            # bookkeeping
            total_requests[node] += 1
            active_requests[node] += 1

            extra_load = workload_train[i][node] if i < len(workload_train) else 0
            total_latency, startup_ms, invoke_ms, cold = invoke_function_docker(node, func_type, extra_load=extra_load, user_id=user_id)

            # release
            active_requests[node] = max(0, active_requests[node] - 1)

            # capacity-aware reward
            avg_per_node = max(1.0, N / float(NUM_NODES))
            usage_ratio = node_counts[node] / avg_per_node
            load_pen = 50.0 * max(0.0, usage_ratio - 1.0)
            reward = -total_latency - load_pen

            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            entropies.append(entropy)

            time.sleep(0.005)

        if len(rewards) == 0:
            continue
        rewards = torch.stack(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        usage_dist = torch.tensor(node_counts, dtype=torch.float32).to(device)
        if usage_dist.sum() > 0:
            usage_dist /= usage_dist.sum()
        load_variance = torch.var(usage_dist)

        loss = 0.0
        for lp, r, ent in zip(log_probs, rewards, entropies):
            loss = loss + (-lp * r - entropy_coef * ent)
        loss = loss + lambda_balance * load_variance

        reinforce_optimizer.zero_grad()
        loss.backward()
        reinforce_optimizer.step()
        print(f"[REINFORCE] Epoch {epoch+1} Loss {loss.item():.2f} LoadVar {load_variance.item():.4f} NodeCounts {node_counts}")

# -------------------- UTILITIES: percentiles / printing / prewarm --------------------
def percentiles_from_list(lat_list):
    if not lat_list:
        return 0.0, 0.0
    a = np.array(lat_list)
    return np.percentile(a, 50), np.percentile(a, 95)

def print_overall_percentiles(name, response_times):
    all_lats = [l for lst in response_times for l in lst]
    if not all_lats:
        print(f"{name}: no data")
        return
    p50, p95 = percentiles_from_list(all_lats)
    print(f"{name}: p50={p50:.2f} ms, p95={p95:.2f} ms, mean={np.mean(all_lats):.2f} ms")

def prewarm_all(node_count=NUM_NODES, per_node_per_func=POOL_K):
    print("[ACTION] Prewarming pools...")
    for node in range(node_count):
        for func in range(len(FUNCTION_IMAGES)):
            for j in range(per_node_per_func):
                try:
                    ensure_container_for_function(node, func, user_id=9999 + j)
                except Exception as e:
                    print("prewarm error:", e)
    print("[ACTION] Prewarming done.")

def save_bar_plot(labels, values, title, ylabel, fname):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def print_node_stats(prefix, response_times, startup_times, invoke_times, cold_events, total_requests_local):
    print(f"\n--- {prefix} Node-wise breakdown ---")
    for i in range(NUM_NODES):
        avg_lat = np.mean(response_times[i]) if response_times[i] else 0
        avg_start = np.mean(startup_times[i]) if startup_times[i] else 0
        avg_invoke = np.mean(invoke_times[i]) if invoke_times[i] else 0
        containers_kept = sum(len(warm_pool[i].get(k, [])) for k in warm_pool[i].keys())
        print(f"Node {i}: reqs={total_requests_local[i]}, containers_kept={containers_kept}, avg_total_lat={avg_lat:.2f} ms, avg_startup={avg_start:.2f} ms, avg_invoke={avg_invoke:.2f} ms, cold_starts={cold_events[i]}")

# -------------------- MAIN: run everything --------------------
def main():
    global scaler
    try:
        # 1) Real FIFO baseline
        reset_warm_pool()
        print("\nRunning FIFO baseline (real containers)...")
        fifo_resp, fifo_start, fifo_invoke, fifo_cold_counts, fifo_reqs = run_baseline_real("fifo", workload_pattern=workload_test)
        fifo_all = [lat for lst in fifo_resp for lat in lst]
        fifo_avg_real = np.mean(fifo_all) if fifo_all else 0
        print_node_stats("FIFO (real)", fifo_resp, fifo_start, fifo_invoke, fifo_cold_counts, fifo_reqs)
        print_overall_percentiles("FIFO (real)", fifo_resp)

        # 2) Real Round Robin baseline
        reset_warm_pool()
        print("\nRunning RoundRobin baseline (real containers)...")
        rr_resp, rr_start, rr_invoke, rr_cold_counts, rr_reqs = run_baseline_real("round_robin", workload_pattern=workload_test)
        rr_all = [lat for lst in rr_resp for lat in lst]
        rr_avg_real = np.mean(rr_all) if rr_all else 0
        print_node_stats("RR (real)", rr_resp, rr_start, rr_invoke, rr_cold_counts, rr_reqs)
        print_overall_percentiles("RR (real)", rr_resp)

        # 3) CASSIA initial policy (real)
        reset_warm_pool()
        print("\nRunning CASSIA (initial policy) on real containers...")
        cassia_resp, cassia_start, cassia_invoke, cassia_cold, cassia_reqs = run_cassia_inference_on_test(scheduler, X_test, workload_test)
        cassia_all = [lat for lst in cassia_resp for lat in lst]
        cassia_avg = np.mean(cassia_all) if cassia_all else 0
        print_node_stats("CASSIA (initial)", cassia_resp, cassia_start, cassia_invoke, cassia_cold, cassia_reqs)
        print_overall_percentiles("CASSIA (initial)", cassia_resp)

        # save initial plots
        os.makedirs("outputs", exist_ok=True)
        save_bar_plot(['FIFO(real)', 'RR(real)', 'CASSIA'], [fifo_avg_real, rr_avg_real, cassia_avg],
                      "Average Latency per Strategy (initial-real)", "Latency (ms)", "outputs/avg_latency_initial_real.png")
        save_bar_plot(['FIFO(real)', 'RR(real)', 'CASSIA'], [sum(fifo_cold_counts), sum(rr_cold_counts), sum(cassia_cold)],
                      "Total Cold Starts per Strategy (initial-real)", "Cold Starts", "outputs/cold_starts_initial_real.png")

        # 4) Fine-tune REINFORCE (small real subset)
        print("\nStarting REINFORCE fine-tuning (small real subset)...")
        train_reinforce_real(reinforce_model, X_train, epochs=REINFORCE_EPOCHS, max_samples_for_real=120)

        # 5) Evaluate REINFORCE
        reset_warm_pool()
        print("\nEvaluating REINFORCE policy on test set (real containers)...")
        reinforce_resp, reinforce_start, reinforce_invoke, reinforce_cold, reinforce_reqs = run_cassia_inference_on_test(reinforce_model, X_test, workload_test)
        reinforce_all = [lat for lst in reinforce_resp for lat in lst]
        reinforce_avg = np.mean(reinforce_all) if reinforce_all else 0
        print_node_stats("REINFORCE", reinforce_resp, reinforce_start, reinforce_invoke, reinforce_cold, reinforce_reqs)
        print_overall_percentiles("REINFORCE", reinforce_resp)

        # final plots
        labels = ['FIFO(real)', 'RR(real)', 'CASSIA (initial)', 'REINFORCE']
        avg_latencies = [fifo_avg_real, rr_avg_real, cassia_avg, reinforce_avg]
        total_cold_starts = [sum(fifo_cold_counts), sum(rr_cold_counts), sum(cassia_cold), sum(reinforce_cold)]
        save_bar_plot(labels, avg_latencies, "Average Latency per Strategy (final-real)", "Latency (ms)", "outputs/avg_latency_final_real.png")
        save_bar_plot(labels, total_cold_starts, "Total Cold Starts per Strategy (final-real)", "Cold Starts", "outputs/cold_starts_final_real.png")
        print("Plots saved to outputs/")

    finally:
        cleanup_all_warm_containers()
        print("Done. Cleaned up containers.")

if __name__ == "__main__":
    main()
