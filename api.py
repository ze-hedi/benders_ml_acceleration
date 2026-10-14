import asyncio
import os
from collections import defaultdict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlp import ShallowMLP

# --- config ---
INPUT_SIZE = 22   # 18 (state_vector) + 4 (per-subproblem: violation, duals_norm2, cost, subgradient_norm2)
TOP_K = 4
GAMMA = 0.99
LR = 1e-3
NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))  # backprop once all these envs finished an episode (run uvicorn with 1 worker)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ShallowMLP(INPUT_SIZE).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

app = FastAPI()

# episode buffers: env_id -> one entry per step
# each entry: top-k probs (with grad), top-k indices, reward from C++
buffers: dict[int, list[tuple[torch.Tensor, list[int], float]]] = defaultdict(list)
done: set[int] = set()          # envs whose episode ended, waiting for the batch update
batch_event = asyncio.Event()   # set when the batch update is done
last_result = None


# --- schemas ---
class StepRequest(BaseModel):
    env_id: int
    state_vector: list[float]                        # (18,) global Benders state
    subproblem_state_vectors: dict[str, list[float]] # name → [violation, duals_norm2, cost, subgradient_norm2]
    reward: float                                    # reward computed on C++ side for this step
    top_k: int = TOP_K


class EnvRequest(BaseModel):
    env_id: int


class StepResponse(BaseModel):
    selected_indices: list[int]    # top-k indices into the subproblem list
    selected_names: list[str]      # corresponding subproblem names
    selected_probs: list[float]    # their probabilities (for C++ logging)


class UpdateResponse(BaseModel):
    loss: float
    steps: int


# --- endpoints ---

@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    """
    Called at each env step.
    Runs forward pass, selects top-k, stores (top_k_probs, indices, reward) in buffer.
    """
    # build batch: one row per subproblem = state_vector + subproblem_state
    subproblem_names = list(req.subproblem_state_vectors.keys())
    batch = [req.state_vector + req.subproblem_state_vectors[name] for name in subproblem_names]
    x = torch.tensor(batch, dtype=torch.float32, device=device)  # (B, 22)

    logits = model(x).squeeze(1)           # (B,)
    policy = torch.softmax(logits, dim=0)  # (B,)

    k = min(req.top_k, policy.shape[0])
    top_k_probs, top_k_indices = torch.topk(policy, k)  # both (k,), top_k_probs retains grad_fn

    buffers[req.env_id].append((top_k_probs, top_k_indices.tolist(), req.reward))

    indices = top_k_indices.tolist()
    return StepResponse(
        selected_indices=indices,
        selected_names=[subproblem_names[i] for i in indices],
        selected_probs=top_k_probs.detach().cpu().tolist(),
    )


@app.post("/update", response_model=UpdateResponse)
async def update(req: EnvRequest):
    """
    Called at episode end by C++ for each env. Blocks until all NUM_ENVS episodes ended,
    then computes discounted returns G_t, Plackett-Luce log probs, REINFORCE loss, backprops once.
    """
    global batch_event, last_result
    if not buffers[req.env_id]:
        raise HTTPException(status_code=400, detail="Episode buffer is empty.")

    done.add(req.env_id)
    if len(done) < NUM_ENVS:
        await batch_event.wait()
        return last_result

    # --- compute discounted returns G_t per episode (backwards, no grad needed) ---
    steps = []
    returns: list[float] = []
    for env_steps in buffers.values():
        steps += env_steps
        episode_returns: list[float] = []
        G = 0.0
        for _, _, r in reversed(env_steps):
            G = r + GAMMA * G
            episode_returns.insert(0, G)
        returns += episode_returns

    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)  # (T,)
    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    # --- compute Plackett-Luce log prob per step ---
    log_probs: list[torch.Tensor] = []
    for top_k_probs, _, _ in steps:
        # top_k_probs is sorted descending by torch.topk
        # log P(seq) = sum_k log( p_(k) / (1 - sum_{j<k} p_(j)) )
        cumsum = torch.cumsum(top_k_probs, dim=0)           # (k,)
        consumed_before = torch.cat([
            torch.zeros(1, device=device),
            cumsum[:-1]
        ])                                                   # (k,) shifted by 1
        denominators = 1.0 - consumed_before                # remaining mass at each step
        log_prob = torch.sum(torch.log(top_k_probs / denominators))
        log_probs.append(log_prob)

    log_probs_tensor = torch.stack(log_probs)               # (T,)

    # --- REINFORCE loss ---
    loss = -(log_probs_tensor * returns_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    buffers.clear()
    done.clear()
    last_result = UpdateResponse(loss=loss.item(), steps=len(steps))
    batch_event.set()
    batch_event = asyncio.Event()
    return last_result


@app.post("/reset")
async def reset(req: EnvRequest):
    """Discard this env's buffer without training (e.g. on episode abort)."""
    for top_k_probs, _, _ in buffers[req.env_id]:
        top_k_probs.detach_()
    buffers[req.env_id].clear()
    return {"status": "buffer cleared"}
