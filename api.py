import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlp import ShallowMLP

# --- config ---
INPUT_SIZE = 22   # 18 (state_vector) + 4 (per-subproblem: violation, duals_norm2, cost, subgradient_norm2)
TOP_K = 4
GAMMA = 0.99
LR = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ShallowMLP(INPUT_SIZE).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

app = FastAPI()

# episode buffer: one entry per step
# each entry: top-k probs (with grad), top-k indices, reward from C++
episode_buffer: list[tuple[torch.Tensor, list[int], float]] = []


# --- schemas ---
class StepRequest(BaseModel):
    state_vector: list[float]                        # (18,) global Benders state
    subproblem_state_vectors: dict[str, list[float]] # name → [violation, duals_norm2, cost, subgradient_norm2]
    reward: float                                    # reward computed on C++ side for this step
    top_k: int = TOP_K


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

    episode_buffer.append((top_k_probs, top_k_indices.tolist(), req.reward))

    indices = top_k_indices.tolist()
    return StepResponse(
        selected_indices=indices,
        selected_names=[subproblem_names[i] for i in indices],
        selected_probs=top_k_probs.detach().cpu().tolist(),
    )


@app.post("/update", response_model=UpdateResponse)
async def update():
    """
    Called at episode end by C++.
    Computes discounted returns G_t, Plackett-Luce log probs, REINFORCE loss, backprops.
    """
    if not episode_buffer:
        raise HTTPException(status_code=400, detail="Episode buffer is empty.")

    # --- compute discounted returns G_t (backwards, no grad needed) ---
    rewards = [r for _, _, r in episode_buffer]
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)  # (T,)
    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    # --- compute Plackett-Luce log prob per step ---
    log_probs: list[torch.Tensor] = []
    for top_k_probs, _, _ in episode_buffer:
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

    n = len(episode_buffer)
    episode_buffer.clear()

    return UpdateResponse(loss=loss.item(), steps=n)


@app.post("/reset")
async def reset():
    """Discard buffer without training (e.g. on episode abort)."""
    for top_k_probs, _, _ in episode_buffer:
        top_k_probs.detach_()
    episode_buffer.clear()
    return {"status": "buffer cleared"}
