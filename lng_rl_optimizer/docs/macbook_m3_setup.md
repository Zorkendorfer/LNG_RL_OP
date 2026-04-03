# MacBook M3 Pro Setup

Use this on macOS to continue the project with Apple Silicon acceleration.

## 1. Create and activate a virtual environment

```bash
cd /path/to/LNG_RL_OP/lng_rl_optimizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
```

## 2. Confirm PyTorch can use MPS

```bash
python -c "import torch; print(torch.__version__); print('mps available =', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('mps built =', hasattr(torch.backends, 'mps') and torch.backends.mps.is_built())"
```

Expected result on an M3 Pro: `mps available = True`

## 3. Run the project on Apple Silicon

```bash
python scripts/check_env.py
python scripts/generate_synthetic_nordpool_data.py
python scripts/generate_surrogate_data.py --n-episodes 500 --episode-length 168
python scripts/train_surrogate.py --epochs 50 --device mps
python scripts/train_price_forecaster.py --synthetic --epochs 20 --device mps
python scripts/train_agent.py --synthetic-prices --device mps
```

## 4. If MPS has trouble with a workload

Fallback to CPU:

```bash
python scripts/train_surrogate.py --epochs 50 --device cpu
python scripts/train_price_forecaster.py --synthetic --epochs 20 --device cpu
python scripts/train_agent.py --synthetic-prices --device cpu
```
