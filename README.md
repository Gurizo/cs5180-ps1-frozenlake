# CS5180 PS1 FrozenLake

Planning algorithms for Gymnasium `FrozenLake-v1` with `is_slippery=True`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files

- `CS5180_PS1_Problem3_FrozenLake.ipynb`: notebook with code, outputs, plot, and report text.
- `vi.py`: value iteration for Problem 3(a) and 3(d).
- `pi.py`: policy iteration for Problem 3(b).
- `compare.py`: comparison runner for Problem 3(c).
- `utils.py`: environment, formatting, and timing helpers.

## Commands

```bash
jupyter notebook CS5180_PS1_Problem3_FrozenLake.ipynb
python vi.py --gamma 0.99 --theta 1e-4
python pi.py --gamma 0.99
python compare.py
```

`compare.py` writes:

- `results.json`
- `iteration_counts.png`
