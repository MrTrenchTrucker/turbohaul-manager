# System Requirements

## Minimum

The minimum specification covers this workload:

- One 35B-A3B mixture-of-experts model at Q4 quantization or smaller
- Inactive experts offloaded to system RAM
- 250K context window with Turbo 3 KV cache
- Turbohaul Manager and one agent harness running on the same machine

| Component | Minimum |
|---|---|
| CPU | AMD Ryzen 5 3600 or equivalent (6 cores / 12 threads, AVX2) |
| RAM | 32GB DDR4-3200 in dual-channel mode (2×16GB) |
| GPU | NVIDIA RTX 2060 Super or better (RTX-class, 8GB VRAM) |
| Storage | NVMe SSD, 250GB |

Notes:

- RAM must run in dual-channel mode. Use two or four matched sticks, not a single stick.
- The GPU must be an RTX-class card with at least 8GB of VRAM. GTX cards and 6GB cards (RTX 2060, RTX 3050 6GB) do not meet the minimum.

### CPU-only operation

Turbohaul Manager can run models with no graphics card. This is strongly not recommended for chat or agent models. CPU-only operation is recommended only for embedding models.

## Recommended

The recommended specification covers this workload:

- One 27B model at Q4_K_M quantization, fully in VRAM
- 250K context window with Turbo 3 KV cache
- Vision projector and MTP enabled
- Turbohaul Manager and one agent harness running on the same machine

| Component | Recommended |
|---|---|
| CPU | AMD Ryzen 7 9700X or better (8+ cores) |
| RAM | 64GB DDR5-6000 in dual-channel mode |
| GPU | 24GB VRAM: a single 24GB NVIDIA RTX card, or two RTX 30-series or newer cards totaling 24GB |
| Storage | NVMe SSD, 1TB |

Notes:

- Single-card examples: RTX 3090, RTX 4090, or better.
- Two-card example: 2× RTX 3060 12GB.
- Consumer RTX cards are recommended over the Pro series.

## High tier

The high tier is the class of system Turbohaul Manager is developed on. The developer's system runs 14 agents on a mix of local and external models.

| Component | High tier |
|---|---|
| CPU | AMD Ryzen 9 9900X or better |
| RAM | 128GB DDR5 or more |
| GPU | 48GB VRAM or more: NVIDIA RTX 50-series or newer, or Pro series cards |
| Storage | 2TB system disk or larger |

Notes:

- The RTX PRO 4000 SFF works, but its token generation speed is comparable to an RTX 5060.
- There is no upper limit on this tier.
