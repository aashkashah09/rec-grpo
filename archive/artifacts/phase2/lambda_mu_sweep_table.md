| λ | μ | always_api DR | best learned DR | winner | margin | learned wins? |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.00 | 0.892 | 0.890 | thompson_logistic | -0.002 | no |
| 0.00 | 0.05 | 0.864 | 0.862 | thompson_logistic | -0.002 | no |
| 0.00 | 0.10 | 0.836 | 0.834 | thompson_logistic | -0.001 | no |
| 0.00 | 0.20 | 0.780 | 0.778 | thompson_logistic | -0.001 | no |
| 0.10 | 0.00 | 0.806 | 0.805 | thompson_logistic | -0.001 | no |
| 0.10 | 0.05 | 0.778 | 0.777 | thompson_logistic | -0.001 | no |
| 0.10 | 0.10 | 0.750 | 0.749 | thompson_logistic | -0.001 | no |
| 0.10 | 0.20 | 0.694 | 0.692 | thompson_logistic | -0.002 | no |
| 0.20 | 0.00 | 0.721 | 0.719 | thompson_logistic | -0.002 | no |
| 0.20 | 0.05 | 0.693 | 0.691 | thompson_logistic | -0.002 | no |
| 0.20 | 0.10 | 0.665 | 0.662 | thompson_logistic | -0.002 | no |
| 0.20 | 0.20 | 0.609 | 0.608 | thompson_logistic | -0.001 | no |
| 0.30 | 0.00 | 0.635 | 0.634 | thompson_logistic | -0.002 | no |
| 0.30 | 0.05 | 0.607 | 0.608 | thompson_logistic | 0.000 | yes |
| 0.30 | 0.10 | 0.579 | 0.584 | epsilon_greedy | 0.005 | yes |
| 0.30 | 0.20 | 0.523 | 0.540 | epsilon_greedy | 0.017 | yes |
| 0.50 | 0.00 | 0.464 | 0.509 | epsilon_greedy | 0.044 | yes |
| 0.50 | 0.05 | 0.436 | 0.493 | epsilon_greedy | 0.057 | yes |
| 0.50 | 0.10 | 0.408 | 0.482 | epsilon_greedy | 0.074 | yes |
| 0.50 | 0.20 | 0.352 | 0.458 | thompson_logistic | 0.106 | yes |
| 1.00 | 0.00 | 0.037 | 0.421 | thompson_logistic | 0.384 | yes |
| 1.00 | 0.05 | 0.009 | 0.417 | thompson_logistic | 0.409 | yes |
| 1.00 | 0.10 | -0.019 | 0.414 | thompson_logistic | 0.433 | yes |
| 1.00 | 0.20 | -0.076 | 0.406 | thompson_logistic | 0.481 | yes |

_Stub-agent / CPU-simulator results — not real-model numbers (those land in Phase 4)._

Learned router beats always_api in 11/24 grid cells.
The win regime starts at λ ≥ 0.30 (cost weight); higher λ widens the margin.
Largest margin: +0.481 at λ=1.00, μ=0.20 (winner: thompson_logistic).
