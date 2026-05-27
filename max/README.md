# Max/MSP Patches

Place `.maxpat` patches here. Each MVP module (A through I) has a dedicated OSC port pair for real-time control and remote rendering.

## OSC Port Map

| MVP | Domain | Python Listen (<- Max) | Python Send (-> Max) |
|:---:|:---|:---:|:---:|
| **A** | Latent Perturb | 7400 | 7401 |
| **B** | Caption Loop | 7410 | 7411 |
| **C** | Token Bending | 7420 | 7421 |
| **D** | Morphing | 7430 | 7431 |
| **E** | Neural Granular | 7440 | 7441 |
| **F** | Spectral Freeze | 7450 | 7451 |
| **G** | Latent Feedback | 7460 | 7461 |
| **H** | Codebook Organ | 7470 | 7471 |
| **I** | Bass Massive | 7480 | 7481 |

## RAVE Integration
For RAVE-based modules (A, D, E, F, G), we recommend using the `nn~` external for native Max/MSP inference:
- [nn~ GitHub](https://github.com/acids-ircam/nn_tilde)

You can load the `.ts` checkpoints directly in `nn~` and use standard Max messages to perturb the latent space. For EnCodec-based modules (C, H, I), use the Python OSC bridge to trigger asynchronous processing.
