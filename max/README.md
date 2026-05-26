# Max/MSP Patches

Place `.maxpat` patches here. Each MVP has a paired OSC port:

| MVP | Listen (Python <- Max) | Send (Python -> Max) |
|-----|------------------------|----------------------|
| A   | 7400 | 7401 |
| B   | 7410 | 7411 |
| C   | 7420 | 7421 |
| D   | 7430 | 7431 |

## RAVE in Max
MVP-A and MVP-D require the `nn~` external from ACIDS-IRCAM:
- https://github.com/acids-ircam/nn_tilde

Load a `.ts` model in `nn~` for native realtime inference, OR run the Python side and exchange audio via `[udpsend~]`/`[udpreceive~]` over OSC.
