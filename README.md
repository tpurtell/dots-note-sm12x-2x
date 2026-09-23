This is a recipe for running dots note on 2xRT 6000 or a pair of DGX Spark.
# dots-note-sm12x-2x

## Third-party sources

The quantization and serving sources are pinned as Git submodules:

| Path | Purpose | Branch |
| --- | --- | --- |
| `third_party/GPTQModel` | Quantization | `main` |
| `third_party/sparkinfer-glmrt` | GPU kernels and serving integration | `master` |

Clone with `git clone --recurse-submodules https://github.com/tpurtell/dots-note-sm12x-2x.git`, or run `git submodule update --init --recursive` after an ordinary clone. The parent repository records exact commits; a fresh checkout does not depend on the current tips of the submodule branches.

Make library changes inside the corresponding submodule and push them to its own repository before committing the updated submodule pointer here. To advance both submodules to their tracked branches, run `git submodule update --remote --merge`, review and test the changes, then commit the pointer updates in this repository.
