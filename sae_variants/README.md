# SAE Variants

This directory hosts custom sparse autoencoders that can be trained with the BatchTopK
infrastructure without modifying that repository. Import them via

```python
from sae_variants import BandedCovarianceSAE

sae = BandedCovarianceSAE(cfg)
```

## BandedCovarianceSAE

Implements the bSAE idea described in the provided spec:

* Uses `BaseAutoencoder` from BatchTopK for preprocessing, decoder renormalisation,
  and bookkeeping.
* Supports two sparsity penalties:
  * `sparsity_mode="l1"` → vanilla L1 (mean over batch of the per-example sum).
  * `sparsity_mode="l0"` → smooth L0 approximation with parameters `delta` and `epsilon`.
* Optional universal scalars `beta_k` (`use_beta=True/False`) and per-latent
  correlation parameters `alpha_i` (`use_alpha=True/False`).
* Adds a banded covariance penalty parameterised by:
  * `p`: number of preceding latents to correlate.
  * `beta_slope`: scaling inside the tanh term.
  * `lambda_ar`: weight on the AR loss term.
* The total loss is

  ```
  L = L_recon
      + lambda_sparse * L_sparse
      + lambda_ar * L_AR
  ```

### Expected config keys

Alongside the standard BatchTopK keys (`act_size`, `dict_size`, `lr`, etc.),
specify the following:

| Key | Description | Default |
| --- | ----------- | ------- |
| `lambda_sparse` | weight on sparsity penalty | `0.0` |
| `lambda_ar` | weight on AR/banded covariance penalty | `0.0` |
| `sparsity_mode` | `"l1"` or `"l0"` (smooth approximation) | `"l0"` |
| `delta`, `epsilon` | parameters of smooth L0 penalty | `1.0`, `1e-4` |
| `p` | AR bandwidth | `1` |
| `beta_slope` | slope in tanh term | `1.0` |
| `use_beta` | learn β scalars per lag | `True` |
| `use_alpha` | learn α per latent | `True` |

The forward pass returns the same dictionary keys as BatchTopK SAEs so it
can be trained via `BatchTopK.training.train_sae`.
