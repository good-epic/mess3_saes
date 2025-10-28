#!/usr/bin/bash

echo "Force JAX to use CPU"
python - <<'PY'
print("Loading imports")
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
from simplexity.generative_processes.builder import build_generalized_hidden_markov_model

print("Creating GHMM Instance")a
tomq = build_generalized_hidden_markov_model("tom_quantum", alpha=1.07, beta=7.1)
PY


echo "Don't force JAX to use CPU"
python - <<'PY'
print("Loading imports")
import os
import jax
from simplexity.generative_processes.builder import build_generalized_hidden_markov_model

print("Creating GHMM Instance")
tomq = build_generalized_hidden_markov_model("tom_quantum", alpha=1.07, beta=7.1)
PY