## Dependencies
```
pytorch, sklearn, numpy
```

## Running the Experiments
### Loss-function mismatch, data-distribution mismatch, hypothesis-space mismatch
```
bash run_script/run_*TASK*_mismatch.sh
```
where `*TASK*` is taken as `loss-function_mismatch` for loss-function mismatch, `data-distribution_mismatch` for data-distribution mismatch and `hypothesis-space_mismatch` for hypothesis-space mismatch.

### Upstream-upstream entanglement
Run the upstream script first.
```
bash run_script/run_upstream-upstream_entanglement-up.sh
```
After the upstream's training is finished, run the downstream script.
```
bash run_script/run_upstream-upstream_entanglement-down.sh
```

## Result Plots
See the jupyter notebook `Figures.ipynb`
