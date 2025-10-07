#!/usr/bin/env bash

set -euf -o pipefail

dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd $dir

micromamba run -p ../../.rocm-env python fit.py                                \
    --tensor-ff-path ../04-parametrize/outputs/tensor_ff.pt                    \
    --tensor-tops-path ../04-parametrize/outputs/smiles_to_topologies.pt       \
    --train-dataset-paths ../02-select-data/datasets/tetramers/train           \
    --test-dataset-paths ../02-select-data/datasets/tetramers/test             \
    --training-config-json-path fit.jsonc                                      \
    --n-epochs 1000                                                            \
    --batch-size 500                                                           \
    --learning-rate 1e-3                                                       \
    --device gpu                                                               \
    --vram-limit-fraction=0.5                                                  \
    --fitting-dir-path tetramers-test-fit1                                     \
    --smirnoff-template ../03-generate-initial-ff/aam-ff.offxml
