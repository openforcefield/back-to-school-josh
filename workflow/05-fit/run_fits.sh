#!/usr/bin/env bash

set -euf -o pipefail

dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd $dir

pixi r python fit.py \
    --tensor-ff-path ../04-parametrize/outputs/tensor_ff.pt                    \
    --tensor-tops-path ../04-parametrize/outputs/smiles_to_topologies.pt       \
    --train-dataset-paths ../02-select-data/datasets/spice2/train              \
    --test-dataset-paths ../02-select-data/datasets/spice2/test                \
    --training-config-json-path fit.jsonc                                      \
    --n-epochs 1000                                                            \
    --learning-rate 0.001                                                      \
    --batch-size 500

pixi r python fit.py \
    --tensor-ff-path ../04-parametrize/outputs/tensor_ff.pt                    \
    --tensor-tops-path ../04-parametrize/outputs/smiles_to_topologies.pt       \
    --train-dataset-paths                                                      \
        ../02-select-data/datasets/spice2/train                                \
        ../02-select-data/datasets/tetramers/train                             \
    --test-dataset-paths                                                       \
        ../02-select-data/datasets/spice2/test                                 \
        ../02-select-data/datasets/tetramers/test                              \
    --training-config-json-path fit.jsonc                                      \
    --n-epochs 1000                                                            \
    --learning-rate 0.001                                                      \
    --batch-size 500
