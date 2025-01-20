#!/bin/bash

set -x

nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --force-overwrite true \
    --capture-range nvtx --nvtx-capture VCIProfiledSection \
    --output test_range  -x true  \
    python   -m vci.train  --config conf/defaults.yaml
