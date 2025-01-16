#!/usr/bin/env bash


PYTHONPATH="$(dirname $0)/..":PYTHONPATH \
python $(dirname $0)/main_test_swinir.py \
${@:1}
