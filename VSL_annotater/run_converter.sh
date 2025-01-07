#!/bin/bash

CHECKPOINT_DIR=/home/ldapusers/gogebakan/.llama/checkpoints/Llama3.2-3B-Instruct
INPUT_DIR=run_1_input/
OUTPUT_DIR=run_1_output/
# INPUT_DIR=example_inputs/
# OUTPUT_DIR=example_2_outputs/
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun converter.py $CHECKPOINT_DIR $INPUT_DIR $OUTPUT_DIR
