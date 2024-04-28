#!/bin/bash

# Run the main script and save outputs (.out) and errors (.err)
python3 main.py 2>&1 | tee drc.out