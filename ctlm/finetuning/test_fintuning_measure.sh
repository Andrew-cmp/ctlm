#!/bin/bash

bash ./rename_source_file.sh false default

PYTHONUNBUFFERED=1 python run_measure.py --reg_times=-1 --cuda_id1=1 --cuda_id2=2 |& tee run.log
