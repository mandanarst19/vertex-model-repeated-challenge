#!/bin/bash
cd "$(dirname "$0")"
../../venv_pinned/bin/python stage2_core.py --sims 0 20 --dstar 0.7 --out ../results/stage2
../../venv_pinned/bin/python stage2_core.py --sims 0 10 --dstar 0.7 --conds ab --first-seed random --out ../results/stage2_rand
echo ALLDONE
