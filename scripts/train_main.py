#!/usr/bin/env python
import argparse
from src.utils.config import load_config
p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/corrected.yaml'); a=p.parse_args(); print(load_config(a.config)); print('Training entry point scaffold: supply real data to run full training.')
