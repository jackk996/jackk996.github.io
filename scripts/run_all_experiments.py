#!/usr/bin/env python
import argparse
p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/strict_paper.yaml'); a=p.parse_args(); print(f'Experiment runner scaffold for {a.config}. Real data required for numerical reproduction; synthetic runs must be labelled synthetic_demo.')
