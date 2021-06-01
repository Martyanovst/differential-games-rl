import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, choices=('simple-motions', 'van-der-pol', 'pendulum', 'dubins-car'),
                    required=True)
parser.add_argument('--model', type=str, choices=('naf', 'b-naf', 'rb-naf', 'gb-naf'), required=True)
parser.add_argument('--epoch-num', type=int, default=500)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--save-model-path', type=str, default=None)
parser.add_argument('--save-train-rewards-path', type=str, default=None)
parser.add_argument('--save-train-plot-path', type=str, default=None)
