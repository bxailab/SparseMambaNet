import json
import os
import argparse

def model_args():
    parser = argparse.ArgumentParser()
    print("extracting arguments")
    ## Model Settings
    parser.add_argument("--input_dim", type=int, default=256, help = "It is default setting, not real input_dim")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_state", type=int, default=256)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)



    args, _ = parser.parse_known_args()

    return args

