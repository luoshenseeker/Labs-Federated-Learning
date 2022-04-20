import os
import json

filenames = os.listdir("saved_exp_info/acc")

filenames = [i for i in filenames if i[-3:] == "pkl"]

exps = {}

sampling_random = "random"
sampling_FedAvg = "FedAvg"
sampling_clustered_1 = "clustered_1"
sampling_ours = "ours"

for filename in filenames:
    if "random" in filename:
        exp_arg = filename.replace("random")