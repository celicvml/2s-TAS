import torch
 
from model_load import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 20011015 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')

args = parser.parse_args()
 
num_epochs = 120

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim_RGB = 768
features_dim_FLOW = 1024
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001


vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path_RGB = "./data/"+args.dataset+"/768_features/"
features_path_FLOW = "./data/"+args.dataset+"/flow_features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"
 
mapping_file = "./data/"+args.dataset+"/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)

def eval():
    cnt_split_dict = {
        '50salads': 5,
        'gtea': 4,
        'breakfast': 4
    }
    acc_all = 0.
    edit_all = 0.
    f1s_all = [0., 0., 0.]

    if args.split == 0:
        for split in range(1, cnt_split_dict[args.dataset] + 1):
            recog_path = "./{}/".format(args.result_dir) + args.dataset + "/split_{}".format(split) + "/"
            file_list = "./data/" + args.dataset + "/splits/test.split{}".format(split) + ".bundle"
            acc, edit, f1s = func_eval(args.dataset, recog_path, file_list)
            acc_all += acc
            edit_all += edit
            f1s_all[0] += f1s[0]
            f1s_all[1] += f1s[1]
            f1s_all[2] += f1s[2]

        acc_all /= cnt_split_dict[args.dataset]
        edit_all /= cnt_split_dict[args.dataset]
        f1s_all = [i / cnt_split_dict[args.dataset] for i in f1s_all]
    else:
        split = args.split
        recog_path = "./{}/".format(args.result_dir) + args.dataset + "/split_{}".format(split) + "/"
        file_list = "./data/" + args.dataset + "/splits/test.split{}".format(split) + ".bundle"
        acc_all, edit_all, f1s_all = func_eval(args.dataset, recog_path, file_list)

    print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim_RGB, features_dim_FLOW , num_classes, channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path_RGB, features_path_FLOW, sample_rate)
    batch_gen.read_data(vid_list_file)

    # batch_gen_FLOW = BatchGenerator(num_classes, actions_dict, gt_path, features_path_FLOW, sample_rate)
    # batch_gen_FLOW.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path_RGB, features_path_FLOW, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)


    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    for epochs in range(num_epochs):
        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path_RGB, features_path_FLOW, sample_rate)
        batch_gen_tst.read_data(vid_list_file_tst)
        trainer.predict(model_dir, results_dir, features_path_RGB, features_path_FLOW, batch_gen_tst, epochs+80, actions_dict, sample_rate)
        eval()

