# Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import torch
import os
import argparse
import pandas as pd
from datetime import datetime
from termcolor import colored
import random
import yaml
import wandb
from box import Box

from Trainer import Trainer
from batch_gen import BatchGenerator


if __name__ == "__main__":
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.yaml", type=str)

    parser.add_argument('--project', default=" project name ", type=str)
    parser.add_argument('--group', default=dt_string + " group ", type=str)
    parser.add_argument('--dataset', choices=['APAS'], default="APAS")
    parser.add_argument('--task', choices=['gestures'], default="gestures")

    parser.add_argument('--use_gpu_num', default="0", type=str)
    parser.add_argument('--upload', default=1, type=int)
    parser.add_argument('--debugging', default=0, type=int)

    parser.add_argument('--split', choices=['0', '1', '2', '3', '4', 'all'], default='all')
    parser.add_argument('--include_video', choices=[0, 1, 2], default=1, type=int)
    parser.add_argument('--encoder_type', choices=[1, 2], default=2, type=int)
    parser.add_argument('--decoder_type', choices=[2, 3], default=3, type=int)
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--lr', default=0.00316227766, type=float)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--normalization', choices=[0, 1], default=0, type=int)
    parser.add_argument('--optimizer', choices=[1, 2], default=1, type=int)
    parser.add_argument('--offline_mode', default=True, type=bool)

    parser.add_argument('--kin_encoder_input_dim', default=36, type=int)
    parser.add_argument('--kin_encoder_hidden_dim', default=64, type=int)
    parser.add_argument('--kin_encoder_num_layers', default=3, type=int)
    parser.add_argument('--vid_encoder_input_dim', default=36, type=int)
    parser.add_argument('--vid_encoder_hidden_dim', default=64, type=int)
    parser.add_argument('--vid_encoder_num_layers', default=3, type=int)

    parser.add_argument('--decoder_hidden_dim', default=128, type=int)
    parser.add_argument('--decoder_num_layers', default=3, type=int)
    args = parser.parse_args()

    # if a config file is given use the parameters defined there
    if args.config != '':
        with open(args.config, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        args = Box(args)
    else:
        delattr(args, 'config')

    debugging = args.debugging
    if debugging:
        args.upload = False
    sample_rate = 6  # downsample the frequency to 5Hz
    print(args)

    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use the full temporal resolution @ 30Hz

    list_of_splits = []
    if len(args.split) == 1:
        list_of_splits.append(int(args.split))
    elif args.dataset == "APAS":
        list_of_splits = list(range(0, 5))
    else:
        raise NotImplemented

    experiment_name = f"{dt_string} net: {args.encoder_type} {args.decoder_type} splits: {args.split}"
    args.group = experiment_name
    print(colored(experiment_name, "green"))

    summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
    if not debugging:
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

    full_eval_results = pd.DataFrame()
    full_train_results = pd.DataFrame()

    folds_folder = "/datashare/" + args.dataset + "/folds"
    features_path = "/datashare/" + args.dataset + "/kinematics_npy/"
    videos_path = "./video_tensors/"

    gt_path_gestures = "/datashare/" + args.dataset + "/transcriptions_gestures/"
    gt_path_tools_left = "/datashare/" + args.dataset + "/transcriptions_tools_left/"
    gt_path_tools_right = "/datashare/" + args.dataset + "/transcriptions_tools_right/"

    mapping_gestures_file = "/datashare/" + args.dataset + "/mapping_gestures.txt"
    mapping_tool_file = "/datashare/" + args.dataset + "/mapping_tools.txt"

    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict_gestures = dict()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])
    num_classes_tools = 0
    actions_dict_tools = dict()
    if args.dataset == "APAS":
        file_ptr = open(mapping_tool_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            actions_dict_tools[a.split()[1]] = int(a.split()[0])
        num_classes_tools = len(actions_dict_tools)

    num_classes_gestures = len(actions_dict_gestures)
    num_classes_list = [num_classes_gestures]

    backup_args = args.copy()
    for split_num in list_of_splits:
        print("split number: " + str(split_num))

        # CHANGE - moved the initiation of the wandb run to allow more comfortable way to work with sweeps
        if args.upload:
            wandb.init(project=args.project, group=args.group, name="split: " + str(split_num), reinit=True)
            args = backup_args.copy()
            delattr(args, 'split')
            wandb.config.update(args)
            args = wandb.config

        model_dir = "./models/" + args.dataset + "/" + experiment_name + "/split_" + str(split_num)
        if not debugging:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        trainer = Trainer(num_classes_list, device=device, **args)

        # CHANGE - added frames path to BatchGenerator for video features integration
        batch_gen = BatchGenerator(num_classes_gestures, num_classes_tools, actions_dict_gestures, actions_dict_tools,
                                   features_path, videos_path, split_num, folds_folder, gt_path_gestures, gt_path_tools_left,
                                   gt_path_tools_right, sample_rate=sample_rate, normalization=args.normalization,
                                   task=args.task, include_video=args.include_video)
        # CHANGE - added frames path to eval_dict for video features integration
        eval_dict = {"features_path": features_path, "videos_path": videos_path, "actions_dict_gestures": actions_dict_gestures,
                     "actions_dict_tools": actions_dict_tools, "device": device, "sample_rate": sample_rate,
                     "eval_rate": args.eval_rate, "gt_path_gestures": gt_path_gestures,
                     "gt_path_tools_left": gt_path_tools_left, "gt_path_tools_right": gt_path_tools_right,
                     "task": args.task, "include_video": args.include_video}
        eval_results, train_results = trainer.train(model_dir, batch_gen, num_epochs=args.num_epochs,
                                                    batch_size=args.batch_size, learning_rate=args.lr, optimizer=args.optimizer,
                                                    eval_dict=eval_dict, split=str(split_num), args=args)

        if not debugging:
            eval_results = pd.DataFrame(eval_results)
            train_results = pd.DataFrame(train_results)
            eval_results = eval_results.add_prefix('split_' + str(split_num) + '_')
            train_results = train_results.add_prefix('split_' + str(split_num) + '_')
            full_eval_results = pd.concat([full_eval_results, eval_results], axis=1)
            full_train_results = pd.concat([full_train_results, train_results], axis=1)
            full_eval_results.to_csv(summaries_dir + "/evaluation_results.csv", index=False)
            full_train_results.to_csv(summaries_dir + "/train_results.csv", index=False)
