# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

import torch
import os
import argparse
import random
import numpy as np
from metrics import pars_ground_truth
from visualization import plot_seq_gestures, plot_legend
from Trainer import Trainer
from datetime import datetime
from batch_gen import BatchGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.font_manager


def get_gt(ground_truth_path,list_of_videos):
    gt_list =[]
    for i, seq in enumerate(list_of_videos):

        file_ptr = open(ground_truth_path + seq.split('.')[0] + '.txt', 'r')
        gt_source = file_ptr.read().split('\n')[:-1]
        gt_content = pars_ground_truth(gt_source)

        gt_list.append(gt_content)
    return gt_list

def predict(trainer, model_dir, features_path, list_of_vids, epoch, actions_dict_gestures,actions_dict_tools, device, sample_rate,task,network):
    trainer.model.eval()
    with torch.no_grad():
        trainer.model.to(device)
        trainer.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        recognition1_list = []
        recognition2_list = []
        recognition3_list = []
        for seq in list_of_vids:

                features = np.load(features_path + seq.split('.')[0] + '.npy')
                if batch_gen.normalization == "Min-max":
                    numerator = features.T - batch_gen.min
                    denominator = batch_gen.max - batch_gen.min
                    features = (numerator / denominator).T
                elif batch_gen.normalization == "Standard":
                    numerator = features.T - batch_gen.mean
                    denominator = batch_gen.std
                    features = (numerator / denominator).T
                elif batch_gen.normalization == "samplewise_SD":
                    samplewise_meam = features.mean(axis=1)
                    samplewise_std = features.std(axis=1)
                    numerator = features.T - samplewise_meam
                    denominator = samplewise_std
                    features = (numerator / denominator).T

                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                if task == "multi-taks":
                    predictions1, predictions2, predictions3 = trainer.model(input_x)
                elif task == "tools":
                    if network == "LSTM" or network == "GRU":
                        predictions2, predictions3 = self.model(input_x, torch.tensor([features.shape[1]]))
                        predictions2 = predictions2.unsqueeze_(0)
                        predictions2 = torch.nn.Softmax(dim=2)(predictions2)
                        predictions3 = predictions3.unsqueeze_(0)
                        predictions3 = torch.nn.Softmax(dim=2)(predictions3)



                    else:
                        predictions2, predictions3 = trainer.model(input_x)
                else:
                    if network == "LSTM" or network == "GRU":
                        predictions1 = trainer.model(input_x, torch.tensor([features.shape[1]]))
                        predictions1 = predictions1[0].unsqueeze_(0)
                        predictions1 = torch.nn.Softmax(dim=2)(predictions1)
                    else:
                        predictions1 = trainer.model(input_x)[0]

                if task == "multi-taks" or task == "gestures":
                    _, predicted1 = torch.max(predictions1[-1].data, 1)
                    predicted1 = predicted1.squeeze()

                if task == "multi-taks" or task == "tools":
                    _, predicted2 = torch.max(predictions2[-1].data, 1)
                    _, predicted3 = torch.max(predictions3[-1].data, 1)
                    predicted2 = predicted2.squeeze()
                    predicted3 = predicted3.squeeze()

                recognition1 = []
                recognition2 = []
                recognition3 = []
                if task == "multi-taks" or task == "gestures":
                    for i in range(len(predicted1)):
                        recognition1 = np.concatenate((recognition1, [list(actions_dict_gestures.keys())[
                                                                          list(actions_dict_gestures.values()).index(
                                                                              predicted1[i].item())]] * sample_rate))
                    recognition1_list.append(recognition1)
                if task == "multi-taks" or task == "tools":

                    for i in range(len(predicted2)):
                        recognition2 = np.concatenate((recognition2, [list(actions_dict_tools.keys())[
                                                                          list(actions_dict_tools.values()).index(
                                                                              predicted2[i].item())]] * sample_rate))
                    recognition2_list.append(recognition2)

                    for i in range(len(predicted3)):
                        recognition3 = np.concatenate((recognition3, [list(actions_dict_tools.keys())[
                                                                          list(actions_dict_tools.values()).index(
                                                                              predicted3[i].item())]] * sample_rate))
                    recognition3_list.append(recognition3)

        return recognition1_list, recognition2_list, recognition3_list

def actions_list_to_ids(recognition_list,actions_dict):
    """

    :param recognition_list: list os lists of labels
    :param actions_dict: dicts from labels to action ids
    :return: list of lists of ids
    """
    output = []
    for video_labels in recognition_list:
        list_of_ids =[]
        for label in list(video_labels):
            list_of_ids.append(actions_dict[label])
        output.append(list_of_ids)
    return output

def prepare_for_visual_sammary(recognition_id_list,gt_id_list,list_of_vidios):
    merged_ids_list=[]
    name_list =[]
    for recog_list, gt_list, video_name in zip(recognition_id_list,gt_id_list,list_of_vidios):
        merged_ids_list.append(recog_list)
        name_list.append(video_name[:-4] + " predicted")
        merged_ids_list.append(gt_list)
        name_list.append(video_name[:-4] + " ground truth")
    return merged_ids_list, name_list

def conf_mat_calc(all_recogs,all_gts,labels):
    flatten_recogs=[]
    flatten_gt =[]
    for split in all_gts:
        for seq in split:
            flatten_gt += seq

    for split in all_recogs:
        for seq in split:
            flatten_recogs += seq.tolist()

    distribution = confusion_matrix(flatten_gt, flatten_recogs, labels=labels)
    distribution = distribution / np.sum(distribution)
    if "G0" in labels:
        ax = sns.heatmap(distribution, annot=True,
                         xticklabels=['no gesture', "needle passing", "pull the suture", "instrument tie", "lay the knot",
                                      "cut the suture"],
                         yticklabels=['no gesture', "needle passing", "pull the suture", "instrument tie", "lay the knot",
                                      "cut the suture"], fmt='.3f', cmap=sns.color_palette("mako"))
    else:
        ax = sns.heatmap(distribution, annot=True,
                         xticklabels=["no tool", "needle driver", "forceps",
                                      "scissors"],
                         yticklabels=["no tool", "needle driver", "forceps",
                                      "scissors"], fmt='.3f', cmap=sns.color_palette("mako"))


    plt.show()

    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', default="29.10.2021 20:53:46  task:multi-taks splits: all net: TCN-LSTM is Offline: True")

    parser.add_argument('--dataset', choices=['APAS'], default="APAS")
    parser.add_argument('--task', choices=['gestures', 'tools', 'multi-taks'], default="multi-taks")
    parser.add_argument('--network',
                        choices=['MS-TCN2', 'MS-TCN2_IUR', 'LSTM', 'GRU', 'MS-LSTM-TCN', 'MS-TCN-LSTM', 'MS-GRU-TCN', 'MS-TCN-GRU'],
                        default="MS-TCN-LSTM")
    parser.add_argument('--split', choices=['0', '1', '2', '3', '4', 'all'], default='2')
    # features_dim for jigwaw 14  APAS 36
    parser.add_argument('--features_dim', default='36', type=int)
    #[164,244,192,231,224]
    parser.add_argument('--list_of_num_epochs', default=[192], type=list)
    # Architectuyre
    parser.add_argument('--num_f_maps', default='64', type=int)

    parser.add_argument('--num_layers_PG', default=13, type=int)
    parser.add_argument('--num_layers_R', default=13, type=int)
    parser.add_argument('--filtered_data', default=True, type=bool)
    parser.add_argument('--normalization', choices=['Min-max', 'Standard', 'samplewise_SD', 'none'], default='Standard',
                        type=str)
    parser.add_argument('--num_R', default=3, type=int)
    parser.add_argument('--loss_tau', default=16, type=float)
    parser.add_argument('--loss_lambda', default=0.5, type=float)
    parser.add_argument('--offline_mode', default=True, type=bool)
    parser.add_argument('--project', default="Offline RNN nets Sensor paper Final", type=str)
    parser.add_argument('--use_gpu_num', default="0", type=str)

    args = parser.parse_args()
    print(args)
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use the full temporal resolution @ 30Hz
    if args.network in ["GRU", "LSTM", "MS-LSTM", "MS-GRU"]:
        sample_rate = 6
        bz = 1

    else:
        sample_rate = 1
        bz = 1

    list_of_splits = []
    if len(args.split) == 1:
        list_of_splits.append(int(args.split))

    elif args.dataset == "APAS":
        list_of_splits = list(range(0, 5))
    else:
        list_of_splits = list(range(0, 8))

    num_epoch_list = args.list_of_num_epochs
    assert len(num_epoch_list) == len(list_of_splits)
    for i,epoch in enumerate(num_epoch_list):
        num_epoch_list[i] = num_epoch_list[i] + 1

    features_dim = args.features_dim
    offline_mode = args.offline_mode
    num_layers_PG = args.num_layers_PG
    num_layers_R = args.num_layers_R
    num_R = args.num_R
    num_f_maps = args.num_f_maps
    experiment_name = args.experiment_name

    summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
    all_recogs1=[]
    all_recogs2=[]
    all_recogs3=[]
    all_gt1=[]
    all_gt2=[]
    all_gt3=[]


    for split_num in list_of_splits:
        if args.split == "all":
            num_epoch = num_epoch_list[split_num]
        else:
            num_epoch = num_epoch_list[0]

        print("split number: " + str(split_num))
        args.split = str(split_num)

        folds_folder = "./data/" + args.dataset + "/folds"
        if args.dataset == "APAS":
            if args.filtered_data:
                features_path = "./data/" + args.dataset + "/kinematics_with_filtration_npy/"
            else:
                features_path = "./data/" + args.dataset + "/kinematics_without_filtration_npy/"
        else:
            features_path = "./data/" + args.dataset + "/kinematics_npy/"

        gt_path_gestures = "./data/" + args.dataset + "/transcriptions_gestures/"
        gt_path_tools_left = "./data/" + args.dataset + "/transcriptions_tools_left/"
        gt_path_tools_right = "./data/" + args.dataset + "/transcriptions_tools_right/"

        mapping_gestures_file = "./data/" + args.dataset + "/mapping_gestures.txt"
        mapping_tool_file = "./data/" + args.dataset + "/mapping_tools.txt"

        model_dir = "./models/" + args.dataset + "/" + experiment_name + "/split_" + args.split

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

        if args.task == "gestures":
            num_classes_list = [num_classes_gestures]
        elif args.dataset == "APAS" and args.task == "tools":
            num_classes_list = [num_classes_tools, num_classes_tools]
        elif args.dataset == "APAS" and args.task == "multi-taks":
            num_classes_list = [num_classes_gestures, num_classes_tools, num_classes_tools]

        trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes_list,
                          offline_mode=offline_mode, tau=0, lambd=0, task=args.task, device=device,
                          network=args.network, debagging=True)

        batch_gen = BatchGenerator(num_classes_gestures, num_classes_tools, actions_dict_gestures, actions_dict_tools,
                                   features_path, split_num, folds_folder, gt_path_gestures, gt_path_tools_left,
                                   gt_path_tools_right, sample_rate=sample_rate, normalization=args.normalization,
                                   task=args.task)
        eval_dict = {"features_path": features_path, "actions_dict_gestures": actions_dict_gestures,
                     "actions_dict_tools": actions_dict_tools, "device": device, "sample_rate": sample_rate,
                     "eval_rate": 1,
                     "gt_path_gestures": gt_path_gestures, "gt_path_tools_left": gt_path_tools_left,
                     "gt_path_tools_right": gt_path_tools_right, "task": args.task}


        list_of_vids = batch_gen.list_of_valid_examples

        recognition1_list, recognition2_list, recognition3_list = predict(trainer, model_dir, features_path, list_of_vids, num_epoch, actions_dict_gestures,actions_dict_tools, device,
                        sample_rate,args.task,args.network)

        if args.task == "multi-taks" or args.task == "gestures":
            print("gestures results")
            gt_list_1 = get_gt(ground_truth_path=gt_path_gestures,
                                              list_of_videos=list_of_vids)
            for i in range(len(gt_list_1)):
                min_len = min(len(gt_list_1[i]),len(recognition1_list[i]))
                gt_list_1[i] = gt_list_1[i][:min_len]
                recognition1_list[i] = recognition1_list[i][:min_len]

            all_recogs1.append(recognition1_list)
            all_gt1.append(gt_list_1)



        if args.task == "multi-taks" or args.task == "tools":
            gt_list_2 = get_gt(ground_truth_path=gt_path_tools_right,list_of_videos=list_of_vids)

            for i in range(len(gt_list_2)):
                min_len = min(len(gt_list_2[i]),len(recognition2_list[i]))
                gt_list_2[i] = gt_list_2[i][:min_len]
                recognition2_list[i] = recognition2_list[i][:min_len]



            gt_list_3 = get_gt(ground_truth_path=gt_path_tools_left,
                                              list_of_videos=list_of_vids)

            for i in range(len(gt_list_3)):
                min_len = min(len(gt_list_3[i]),len(recognition3_list[i]))
                gt_list_3[i] = gt_list_3[i][:min_len]
                recognition3_list[i] = recognition3_list[i][:min_len]

            all_recogs2.append(recognition2_list)
            all_recogs3.append(recognition3_list)
            all_gt2.append(gt_list_2)
            all_gt3.append(gt_list_3)

    plot_seq_gestures(all_recogs1, all_gt1, "gesture", True)
    plot_seq_gestures(all_recogs2, all_gt2, "right hand", True)
    plot_seq_gestures(all_recogs3, all_gt3, "left hand", True)
    plot_seq_gestures(all_recogs1, all_gt1, "gesture", False)
    plot_seq_gestures(all_recogs2, all_gt2, "right hand", False)
    plot_seq_gestures(all_recogs3, all_gt3, "left hand", False)




    # conf1 = conf_mat_calc(all_recogs1,all_gt1,["G0","G1","G2","G3","G4","G5"])
    # conf2 = conf_mat_calc(all_recogs2,all_gt2,["T0","T1","T2","T3"])
    # conf3 = conf_mat_calc(all_recogs3,all_gt3,["T0","T1","T2","T3"])






