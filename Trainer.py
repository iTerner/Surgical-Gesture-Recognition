# Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

import torch
from torch import optim
import torch.nn as nn
import math
import wandb
from datetime import datetime
import tqdm

from model import RecognitionModel
from metrics import *


class Trainer:
    def __init__(self, num_classes_list, device="cuda", debugging=False, **kwargs):
        """
        Initiates a Trainer object. CHANGE - the arguments passed to this method were changed according to our needs.
        :param num_classes_list: list of classes
        :param debugging: True for debugging mode, else False
        :param device: cpu or cuda
        :param kwargs: key-word arguments needed for the model.
        """
        self.model = RecognitionModel(device=device, **kwargs)
        self.debugging = debugging
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_classes_list = num_classes_list

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, optimizer, eval_dict, split, args):

        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + split)

        self.model.train()
        self.model.to(self.device)

        eval_rate = eval_dict["eval_rate"]
        # CHANGE - enabled a choice of optimizers (through yaml parameters or command line arguments)
        if optimizer == 1:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 2:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError()

        # CHANGE - recording the best epoch so far to avoid saving models in each epoch
        max_epoch = 0
        max_metric = 0
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0
            while batch_gen.has_next():
                # CHANGE - different outputs for different usage of videos (single video input or 2 video inputs)
                if args.include_video in [0, 1]:
                    batch_input, batch_videos, batch_target_gestures, mask = batch_gen.next_batch(batch_size)
                    batch_input, batch_videos, batch_target_gestures, mask = batch_input.to(self.device), batch_videos.to(self.device), \
                                                                             batch_target_gestures.to(self.device), mask.to(self.device)
                elif args.include_video == 2:
                    batch_input, batch_videos, batch_aux_videos, batch_target_gestures, mask = batch_gen.next_batch(batch_size)
                    batch_input, batch_videos, batch_aux_videos, batch_target_gestures, mask = batch_input.to(self.device),\
                                                                                               batch_videos.to(self.device),\
                                                                                               batch_aux_videos.to(self.device),\
                                                                                               batch_target_gestures.to(self.device),\
                                                                                               mask.to(self.device)
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                # CHANGE - mask is passed to model forward method (for usage with the Transformers). The lengths are
                # also calculated inside the model when LSTM is used.
                videos_input = [batch_videos, batch_aux_videos] if args.include_video == 2 else [batch_videos]
                predictions1 = self.model(batch_input, videos_input, mask[:, 0, :])
                predictions1 = (predictions1[0] * mask).unsqueeze_(0)
                loss = 0
                for p in predictions1:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[0]), batch_target_gestures.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted1 = torch.max(predictions1[-1].data, 1)
                for i in range(len(lengths)):
                    correct1 += (predicted1[i][:lengths[i]] == batch_target_gestures[i][:lengths[i]]).float().sum().item()
                    total1 += lengths[i]

                pbar.update(1)

            batch_gen.reset()
            pbar.close()

            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(colored(dt_string, 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          batch_gen.list_of_train_examples),
                                                                                                      float(correct1) / total1))
            train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                             "train acc": float(correct1) / total1}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload:
                    wandb.log(results)

                # CHANGE - moved the timing of model saving to allow saving the best performing model so far
                if not self.debugging:
                    # save the best model so far
                    if results["mean_metric"] > max_metric:
                        torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                        torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
                        # remove previously saved model
                        if os.path.exists(save_dir + "/epoch-" + str(max_epoch + 1) + ".model") and epoch > 0:
                            os.remove(save_dir + "/epoch-" + str(max_epoch + 1) + ".model")
                            os.remove(save_dir + "/epoch-" + str(max_epoch + 1) + ".opt")
                        # update the the max values
                        max_epoch = epoch
                        max_metric = results["mean_metric"]
        return eval_results_list, train_results_list

    def evaluate(self, eval_dict, batch_gen):
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        videos_path = eval_dict["videos_path"]  # CHANGE - added video path to eval_dict for video features integration
        include_video = eval_dict["include_video"]  # CHANGE - added include_video to eval_dict for video features integration
        sample_rate = eval_dict["sample_rate"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            list_of_vids = batch_gen.list_of_valid_examples
            recognition1_list = []

            for seq in list_of_vids:
                features = np.load(features_path + seq.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)

                if include_video in [1, 2]:
                    side_video_path = f"{videos_path}{seq.split('.')[0]}_side.pt"
                    side_video_tensor = torch.load(side_video_path)
                    side_video_tensor.unsqueeze_(0)
                    input_y = side_video_tensor.to(device)
                if include_video in [0, 2]:
                    top_video_path = f"{videos_path}{seq.split('.')[0]}_top.pt"
                    top_video_tensor = torch.load(top_video_path)
                    top_video_tensor.unsqueeze_(0)
                    input_z = top_video_tensor.to(device)

                # CHANGE - added number of frames to deal with inconsistencies between kinematic and video features
                num_frames = min(input_x.shape[2], input_y.shape[1] if include_video in [1, 2] else np.inf,
                                 input_z.shape[1] if include_video in [0, 2] else np.inf)

                # CHANGE - according to our change in the training loop, a mask should be passed to the model, instead
                # of the length. Also passing the video input.
                if include_video == 0:
                    videos_tensors = [input_z[:, :num_frames, :]]
                elif include_video == 1:
                    videos_tensors = [input_y[:, :num_frames, :]]
                elif include_video == 2:
                    videos_tensors = [input_y[:, :num_frames, :], input_z[:, :num_frames, :]]
                else:
                    raise NotImplementedError()
                predictions1 = self.model(input_x[:, :, :num_frames], videos_tensors, torch.ones((1, num_frames), device=device))
                predictions1 = predictions1[0].unsqueeze_(0)
                predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                _, predicted1 = torch.max(predictions1[-1].data, 1)
                predicted1 = predicted1.squeeze()

                recognition1 = []
                for i in range(len(predicted1)):
                    recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                        list(actions_dict_gesures.values()).index(predicted1[i].item())]] * sample_rate))
                recognition1_list.append(recognition1)

            print("gestures results")
            results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures, recognition_list=recognition1_list,
                                             list_of_videos=list_of_vids, suffix="gesture")
            results.update(results1)

            self.model.train()
            return results
