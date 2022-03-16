import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import os

def plot_seq_gestures(all_recogs, all_gt,task,best):
    if task == "gesture":
        cmap_ = "tab10"
        gest_to_int = {"G0": 0, "G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
        vmax_ = 6
        gesture_descriptions = ['no gesture', "needle passing", "pull the suture", "instrument tie", "lay the knot",
     "cut the suture"]

    else:
        cmap_ = 'Accent'
        gest_to_int = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
        vmax_ = 4
        if task == "left hand":
            gesture_descriptions = ["no tool in left hand","needle driver", "forceps","scissors"]
        if task == "right hand":
            gesture_descriptions = ["no tool in right hand","needle driver", "forceps","scissors"]


    max_acc = 0
    min_acc = 200
    min_r =[]
    max_r =[]
    min_gt =[]
    max_gt =[]

    for split_r,split_gt in zip(all_recogs, all_gt):
        for r,gt in zip(split_r,split_gt):
            acc = 100*(sum(r == gt)/len(r))
            if acc < min_acc:
                min_acc = acc
                min_r = r
                min_gt = gt
            if acc > max_acc:
                max_acc = acc
                max_r = r
                max_gt = gt

    names= ['TCN-LSTM',"Ground Truth"]
    gestures_list_min=[min_r,min_gt]
    gestures_list_max=[max_r,max_gt]
    if best:
        gestures_list = gestures_list_max
    else:
        gestures_list = gestures_list_min
    fig, axs = plt.subplots(len(gestures_list), 1, figsize=(14,  0.7*len(gestures_list)))

    fig.suptitle(f'{""}', fontsize=1)

    for i, gestures in enumerate(gestures_list):
        gestures_ = []

        if "numpy.ndarray" in str(type(gestures)):
            gestures = gestures.tolist()
        for gest in gestures:
            gestures_.append(gest_to_int[gest])
        gestures = np.array(gestures_)
        gestures = gestures[0:-1:25]
        map = np.tile(gestures, (100, 1))
        axs[i].axis('off')
        # axs[i].set_title(names[i], fontsize=10, pad= 10)
        axs[i].imshow(map,cmap=cmap_,aspect="auto",interpolation='nearest',vmax=vmax_)

    plt.tight_layout()
    plt.show()
    plot_legend(gesture_descriptions,num_classes =vmax_,cmap_=cmap_)





def plot_legend(gesture_descriptions:list, num_classes=6, fig_width=4.5, fig_height_per_seq=0.3,cmap_='tab10'):
    box_width = 25
    text_offset = 10
    x_lim = 500

    figsize = (fig_width, (num_classes * 1) * fig_height_per_seq)
    fig, axes = plt.subplots(nrows=num_classes, ncols=1, sharex=True, figsize=figsize,
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    gesture_labels = list(np.arange(0, num_classes))

    for label, description, ax in zip(gesture_labels, gesture_descriptions, axes):
        plt.sca(ax)
        plt.axis('off')

        x = np.arange(0, box_width)
        y = 0 * np.ones(box_width)
        colors = label * np.ones(box_width)
        ax.scatter(x, y, c=colors, marker='|', s=200, lw=2, vmin=0, vmax=num_classes, cmap=cmap_)
        ax.text(box_width + text_offset, 0.0, description, fontsize=10, va='center')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.xlim(0, x_lim*1.1)
        plt.ylim(-0.5, 0.5)

    plt.tight_layout()
    # plt.savefig(os.path.join(out_file,plot_name+"_legend.png"))

    plt.show()
    plt.close(fig)

# plot_legend("")
# plot_video_gestures("video name",[array_a,array_b],["predicted","ground truth"])
