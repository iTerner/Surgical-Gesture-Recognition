import os
from PIL import Image
import torch
from tqdm import tqdm
from img2vec_pytorch import Img2Vec


if __name__ == "__main__":
    frames_path = "/datashare/APAS/frames/"
    out_path = "./video_tensors"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    sample_rate = 6

    videos_list = os.listdir(frames_path)
    # run over all videos
    for video_name in tqdm(videos_list):
        if video_name.endswith("side") or video_name.endswith("top"):
            print("Parsing video:", video_name)
            video_path = f"{frames_path}{video_name}/"
            num_frames = len(os.listdir(video_path))

            img_list = []  # holds all the tensor frames before the stacking
            # run over all frames in video and load according to sample rate
            for i in tqdm(range(0, num_frames, sample_rate)):
                img_name = f"img_{str(i + 1).zfill(5)}.jpg"
                img_path = f"{video_path}{img_name}"

                # load image in RGB mode (png files contains additional alpha channel)
                img_list.append(Image.open(img_path).convert('RGB'))

            # extract feature vectors for all frames using ResNet-34
            img2vec = Img2Vec(cuda=False, model='resnet34')
            video_tensor = img2vec.get_vec(img_list, tensor=True).squeeze()

            torch.save(video_tensor, f"{out_path}/{video_name}.pt")
