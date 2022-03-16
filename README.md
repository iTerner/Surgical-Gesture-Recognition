# Surgical Gesture Recognition Using Multi-Encoder Based Architecture

## Table of contents

- [General info](#general-info)
- [Background](#Background)
- [Repository Description](#repository-description)
- [Architecture](#Architecture)
- [Requirement](#Requirement)
- [Notes](#Notes)

## General info

In this project, we propose to use a multi encoder-single decoder architecture based on LSTM to solve the gesture recognition task, using combined kinematic and video input data.

## Background

Gesture recognition is a type of perceptual computing user interface that allows computers to capture and interpret human gestures as commands. The general definition of gesture recognition is the ability of a computer to understand gestures and execute commands based on those gestures.
Automatically recognizing surgical gestures is a crucial step towards a thorough understanding of the surgical skill. Possible areas of application include automatic skill assessment, intra-operative monitoring of critical surgical steps, and semi-automation of surgical tasks.

Solutions that rely only on raw video and do not require additional sensor hardware are especially attractive as they can be implemented at a low cost in many scenarios. However, surgical gesture recognition based only on video is a challenging problem that requires effective means to extract both visual and temporal information from the video.

## Repository Description

| Filename                         | description                                                          |
| -------------------------------- | -------------------------------------------------------------------- |
| `analysis.py`                    | Python file consists of the analysis.                                |
| `batch_gen.py`                   | Python file consists of the code to transform the data into batches. |
| `metrics.py`                     | Python file consists of the metrics compution used in the project.   |
| `preprocess.py`                  | Python file consists of the preprocessing phase.                     |
| `model.py `                      | Python file consists of the implementation of the model.             |
| `train_experiment.py `           | The main file.                                                       |
| `Trainer.py`                     | Python file consists of the Trainer.                                 |
| `visualization.py`               | Python file consists of the visualization tools.                     |
| `figures `                       | Folder consists of all the images from the project.                  |
| `models`                         | Folder consists of all the models and optimizers for each split.     |
| `Gesture Recognition report.pdf` | The report.                                                          |
| `config.yaml`                    | The configuration file.                                              |
| `requirement.txt`                | File containing all the packages we used in this project.            |

## Architecture

In this project, we wanted to leverage the sequence dependency that appears in videos, in addition to the multi types of data we had (raw frames and kinematics data).

<p align="center">
  <img src=".\figures\surgical_data_science_model.png" width="350" alt="accessibility text">
</p>

## Requirement

To set up the environment and install the dependencies, run the following commands:

- conda create --name venv
- conda activate venv
- conda install pip
- pip install -r requirements.txt

To prepare the data, go to directory containing the code, and run:

- `python preprocess.py`

To run the experiment, from the directory containing the code, run:

- `python train_experiment.py -c config.yaml`

If a change of parameters is required, change inside config.yaml.

## Notes

- The APAS dataset was given by the TA.
- Most of the code was adapted from the course TA.
