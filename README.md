# Behind the Scenes Of Sarcasm Detection Model

## Table of contents

- [General info](#general-info)
- [Background](#Background)
- [Repository Description](#repository-description)
- [Pipeline](#Pipeline)
- [Architecture](#Architecture)
- [Requirement](#Requirement)
- [References](#References)

## General info

In this project, we are willing to check whether a certain part of speech is affecting sarcasm in a given sentence.

## Background

Sarcasm is a means of communication that involves a hidden insult. Often, understanding sarcasm requires a high level of comprehension. We present a method for approximating the impact of specific part-of-speech on the decision-making process on sarcasm detection models. Our method is based on INLP\cite{INLP}, a method for removing information from neural representations. We create a map of words and their part-of-speech by tagging the part-of-speech of words in a sentence using a pre-trained model\cite{posmodel}. We repeat training linear classifiers that predict the part of speech from GLOVE embedding and project the embedding on the classifier's null space.

## Repository Description

| Filename                            | description                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `main.py`                           | The main file in Python format. To run the model and the Preprocessing you need to run this file. |
| `final_notebook.ipynb`              | Python notebook containing the training and evaluation, without the Preprocessing.                |
| `Preprocessing.py`                  | Python file consists of the prepossessing functions.                                              |
| `dataset.py`                        | Python file consists of the implementation of the dataset object.                                 |
| `models.py `                        | Python file consists of the implementation of the model.                                          |
| `utils.py `                         | Python file consists helper functions.                                                            |
| `helper_function.py`                | Python file consists of the helper functions for the prepossessing only.                          |
| `figures `                          | Folder consists of all the images from the project.                                               |
| `Sarcasm_Headlines_Dataset_v2.json` | The Sarcasm News Headlines Dataset.                                                               |
| `data`                              | Folder consists of all the data used in the projects.                                             |
| `Sarcasm_INLP.pdf`                  | The report.                                                                                       |
| `requirement.txt`                   | File containing all the packages we used in this project.                                         |

## Pipeline

<p align="center">
  <img src=".\figures\SarcasmINLP_method_diagram.png" width="350" alt="accessibility text">
</p>

## Architecture

In this project, we wanted to leverage the sequence dependency that appears in sentences, so, we used a bi-directional LSTM encoder followed by an MLP decoder, which gave us the sarcasm prediction.

<p align="center">
  <img src=".\figures\architecture.png" width="350" alt="accessibility text">
</p>

## Requirement

To run this project, you need to install several packages. For convenience, we created a `requirement.txt` file consists of all the packages used in this projcet.

In order to install all the packages in the `requirement.txt` file, simply use to command `pip install -r requirements.txt`.

To run the project, you need to open `main.py` and run `python main.py --pos NN`.

For other values you can pass different values through the arguments (such as num_epochs, learning_rate, batch_size, etc).

## References

- [Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection)
