Authors:
Ido Terner 325132850
Eyal Finkelshtein 206123663

# To set up the environment and install the dependencies, run the following commands:
	conda create --name venv
	conda activate venv
	conda install pip
	pip install -r requirements.txt

# To prepare the data, go to directory containing the code, and run:
	python preprocess.py

# To run the experiment, from the directory containing the code, run:
	python train_experiment.py -c config.yaml

# If a change of parameters is required, change inside config.yaml.