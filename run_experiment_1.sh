#!bin/bash

## To execute the Experiment 1, it's need to execute different test. To reproduce the results in papers, it's 
# only need execute the following. The previous research, and conclusions extracted before can be obtained
# modifying the functions parameters. 


# Run Experiment 1 - Part 1: Federated Part

bash run_experiment_1_federated_approach.sh


# Run Experiment 1 - Part 2: Centralized Model

bash run_experiment_1_centralized_approach.sh


# Run Experiment 1 - Part 3: Individuals Models

bash run_experiment_1_individual_approach.sh


# Extract Scores and Results

pathScores=scores/experiment_1/
pathResume=resume_experiment_1.txt

for model in Autoencoder_1_16 Autoencoder_2_16_8 KNN Variational_Autoencoder_1_16 Variational_Autoencoder_2_16_8; do
	for level in SecuriteLevel_0 SecuriteLevel_1 SecuriteLevel_2; do 

		pathModelScore=${pathScores}$level/$model*

		pathModelScore=$pathModelScore/resume_statistics.txt
		echo "${model} ${level}">> $pathResume
		cat $pathModelScore | tail -n 1 >> $pathResume
		echo " ">> $pathResume

	done
	echo " ">> $pathResume
done