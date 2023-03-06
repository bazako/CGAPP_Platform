#!bin/bash



for percent in 0.0 0.08 0.05 0.02 0.01 0.005 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
	echo "Run for ${percent} and ${algorithm} "
	python code/exp2_backdoorAttack.py statistics Autoencoder_2_16_8 --typeCom Workers --algorithmAgregation mean --percent $percent
done


# Using median and trimmed
for percent in 0.5; do
	for algorithm in median trimmed_80 trimmed_60; do 

		echo "Run for ${percent} and ${algorithm} "
		python code/exp2_backdoorAttack.py statistics Autoencoder_2_16_8 --typeCom Workers --algorithmAgregation $algorithm --percent $percent

	done
done











