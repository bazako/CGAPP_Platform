#!bin/bash


python code/exp3_DenielService.py statistics Autoencoder_2_16_8 --algorithmAgregation mean --percent 0.0 --numClientDenial 0

for users in 1 2 3 4 5 6 ; do
	for percent in 0.08 0.05 0.02 0.01 0.005 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

		# echo "TEST " $var $mode
		python code/exp3_DenielService.py statistics Autoencoder_2_16_8 --algorithmAgregation mean --percent $percent --numClientDenial $users
	done
done



# for algorithm in median trimmed_80 trimmed_60 ; do 
# 	python code/exp3_DenielService.py statistics Autoencoder_2_16_8 --algorithmAgregation $algorithm --percent 0.0 --numClientDenial 0

# 	for users in 1 2 3 4 5 6 ; do
# 		for percent in 0.08 0.05 0.02 0.01 0.005 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

# 			# echo "TEST " $var $mode
# 			python code/exp3_DenielService.py statistics Autoencoder_2_16_8 --algorithmAgregation $algorithm --percent $percent --numClientDenial $users
# 		done
# 	done

# done









