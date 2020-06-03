# ASR-HMM-DNN
speech recognition based on deep neural network/hidden markov model  
This project use same data as ASR-SG-GMM-HMM.   
  
Data preparation:
1. Prepare the HMM trained with the ASR-SG-GMM-HMM project;   
2. Perform the GMM/HMM based Viterbi algorithm (made at the project 1) for the whole training data;  
3. Prepare unique HMM state IDs;  
4. Use this unique HMM state ID to convert the all state sequence obtained in the step 2;  
5. Perform the context expansion (3 left and 3 right context) for all feature vector sequences of the training data;  
6. Make a one big label vector and one big feature matrix by concatenating them for all utterances;  
7. Computer the HMM state prior distribution;  
  
DNN training:  
1. Set the DNN topologies;  
2. Perform the DNN training;  
3. Stop the training when the validation score starts degraded;  

Predict the most likely digit for each utterance by selecting the largest likelihood digit;  
Compute the accuracy (# of correct digits / # of test utterances * 100) by using whole training data.  

command:  
python submission.py --mode mlp train_1digit.feat test_1digit.feat

