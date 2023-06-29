# Emotion-classifier-RoBERTa-base
Code used to double-finetune a transformer model for a multi-class classification task of emotion classification, with 4 labels.  
The training should be performed on a GPU (Cuda).   
The base model used is RoBERTa base, within an Emotion Classifier base. 

# Dataset
The dataset is coming from Empathetic Personas and Emotion Dataset, used in previous work at Imperial College London. 

# Hyperparameters
Model: RoBERTa-base  
Classification Taks: Emotion 
Learning Rate: 1.5e-04 
Adam Eps: 1e-08   
Gradient Accumulation: 1   
Batch size: 20   
Epochs: 10  

# Credits
This model was finetuned using the previous findings of [Yisiang Ong](https://github.com/yisiang-ong) in 2022. 
