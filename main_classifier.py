from trainer import Trainer
import logging
import os 
import torch
import gc
from utils import EmoDataset

new_model_dir = 'output_emotclass/logs'
os.makedirs(new_model_dir, exist_ok=True)  # Create the directory if it doesn't exist

# remove any existing handlers from the root loger
[logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]

# configure logging to write to a file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        # output log to a file
        logging.FileHandler(os.path.join(new_model_dir, 'trainlogs.log'))
        ]
    )

if __name__ == "__main__":    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU index (0, 1, 2, etc.) or device UUID
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('Using GPU: {}'.format(torch.cuda.get_device_name(device)))
        # print('Using GPU: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        logging.info('Using CPU')
        # print('Using CPU')
    
    # rubbish collection
    gc.collect()
    torch.cuda.empty_cache()
    
    #we define the paths for train, val, test, same split as already obtained in the T5 notebook
    train_path = "emotion_data/train.txt"
    test_path = "emotion_data/test.txt"
    val_path = "emotion_data/val.txt"

    # train the model
    trainer = Trainer(batch_size=20, epochs=10, train_path=train_path, val_path=val_path, test_path=test_path, location='emotion_model/RoBERTa_emotion_2ft.pt')
    trainer.fit()
    
    # evaluate the model
    trainer.evaluate()
    
    # rubbish collection
    gc.collect()
    torch.cuda.empty_cache()
    
    #we define the paths for train, val, test
    train_path = "emotion_data/empathetic_data/my_train.txt"
    test_path = "emotion_data/empathetic_data/my_test.txt"
    val_path = "emotion_data/empathetic_data/my_val.txt"
    
        # train the model
    trainer_second_ft = Trainer(batch_size=20, epochs=10, second_ft=True, train_path=train_path, val_path=val_path, test_path=test_path, location='emotion_model/RoBERTa_emotion_2ft_2.pt')
    trainer_second_ft.model.load_state_dict(torch.load('saved_models/best_model/best_model_first_ft.pt'))

    trainer_second_ft.fit()
    
    # evaluate the model
    trainer_second_ft.evaluate()
    
    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]