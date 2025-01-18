import os
import warnings

import torch
import random
import numpy as np
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer
from config.all_config import gen_log

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

def main():
    # config
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    # GPU
    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # add log
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_test', msg=msg)
    msg = f'\nconfig={config.__dict__}\n'
    gen_log(model_path=config.model_path, log_name='log_test', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_test', msg='record all testing results')

    # seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("./openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # data I/O
    test_data_loader = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model,
                      loss=loss,
                      metrics=metrics,
                      optimizer=None,
                      config=config,
                      train_data_loader=None,
                      test_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    # path of model
    model_path = "./experiments/MSRVTT-train/2024_12_19_04_11_40/"
    checkpont_list = ["checkpoint-epoch1.pth", "checkpoint-epoch2.pth", "checkpoint-epoch3.pth", "checkpoint-epoch4.pth", "checkpoint-epoch5.pth", "model_best.pth"]
    for checkpont in checkpont_list:
        trainer.load_checkpoint(model_path + checkpont)
        trainer.validate()

if __name__ == '__main__':
    main()