import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.all_config import gen_log
from config.base_config import Config
from collections import defaultdict, deque

from modules.metrics import np_softmax, AverageMeter, accuracy
from trainer.base_trainer import BaseTrainer

class Trainer(BaseTrainer):

    def __init__(self, model, loss, optimizer, config: Config, train_data_loader,
                 test_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best = -1.0

        self.test_batch_size = config.batch_size

        self.split_batch = config.split

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        start_time = time.time()
        if epoch == 1:
            _, Rsum = self._valid_epoch_step(epoch, 0, num_steps - 1)
            msg = (" Zero-Shot of Current Text-Video R@sum is {}".format(Rsum))
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

        for batch_idx, data in enumerate(self.train_data_loader):
            loss_tv = 0.
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            data['video'] = data['video'].to(self.device)
            data['label'] = data['label'].to(self.device)

            output_step, ce_loss, edl_loss = self.model(data, is_train=True)
            loss_tv = loss_tv + self.loss(output_step, self.model.clip.logit_scale)

            loss_all = loss_tv + ce_loss + edl_loss

            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            cost_time = time.time() - start_time
            start_time = time.time()

            eta_time = (len(self.train_data_loader) * self.config.num_epochs - batch_idx * epoch) * cost_time
            eta_time = f"{int(eta_time // 3600):02}:{int((eta_time % 3600) // 60):02}:{int(eta_time % 60):02}"

            if batch_idx % self.log_step == 0:
                msg = (
                'Train epoch: {} dl:{}/{} total_loss:{:.10f}, eta_time:{}'.format(
                    epoch,
                    batch_idx,
                    num_steps - 1,
                    loss_all.detach().item(),
                    eta_time
                ))
                gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            if batch_idx in eval_steps:

                if self.config.skip_eval:
                    msg = '\nSkip eval due to long time usage!\n'
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

                else:
                    test_res, Rsum = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                    self.model.train()

                    if Rsum > self.best:
                        self.best = Rsum
                        self._save_checkpoint(epoch, save_best=True)

                    msg = (" Current Best Text-Video R@sum is {}".format(self.best))
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
                    gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

        res = {
            'loss_train': total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()

        top1 = AverageMeter()
        top5 = AverageMeter()
        top10 = AverageMeter()

        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.test_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)
                data['label'] = data['label'].to(self.device)

                vqa_logits = self.model(data, is_train=False)

                prec1, prec5, prec10 = accuracy(vqa_logits, data['label'], topk=(1, 5, 10))
                top1.update(prec1[0], data['video'].size(0))
                top5.update(prec5[0], data['video'].size(0))
                top10.update(prec10[0], data['video'].size(0))

            msg = ('Video QA:>>> Prec@1: {top1.avg:.3f} - Prec@5: {top5.avg:.3f} - Prec@10: {top10.avg:.3f}'.format(top1=top1, top5=top5, top10=top10))

            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            R1 = top1.avg
            Rsum = top1.avg + top5.avg + top10.avg

            return R1, Rsum
