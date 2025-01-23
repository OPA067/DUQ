import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from VQA.modules.module_edl import EDL
from config.base_config import Config
from modules.metrics import sim_matrix_training
from modules.video_transfomer import video_transformer

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config

        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        config.pooling_type = 'transformer'
        self.pooling_type = 'transformer'
        self.video_transformer = video_transformer(config)

        embed_dim = self.config.embed_dim
        self.t_proj = nn.Linear(embed_dim, 4 * embed_dim)
        self.v_proj = nn.Linear(embed_dim, 4 * embed_dim)
        self.dropout = nn.Dropout(0.1)
        hidden_size = embed_dim * 8
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, self.config.num_labels)
        )

        self.VQA_EDL = EDL(config)

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        label = data['label']

        _, t_feat = self.clip.get_text_features(**text_data)
        _, v_feat = self.clip.get_image_features(video_data)

        v_feat = v_feat.reshape(batch_size, self.config.num_frames, -1)
        if is_train:
            v_feat = self.video_transformer(t_feat, v_feat)
            output = sim_matrix_training(t_feat, v_feat)

            t_feat = self.t_proj(t_feat)
            v_feat = self.v_proj(v_feat)
            input = torch.cat((t_feat, v_feat), dim=-1)
            pooled_output = self.dropout(input)
            vqa_logits = self.classifier(pooled_output)
            ce_loss = self.calc_loss(vqa_logits, label)

            edl_loss = self.VQA_EDL(vqa_logits, label, num_classes=self.config.num_labels)

            return output, ce_loss, edl_loss
        else:
            v_feat = self.video_transformer(t_feat, v_feat)
            t_feat = self.t_proj(t_feat)
            v_feat = self.v_proj(v_feat)
            input = torch.cat((t_feat, v_feat), dim=-1)
            pooled_output = self.dropout(input)
            vqa_logits = self.classifier(pooled_output)

            return vqa_logits

    def calc_loss(self, logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                logits.view(-1, self.config.num_labels),
                labels.view(-1))
        else:
            loss = 0
        return loss

