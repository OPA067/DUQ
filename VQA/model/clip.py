import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from config.base_config import Config
from modules.cluster import CTM, TCBlock
from modules.metrics import sim_matrix_training
from modules.module_edl import EDL
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
        self.text_transformer = text_transformer(config)
        self.video_transformer = video_transformer(config)

        self.v_ctm0 = CTM(sample_ratio=0.75, embed_dim=512, dim_out=512, k=3)
        self.v_block0 = TCBlock(dim=512, num_heads=8)
        self.v_ctm1 = CTM(sample_ratio=0.5, embed_dim=512, dim_out=512, k=3)
        self.v_block1 = TCBlock(dim=512, num_heads=8)

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

        self.EDL = EDL(config)

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        label = data['label']

        _, text_feat = self.clip.get_text_features(**text_data)
        _, video_feat = self.clip.get_image_features(video_data)

        video_feat = video_feat.reshape(batch_size, self.config.num_frames, -1)
        if is_train:

            v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
            v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
            v_mask = torch.ones(video_feat.size(0), video_feat.size(1)).to(video_feat.device)
            v_token_dict = {'x': video_feat,
                            'token_num': video_feat.size(1),
                            'idx_token': v_idx_token,
                            'agg_weight': v_agg_weight,
                            'mask': v_mask.detach()}
            v_token_dict = self.v_block0(self.v_ctm0(v_token_dict), text_feat)
            video_feat_step1 = v_token_dict["x"]
            v_token_dict = self.v_block1(self.v_ctm1(v_token_dict), text_feat)
            video_feat_step2 = v_token_dict["x"]
            video_feat = torch.cat((video_feat, video_feat_step1, video_feat_step2), dim=1)
            video_pooled = self.video_transformer(text_feat, video_feat)
            output_step = sim_matrix_training(text_feat, video_pooled)

            text_feat = self.t_proj(text_feat)
            video_pooled = self.v_proj(video_pooled)
            input = torch.cat((text_feat, video_pooled), dim=-1)
            pooled_output = self.dropout(input)
            vqa_logits = self.classifier(pooled_output)
            ce_loss = self.calc_loss(vqa_logits, label)

            edl_loss = self.EDL(vqa_logits, label, num_classes=self.config.num_labels)

            return output_step, ce_loss, edl_loss
        else:

            v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
            v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
            v_mask = torch.ones(video_feat.size(0), video_feat.size(1)).to(video_feat.device)
            v_token_dict = {'x': video_feat,
                            'token_num': video_feat.size(1),
                            'idx_token': v_idx_token,
                            'agg_weight': v_agg_weight,
                            'mask': v_mask.detach()}
            v_token_dict = self.v_block0(self.v_ctm0(v_token_dict), text_feat)
            video_feat_step1 = v_token_dict["x"]
            v_token_dict = self.v_block1(self.v_ctm1(v_token_dict), text_feat)
            video_feat_step2 = v_token_dict["x"]
            video_feat = torch.cat((video_feat, video_feat_step1, video_feat_step2), dim=1)
            video_pooled = self.video_transformer(text_feat, video_feat)

            text_feat = self.t_proj(text_feat)
            video_pooled = self.v_proj(video_pooled)
            input = torch.cat((text_feat, video_pooled), dim=-1)
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

