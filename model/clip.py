import torch
import torch.nn as nn

from config.base_config import Config
from modules.distance_module import distance_module
from modules.inter_module import inter_module
from modules.intra_module import intra_module
from modules.local_feat_agg_module import lfa_module
from modules.probabilistic_embed import sample_gaussian_tensors, probabilistic_embed
from modules.loss import KLdivergence
from modules.metrics import sim_matrix_training
from modules.video_transfomer import video_transformer

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        self.n_text_samples = self.config.n_text_samples
        self.n_video_samples = self.config.n_video_samples

        """ self.alpha = self.config.alpha         ==>> 1e-1
            self.beta = self.config.beta           ==>> 1e-4
            self.embed_dim = self.config.embed_dim ==>> 512"""
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.embed_dim = self.config.embed_dim

        """Choose pretrained base model \in [ViT-B/32, ViT-B/16]"""
        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        config.pooling_type = 'transformer'
        """Video frame features v_f ==>> pooling video features
           Cross-Attention Module, or using Mean, MLP and Self-Attention Module."""
        self.video_transformer = video_transformer(config)
        """Intra-pair Similarity Uncertainty Module"""
        self.intra_module = intra_module(config)

        """Inter-pair Distance Uncertainty Module"""
        """Local Feature Aggregation"""
        self.lfa_text = lfa_module(1, self.embed_dim, self.embed_dim, self.embed_dim // 2)
        self.per_text = probabilistic_embed(self.embed_dim, self.embed_dim, self.embed_dim // 2)
        self.lfa_video = lfa_module(1, self.embed_dim, self.embed_dim, self.embed_dim // 2)
        self.per_video = probabilistic_embed(self.embed_dim, self.embed_dim, self.embed_dim // 2)
        self.distance_module = distance_module(config)
        self.inter_module = inter_module(config)
        """KL-divergence loss"""
        self.loss_kl = KLdivergence()

    def probabilistic_text(self, s_feat, w_feat):
        output = {}
        out = self.lfa_text(s_feat, w_feat)
        uncertain_out = self.per_text(s_feat, w_feat, out)

        output['mu'] = uncertain_out['mu']
        output['sigma'] = uncertain_out['sigma']
        output['embeddings'] = sample_gaussian_tensors(output['mu'], output['sigma'], self.n_text_samples)

        return output

    def probabilistic_video(self, v_feat, f_feat):
        output = {}
        out = self.lfa_video(v_feat, f_feat)
        uncertain_out = self.per_video(v_feat, f_feat, out)

        output['mu'] = uncertain_out['mu']
        output['sigma'] = uncertain_out['sigma']
        output['embeddings'] = sample_gaussian_tensors(output['mu'], output['sigma'], self.n_video_samples)

        return output

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        w_feat, s_feat = self.clip.get_text_features(**text_data)
        _, f_feat = self.clip.get_image_features(video_data)
        f_feat = f_feat.reshape(batch_size, self.config.num_frames, -1)

        if is_train:

            v_feat = self.video_transformer(s_feat, f_feat)
            sim_matrix = sim_matrix_training(s_feat, v_feat)
            intra_sim_loss, _ = self.intra_module(sim_matrix)

            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            w_feat = w_feat / w_feat.norm(dim=-1, keepdim=True)
            prob_text = self.probabilistic_text(s_feat, w_feat)
            prob_text_embedding = prob_text['embeddings']           # [B, K, D]
            prob_text_sigma = prob_text['sigma']                    # [B, D]

            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
            f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
            prob_video = self.probabilistic_video(v_feat, f_feat)
            prob_video_embedding = prob_video['embeddings']         # [B, K, D]
            prob_video_sigma = prob_video['sigma']                  # [B, D]

            kl_loss = self.loss_kl(prob_video_embedding, prob_video_sigma, prob_text_embedding, prob_text_sigma)

            # Try a self.n_text_samples != self.n_video_samples example
            dis_matrix = self.distance_module(prob_text_embedding, prob_video_embedding)

            inter_dis_loss, _ = self.inter_module(dis_matrix)

            return sim_matrix, dis_matrix, self.alpha * (intra_sim_loss + inter_dis_loss) + self.beta * kl_loss
        else:
            return s_feat, f_feat

    def get_similarity_logits(self, s_feat, f_feat):
        v_feat = self.video_transformer(s_feat, f_feat)
        sim_matrix = sim_matrix_training(s_feat, v_feat)
        intra_sim_loss, intra_sim_u = self.intra_module(sim_matrix)

        s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
        prob_text = self.probabilistic_text(s_feat, s_feat.unsqueeze(1))
        prob_text_embedding = prob_text['embeddings']

        """ s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            w_feat = w_feat / w_feat.norm(dim=-1, keepdim=True)
            prob_text = self.probabilistic_text(s_feat, w_feat)
            prob_text_embedding = prob_text['embeddings'] # \in [B, K, D] """

        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
        prob_video = self.probabilistic_video(v_feat, f_feat)
        prob_video_embedding = prob_video['embeddings']   # \in [B, K, D]

        dis_matrix = self.distance_module(prob_text_embedding, prob_video_embedding)
        inter_dis_loss, inter_dis_u = self.inter_module(dis_matrix)

        """ Maybe you can use more methods!!!
            sim_matrix = sim_matrix
            sim_matrix = (1 - dis_matrix) * sim_matrix """
        sim_matrix = ((1 - dis_matrix) * torch.exp(-0.1 * inter_dis_u)) * (torch.exp(-0.1 * intra_sim_u.T) * sim_matrix)

        return sim_matrix

