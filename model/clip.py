import torch
import torch.nn as nn

from config.base_config import Config
from modules.Distance_Module import Distance_Module
from modules.Inter_Modele import Inter_Modele
from modules.Intra_Module import Intra_Module
from modules.Local_Feat_Agg import LFA_Net
from modules.Probabilistic_Emb_Rep import sample_gaussian_tensors, PER_Net
from modules.loss import KLdivergence
from modules.metrics import sim_matrix_training
from modules.video_transfomer import video_transformer


class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        self.n_text_samples = self.config.n_text_samples
        self.n_video_samples = self.config.n_video_samples

        '''
            self.alpha = self.config.alpha ==>> 1e-1
            self.beta = self.config.beta   ==>> 1e-4
        '''
        self.alpha = self.config.alpha
        self.beta = self.config.beta

        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        config.pooling_type = 'transformer'
        self.video_transformer = video_transformer(config)
        self.Intra_Module = Intra_Module(config)

        embed_dim = self.config.embed_dim
        self.LFA_Net_text = LFA_Net(1, embed_dim, embed_dim, embed_dim // 2)
        self.PER_Net_text = PER_Net(embed_dim, embed_dim, embed_dim // 2)

        self.LFA_Net_video = LFA_Net(1, embed_dim, embed_dim, embed_dim // 2)
        self.PER_Net_video = PER_Net(embed_dim, embed_dim, embed_dim // 2)

        self.Distance_Module = Distance_Module(config)
        self.Inter_Module = Inter_Modele(config)
        self.loss_kl = KLdivergence()

    def probabilistic_text(self, sentence_feat, word_feat):
        output = {}
        out = self.LFA_Net_text(sentence_feat, word_feat)
        uncertain_out = self.PER_Net_text(sentence_feat, word_feat, out)

        output['mu'] = uncertain_out['mu']
        output['sigma'] = uncertain_out['sigma']
        output['embeddings'] = sample_gaussian_tensors(output['mu'], output['sigma'], self.n_text_samples)

        return output

    def probabilistic_video(self, video_feat, frame_feat):
        output = {}
        out = self.LFA_Net_video(video_feat, frame_feat)
        uncertain_out = self.PER_Net_video(video_feat, frame_feat, out)

        output['mu'] = uncertain_out['mu']
        output['sigma'] = uncertain_out['sigma']
        output['embeddings'] = sample_gaussian_tensors(output['mu'], output['sigma'], self.n_video_samples)

        return output

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        _, t_feat = self.clip.get_text_features(**text_data)
        _, v_feat = self.clip.get_image_features(video_data)
        v_feat = v_feat.reshape(batch_size, self.config.num_frames, -1)

        if is_train:

            v_pooled = self.video_transformer(t_feat, v_feat)
            sim_matrix = sim_matrix_training(t_feat, v_pooled)
            intra_sim_loss, _ = self.Intra_Module(sim_matrix)

            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            prob_text = self.probabilistic_text(t_feat, t_feat.unsqueeze(1))
            prob_text_embedding = prob_text['embeddings']           # [B, K, D]
            prob_text_sigma = prob_text['sigma']                    # [B, D]

            v_pooled = v_pooled / v_pooled.norm(dim=-1, keepdim=True)
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
            prob_video = self.probabilistic_video(v_pooled, v_feat)
            prob_video_embedding = prob_video['embeddings']         # [B, K, D]
            prob_video_sigma = prob_video['sigma']                  # [B, D]

            kl_loss = self.loss_kl(prob_video_embedding, prob_video_sigma, prob_text_embedding, prob_text_sigma)

            # Try a self.n_text_samples != self.n_video_samples example
            dis_matrix = self.Distance_Module(prob_text_embedding, prob_video_embedding)

            inter_dis_loss, _ = self.Inter_Module(dis_matrix)

            return sim_matrix, dis_matrix, self.alpha * (intra_sim_loss + inter_dis_loss) + self.beta * kl_loss
        else:
            return t_feat, v_feat

    def get_similarity_logits(self, t_feat, v_feat):
        v_pooled = self.video_transformer(t_feat, v_feat)
        sim_matrix = sim_matrix_training(t_feat, v_pooled)
        intra_sim_loss, intra_sim_u = self.Intra_Module(sim_matrix)

        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        prob_text = self.probabilistic_text(t_feat, t_feat.unsqueeze(1))
        prob_text_embedding = prob_text['embeddings']    # [B, K, D]

        v_pooled = v_pooled / v_pooled.norm(dim=-1, keepdim=True)
        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        prob_video = self.probabilistic_video(v_pooled, v_feat)
        prob_video_embedding = prob_video['embeddings']  # [B, K, D]

        dis_matrix = self.Distance_Module(prob_text_embedding, prob_video_embedding)

        inter_dis_loss, inter_dis_u = self.Inter_Module(dis_matrix)

        sim_matrix = ((1 - dis_matrix) * torch.exp(-0.1 * inter_dis_u)) * (torch.exp(-0.1 * intra_sim_u.T) * sim_matrix)
        # sim_matrix = (1 - dis_matrix) * sim_matrix
        # sim_matrix = sim_matrix

        return sim_matrix

