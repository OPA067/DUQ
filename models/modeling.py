import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from .module_ca import frame_transformer
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .module_local import local_merge
from .module_prob import prob_embed, sample_gaussian_tensors
from .module_uncertainty import uncertainty_module
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KLdivergence

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int), nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class DUQ(nn.Module):
    def __init__(self, config):

        super(DUQ, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)

        new_state_dict = OrderedDict()
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        self.alpha, self.beta = self.config.alpha, self.config.beta
        self.embed_dim = state_dict["text_projection"].shape[1]

        self.frame_transformer = frame_transformer(embed_dim=self.embed_dim, dropout=0.3)

        self.uncertainty = uncertainty_module()

        self.n_text_samples, self.n_video_samples = self.config.n_text_samples, self.config.n_video_samples
        self.local_merge_text = local_merge(1, self.embed_dim, self.embed_dim, self.embed_dim * 2)
        self.local_merge_video = local_merge(1, self.embed_dim, self.embed_dim, self.embed_dim * 2)

        self.prob_text = prob_embed(self.embed_dim, self.embed_dim, self.embed_dim * 2)
        self.prob_video = prob_embed(self.embed_dim, self.embed_dim, self.embed_dim * 2)

        self.prob_t_w = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.prob_v_w = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        self.loss_fct = CrossEn(config)
        self.loss_kl = KLdivergence()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, text, text_mask, video, video_mask, idx=None, global_step=0):

        text_mask = text_mask.view(-1, text_mask.shape[-1])
        text = text.view(-1, text.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat, f_feat, _ = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                text_mask = allgather(text_mask, self.config)
                s_feat = allgather(s_feat, self.config)
                w_feat = allgather(w_feat, self.config)
                video_mask = allgather(video_mask, self.config)
                f_feat = allgather(f_feat, self.config)
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            # Step-I, feat-level
            v_feat = self.frame_transformer(s_feat, f_feat)
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
            feat_sims = self.sim_matrix_training(s_feat, v_feat, pooling_type="transformers")
            feat_sims_loss = self.loss_fct(feat_sims * logit_scale) + self.loss_fct(feat_sims.T * logit_scale)
            feat_sims_u_loss, feat_sims_u = self.uncertainty(feat_sims)

            # Step-II, prob-level
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            w_feat = w_feat / w_feat.norm(dim=-1, keepdim=True)
            prob_text = self.create_prob_text(s_feat, w_feat)
            prob_text_mu, prob_text_sigma = prob_text['mu'], prob_text['sigma']
            prob_text_embeds = prob_text['embeds']

            v_feat = v_feat.diagonal(dim1=0, dim2=1).transpose(0, 1)
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
            f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
            prob_video = self.create_prob_video(v_feat, f_feat)
            prob_video_mu, prob_video_sigma = prob_video['mu'], prob_video['sigma']
            prob_video_embeds = prob_video['embeds']

            prob_text_embeds = prob_text_embeds / prob_text_embeds.norm(dim=-1, keepdim=True)
            prob_video_embeds = prob_video_embeds / prob_video_embeds.norm(dim=-1, keepdim=True)
            prob_sims = torch.einsum('amd,bnd->abmn', [prob_text_embeds, prob_video_embeds])

            prob_t_w = self.prob_t_w(prob_text_embeds).squeeze(-1)
            prob_v_w = self.prob_v_w(prob_video_embeds).squeeze(-1)

            t2v_prob_sims, _ = prob_sims.max(dim=-1)
            t2v_prob_sims = torch.einsum('abm,bm->ab', [t2v_prob_sims, prob_t_w])
            v2t_prob_sims, _ = prob_sims.max(dim=-2)
            v2t_prob_sims = torch.einsum('abn,bn->ab', [v2t_prob_sims, prob_v_w])
            prob_sims = (t2v_prob_sims + v2t_prob_sims) / 2.0

            prob_sims_loss = self.loss_fct(prob_sims * logit_scale) + self.loss_fct(prob_sims.T * logit_scale)
            prob_sims_u_loss, prob_sims_u = self.uncertainty(prob_sims)

            kl_loss = self.loss_kl(prob_video_embeds, prob_video_sigma, prob_text_embeds, prob_text_sigma)

            loss = loss + feat_sims_loss + prob_sims_loss + self.alpha * (feat_sims_u_loss + prob_sims_u_loss) + self.beta * kl_loss

            return loss
        else:
            return None

    def create_prob_text(self, s_feat, w_feat):
        output = {}
        out = self.local_merge_text(s_feat, w_feat)
        uncertain_out = self.prob_text(s_feat, w_feat, out)

        output['mu'],  output['sigma'] = uncertain_out['mu'], uncertain_out['sigma']
        output['embeds'] = sample_gaussian_tensors(output['mu'], output['sigma'], self.n_text_samples)

        return output

    def create_prob_video(self, v_feat, f_feat):
        output = {}
        out = self.local_merge_video(v_feat, f_feat)
        uncertain_out = self.prob_video(v_feat, f_feat, out)

        output['mu'],  output['sigma'] = uncertain_out['mu'], uncertain_out['sigma']
        output['embeds'] = sample_gaussian_tensors(output['mu'], output['sigma'], self.n_video_samples)

        return output

    def sim_matrix_training(self, t_feat, v_feat, pooling_type):

        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        if pooling_type == 'avg':
            sims = torch.mm(t_feat, v_feat.t())
        else:
            t_feat = t_feat.unsqueeze(1)
            v_feat = v_feat.permute(1, 2, 0)
            sims = torch.bmm(t_feat, v_feat).squeeze(1)

        return sims

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float()
        s_feat = s_feat.view(bs_pair, -1, s_feat.size(-1))
        w_feat = w_feat.float()
        w_feat = w_feat.view(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True)
        f_feat = f_feat.float()
        p_feat = p_feat.float()
        f_feat = f_feat.float().view(bs_pair, -1, f_feat.size(-1))
        p_feat = p_feat.float().view(bs_pair, -1, p_feat.size(-1))

        return f_feat, p_feat

    def get_text_video_feat(self, text, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text = text.view(-1, text.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat = self.get_text_feat(text, text_mask, shaped=True)
        f_feat, p_feat = self.get_video_feat(video, video_mask, shaped=True)

        return s_feat.squeeze(1), w_feat, f_feat, p_feat

    def get_similarity_logits(self, text_mask, s_feat, w_feat, video_mask, f_feat, p_feat, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        logit_scale = self.clip.logit_scale.exp()
        loss = 0.

        # Step-I, feat-level
        v_feat = self.frame_transformer(s_feat, f_feat)
        s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        feat_sims = self.sim_matrix_training(s_feat, v_feat, pooling_type="transformers")
        feat_sims_loss = self.loss_fct(feat_sims * logit_scale) + self.loss_fct(feat_sims.T * logit_scale)
        feat_sims_u_loss, feat_sims_u = self.uncertainty(feat_sims)

        # Step-II, prob-level
        s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
        w_feat = w_feat / w_feat.norm(dim=-1, keepdim=True)
        prob_text = self.create_prob_text(s_feat, w_feat)
        prob_text_mu, prob_text_sigma = prob_text['mu'], prob_text['sigma']
        prob_text_embeds = prob_text['embeds']

        v_feat = v_feat.diagonal(dim1=0, dim2=1).transpose(0, 1)
        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
        prob_video = self.create_prob_video(v_feat, f_feat)
        prob_video_mu, prob_video_sigma = prob_video['mu'], prob_video['sigma']
        prob_video_embeds = prob_video['embeds']

        prob_text_embeds = prob_text_embeds / prob_text_embeds.norm(dim=-1, keepdim=True)
        prob_video_embeds = prob_video_embeds / prob_video_embeds.norm(dim=-1, keepdim=True)
        prob_sims = torch.einsum('amd,bnd->abmn', [prob_text_embeds, prob_video_embeds])

        prob_t_w = self.prob_t_w(prob_text_embeds).squeeze(-1)
        prob_v_w = self.prob_v_w(prob_video_embeds).squeeze(-1)

        t2v_prob_sims, _ = prob_sims.max(dim=-1)
        t2v_prob_sims = torch.einsum('abm,bm->ab', [t2v_prob_sims, prob_t_w])
        v2t_prob_sims, _ = prob_sims.max(dim=-2)
        v2t_prob_sims = torch.einsum('abn,bn->ab', [v2t_prob_sims, prob_v_w])
        prob_sims = (t2v_prob_sims + v2t_prob_sims) / 2.0

        prob_sims_loss = self.loss_fct(prob_sims * logit_scale) + self.loss_fct(prob_sims.T * logit_scale)
        prob_sims_u_loss, prob_sims_u = self.uncertainty(prob_sims)

        kl_loss = self.loss_kl(prob_video_embeds, prob_video_sigma, prob_text_embeds, prob_text_sigma)

        loss = loss + feat_sims_loss + prob_sims_loss + self.alpha * (
                    feat_sims_u_loss + prob_sims_u_loss) + self.beta * kl_loss

        sims = feat_sims + prob_sims
        return sims, loss
