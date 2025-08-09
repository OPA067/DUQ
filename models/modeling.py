import os
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from types import SimpleNamespace
import torch
from torch import nn

from .cluster import PCM, Att_Block_Patch
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KLdivergence, KL
from .module_CAttention import CAM
from .uncertainty import UM
from .module_agg import module_agg
from .module_prob import prob_embed


allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int), )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()

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

        self.loss_fct = CrossEn(config)
        self.loss_kl_divergence = KLdivergence(config)
        self.loss_kl = KL(config)

        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        # init params
        sr_p = [0.5, 0.5, 0.5]
        embed_dim = state_dict["text_projection"].shape[1]
        self.alpha, self.beta = self.config.alpha, self.config.beta

        ##### feat-level #####
        # self.cross_atte = CAM(embed_dim=embed_dim, dropout=0.1)
        self.s_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.w_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.f_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.p_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.v_pcm_p_1 = PCM(sample_ratio=sr_p[0], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_1 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_2 = PCM(sample_ratio=sr_p[1], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_2 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_3 = PCM(sample_ratio=sr_p[2], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_3 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.feat_sims_u = UM(type='relu')

        ##### prob-level #####
        self.sample_text_n, self.sample_video_n = 10, 10
        self.agg_text   = module_agg(num_heads=8, embed_dim=embed_dim)
        self.agg_video  = module_agg(num_heads=8, embed_dim=embed_dim)
        self.prob_text  = prob_embed(num_heads=8, embed_dim=embed_dim)
        self.prob_video = prob_embed(num_heads=8, embed_dim=embed_dim)
        self.t_prob_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1),)
        self.v_prob_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1),)
        self.prob_sims_u = UM(type='relu')

        ## ===> Initialization trick [HARD CODE]
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
        self.load_state_dict(new_state_dict, strict=False)

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

        s_feat, w_feat, f_feat, p_feat = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)
        w_mask, f_mask = text_mask, video_mask

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                s_feat = allgather(s_feat, self.config)
                w_feat = allgather(w_feat, self.config)
                w_mask = allgather(w_mask, self.config)
                f_feat = allgather(f_feat, self.config)
                p_feat = allgather(p_feat, self.config)
                f_mask = allgather(f_mask, self.config)
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            logit_scale = self.clip.logit_scale.exp()

            ########## Step-I: feat-level ##########
            sims_sf = self.s_and_f(s_feat, f_feat)
            loss_sims_sf = (self.loss_fct(sims_sf * logit_scale) + self.loss_fct(sims_sf.T * logit_scale)) / 2.0
            loss_sims_sf_u, _ = self.feat_sims_u(sims_sf)
            p_feat = self.get_less_patch_feat(p_feat)
            sims_wp = self.w_and_p(w_feat, p_feat)
            loss_sims_wp = (self.loss_fct(sims_wp * logit_scale) + self.loss_fct(sims_wp.T * logit_scale)) / 2.0
            loss_sims_wp_u, _ = self.feat_sims_u(sims_wp)

            ########## Step-II: prob-level ##########
            t_feat = torch.cat([s_feat.unsqueeze(1), w_feat], dim=1) # [a, 1, d] + [a, w, d]
            t_prob = self.create_text_prob(t_feat)
            t_prob_mu, t_prob_sigma, t_probs = t_prob['mu'], t_prob['sigma'], t_prob['probs']
            v_feat = torch.cat([f_feat, p_feat], dim=1) # [b, f, d] + [b, p, d]
            v_prob = self.create_video_prob(v_feat)
            v_prob_mu, v_prob_sigma, v_probs = v_prob['mu'], v_prob['sigma'], v_prob['probs']
            sims_tv = self.t_and_v(t_probs, v_probs)
            loss_sims_tv = (self.loss_fct(sims_tv * logit_scale) + self.loss_fct(sims_tv.T * logit_scale)) / 2.0
            loss_sims_tv_u, _ = self.prob_sims_u(sims_tv)

            ########## Step-III: total-loss ##########
            loss_tv_kl = self.loss_kl_divergence(v_probs, v_prob_sigma, t_probs, t_prob_sigma)
            loss_kl = (self.loss_kl(sims_sf, sims_wp) + self.loss_kl(sims_sf.T, sims_wp.T) +
                       self.loss_kl(sims_sf, sims_tv) + self.loss_kl(sims_sf.T, sims_tv.T) +
                       self.loss_kl(sims_wp, sims_tv) + self.loss_kl(sims_wp.T, sims_tv.T)) / 6.0
            total_loss = loss_sims_sf + loss_sims_wp + loss_sims_tv \
                       + self.alpha * (loss_sims_sf_u + loss_sims_wp_u + loss_sims_tv_u) \
                       + self.beta * loss_tv_kl \
                       + loss_kl

<<<<<<< HEAD
            text_prob_embeds = text_prob_embeds / text_prob_embeds.norm(dim=-1, keepdim=True)
            video_prob_embeds = video_prob_embeds / video_prob_embeds.norm(dim=-1, keepdim=True)
            prob_sims = torch.einsum('amd,bnd->abmn', [text_prob_embeds, video_prob_embeds])

            t_prob_w = self.t_prob_w(text_prob_embeds).squeeze(-1)
            v_prob_w = self.v_prob_w(video_prob_embeds).squeeze(-1)

            t2v_prob_sims, _ = prob_sims.max(dim=-1)
            t2v_prob_sims = torch.einsum('abm,am->ab', [t2v_prob_sims, t_prob_w])
            v2t_prob_sims, _ = prob_sims.max(dim=-2)
            v2t_prob_sims = torch.einsum('abn,bn->ab', [v2t_prob_sims, v_prob_w])
            prob_sims = (t2v_prob_sims + v2t_prob_sims) / 2.0

            prob_sims_loss = self.loss_fct(prob_sims * logit_scale) + self.loss_fct(prob_sims.T * logit_scale)
            prob_sims_u_loss, prob_sims_u = self.uncertainty(prob_sims)

            kl_loss = self.loss_kl(video_prob_embeds, video_prob_sigma, text_prob_embeds, text_prob_sigma)

            loss = loss + feat_sims_loss + prob_sims_loss + self.alpha * (feat_sims_u_loss + prob_sims_u_loss) + self.beta * kl_loss

            return loss
=======
            return total_loss
>>>>>>> 395f4a2 (first commit)
        else:
            return None

    def sample_gaussian(self, mu, sigma, nums):
        eps = torch.randn(mu.size(0), nums, mu.size(1), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(sigma.unsqueeze(1))).add_(mu.unsqueeze(1))
        return samples

    def create_text_prob(self, t_feat):
        out = self.agg_text(t_feat)
        out = self.prob_text(t_feat, out)
        mu, sigma = out['mu'], out['sigma']
        probs = self.sample_gaussian(mu, sigma, self.sample_text_n)
        output = {'mu': mu, 'sigma': sigma, 'probs': probs}
        return output

    def create_video_prob(self, feat):
        out = self.agg_video(feat)
        out = self.prob_video(feat, out)
        mu, sigma = out['mu'], out['sigma']
        probs = self.sample_gaussian(mu, sigma, self.sample_video_n)
        output = {'mu': mu, 'sigma': sigma, 'probs': probs}
        return output

    def norm(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def agg_f_feat(self, f_feat, f_mask, agg_module):
        f_feat = f_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            f_feat_original = f_feat
            f_feat = pack_padded_sequence(f_feat, torch.sum(f_mask, dim=-1).cpu(), batch_first=True, enforce_sorted=False)
            f_feat, _ = self.lstm_visual(f_feat)
            if self.training:
                self.lstm_visual.flatten_parameters()
            f_feat, _ = pad_packed_sequence(f_feat, batch_first=True)
            f_feat = torch.cat((f_feat, f_feat_original[:, f_feat.size(1):, ...].contiguous()), dim=1)
            f_feat = f_feat + f_feat_original
        elif agg_module == "seqTransf":
            f_feat_original = f_feat
            seq_length = f_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=f_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(f_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            f_feat = f_feat + frame_position_embeddings
            extended_f_mask = (1.0 - f_mask.unsqueeze(1)) * -1000000.0
            extended_f_mask = extended_f_mask.expand(-1, f_mask.size(1), -1)
            f_feat = f_feat.permute(1, 0, 2)
            f_feat = self.transformerClip(f_feat, extended_f_mask)
            f_feat = f_feat.permute(1, 0, 2)
            f_feat = f_feat + f_feat_original

        return f_feat

    def s_and_f(self, s_feat, f_feat):
        f_w = torch.softmax(self.f_feat_w(f_feat).squeeze(-1), dim=-1)
        sims_sf = torch.einsum("ad,bfd->abf", [self.norm(s_feat), self.norm(f_feat)])
        sims_sf = torch.einsum("abf,bf->ab", [sims_sf, f_w])
        return sims_sf

    def w_and_p(self, w_feat, p_feat):
        w_w = torch.softmax(self.w_feat_w(w_feat).squeeze(-1), dim=-1)
        p_w = torch.softmax(self.p_feat_w(p_feat).squeeze(-1), dim=-1)
        sims_wp = torch.einsum("awd,bpd->abwp", [self.norm(w_feat), self.norm(p_feat)])
        sims_w2p, _ = sims_wp.max(dim=-1)
        sims_w2p = torch.einsum('abw,aw->ab', [sims_w2p, w_w])
        sims_p2w, _ = sims_wp.max(dim=-2)
        sims_p2w = torch.einsum('abf,bf->ab', [sims_p2w, p_w])
        sims_wp = (sims_w2p + sims_p2w) / 2.0
        return sims_wp

    def t_and_v(self, t_probs, v_probs):
        sims_tv = torch.einsum('amd,bnd->abmn', [self.norm(t_probs), self.norm(v_probs)])
        t_w = torch.softmax(self.t_prob_w(t_probs).squeeze(-1), dim=-1)
        v_w = torch.softmax(self.v_prob_w(v_probs).squeeze(-1), dim=-1)
        sims_t2v, _ = sims_tv.max(dim=-1)
        sims_t2v = torch.einsum('abm,am->ab', [sims_t2v, t_w])
        sims_v2t, _ = sims_tv.max(dim=-2)
        sims_v2t = torch.einsum('abn,bn->ab', [sims_v2t, v_w])
        sims_tv = (sims_t2v + sims_v2t) / 2.0
        return sims_tv

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float().view(bs_pair, s_feat.size(-1))
        w_feat = w_feat.float().view(bs_pair, -1, w_feat.size(-1))
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
        f_feat = f_feat.float().view(bs_pair, -1, f_feat.size(-1))
        f_feat = self.agg_f_feat(f_feat, video_mask, self.agg_module)
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

        return s_feat, w_feat, f_feat, p_feat

    def get_less_patch_feat(self, p_feat):
        p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
        p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
        p_mask = p_feat.new_ones(p_feat.size(0), p_feat.size(1))
        p_token_dict = {'x': p_feat,
                        'token_num': p_feat.size(1),
                        'idx_token': p_idx_token,
                        'agg_weight': p_agg_weight,
                        'mask': p_mask.detach()}
        p_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(p_token_dict))
        p_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(p_token_dict))
        p_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(p_token_dict))
        p_feat = p_token_dict['x']

        return p_feat

    def get_similarity_logits(self, t_mask, s_feat, w_feat, v_mask, f_feat, p_feat, shaped=False):

        ########## Step-I: feat-level ##########
        sims_sf = self.s_and_f(s_feat, f_feat)
        p_feat = self.get_less_patch_feat(p_feat)
        sims_wp = self.w_and_p(w_feat, p_feat)

        ########## Step-II: prob-level ##########
        t_feat = torch.cat([s_feat.unsqueeze(1), w_feat], dim=1)  # [a, 1, d] + [a, w, d]
        t_prob = self.create_text_prob(t_feat)
        t_prob_mu, t_prob_sigma, t_probs = t_prob['mu'], t_prob['sigma'], t_prob['probs']
        v_feat = torch.cat([f_feat, p_feat], dim=1)  # [b, f, d] + [b, p, d]
        v_prob = self.create_video_prob(v_feat)
        v_prob_mu, v_prob_sigma, v_probs = v_prob['mu'], v_prob['sigma'], v_prob['probs']
        sims_tv = self.t_and_v(t_probs, v_probs)

        sims = (sims_sf + sims_wp + sims_tv) / 3.0

        return sims

<<<<<<< HEAD
        t2v_prob_sims, _ = prob_sims.max(dim=-1)
        t2v_prob_sims = torch.einsum('abm,am->ab', [t2v_prob_sims, t_prob_w])
        v2t_prob_sims, _ = prob_sims.max(dim=-2)
        v2t_prob_sims = torch.einsum('abn,bn->ab', [v2t_prob_sims, v_prob_w])
        prob_sims = (t2v_prob_sims + v2t_prob_sims) / 2.0
=======
    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples
            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype
>>>>>>> 395f4a2 (first commit)

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
