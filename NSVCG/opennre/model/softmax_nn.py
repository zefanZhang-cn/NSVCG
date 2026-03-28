import torch
from torch import nn, optim
from .base_model import SentenceRE
from torch.nn.functional import gelu,relu
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional
from torch import nn, Tensor
import copy
import math
from .attention import MultiheadAttention
# from transformer.Models import PositionalEncoding
from .Layers import EncoderLayer, DecoderLayer
import pdb
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DenoiseNetwork(nn.Module):

    def __init__(self, module_dim=768, module_id=0, layer_num=2, num_class=23,dropout=0.4):
        super().__init__()

        self.module_dim = module_dim
        self.num_class = num_class
        self.dropout=dropout

        # 时间编码器
        # self.time_embed = SinusoidalPositionEmbeddings(d_model) # diff_mom 正余弦编码
        # from DiT_module import TimestepEmbedder
        self.time_embed = TimestepEmbedder(self.module_dim)  # DiT 正余弦加mlp

        # 噪声编码器
        # 4000 --> d
        # self.noise_embed = MLP(num_class,d_model, d_model, 1) # diff_mom MLP
        self.noise_embed = nn.Linear(self.num_class, self.module_dim)  # DiT linear
        from .Layers import DecoderLayer_woSA, DecoderLayerPre_woSA, DecoderLayerPre_Enc, DecoderLayer_Enc, \
            DecoderLayer_context
        if module_id == 0:
            self.diff_decoders = nn.ModuleList(
                [DecoderLayerPre_woSA(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in
                 range(layer_num)])
            # self.diff_decoders1 = nn.ModuleList(
            #     [DecoderLayerPre_woSA(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in
            #      range(layer_num)])
        else:
            self.diff_decoders = nn.ModuleList(
                [DecoderLayerPre_Enc(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim,self.dropout) for i in
                 range(layer_num)])
            # self.diff_decoders1 = nn.ModuleList(
            #     [DecoderLayerPre_Enc(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim,
            #                          self.dropout) for i in
            #      range(layer_num)])
        self.ln = nn.LayerNorm(self.module_dim, eps=1e-6)
        self.gg=nn.Linear(768*2,768)

    def forward(self, span, condition, time,noisy
                ):
        # 输入 span 噪声数据 b, d
        # 条件数据 b, l, d
        # 时间 即 强度 b
        ###  b, n   b, l, d    b
        ### 时间正余弦编码 b, d
        time_emb = self.time_embed(time)  # b, d
        # span = inverse_sigmoid(span).sigmoid() # 反向sigmoid(sigmoid的逆函数)+sigmoid  # b, n
        # condition_time = condition + time_emb.unsqueeze(1)
        ### Span embedding
        output = self.noise_embed(span.to(torch.float32)) #+ 0.1 * time_emb  # b, d
        # output = self.noise_embed(span.to(torch.float32)) # b, d
        output = output.unsqueeze(1)  # b, 1, d
        #output=self.gg(torch.cat([output.unsqueeze(1),noisy],dim=-1))
        #output=noisy

        for diff_decoder in self.diff_decoders:
            output, _, _ = diff_decoder(output, condition)

        # for diff_decoder in self.diff_decoders1:
        #     noisy, _, _ = diff_decoder(noisy, condition)
        #output = output + noisy+condition.mean(1).unsqueeze(1)


        output = self.ln(output)#+condition


        return output.squeeze(1)  # b, d

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    if activation == "elu":
        return F.elu
    if activation == "lrelu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=5, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_t_attn=True,
                 bbox_embed_diff_each_layer=False,
                 num_class=23,
                 scale=2.,
                 denoise_network=None,
                 steps=50
                 ):
        super().__init__()

        # normalize_before transformer前是否进行norm，默认 false
        self.num_class = num_class
        self.d_model = d_model  # 模型维度
        self.nhead = nhead  # 多头注意力头数
        self.dec_layers = num_decoder_layers  # 降噪网络的 decoder 层数
        # self.num_queries = num_queries
        self.num_patterns = num_patterns  # 默认 0
        self.denoise_network = denoise_network  # 降噪网络中的 decoder 层

        # build diffusion
        timesteps = steps  # 训练步数
        sampling_timesteps = steps  # 推理（采样）步数
        # sampling_timesteps = 50
        self.num_proposals = num_queries

        betas = cosine_beta_schedule(timesteps)
        # betas =linear_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False

        self.scale = scale
        self.span_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)  # 不更新，可随模型保存
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.diff_span_embed = MLP(d_model, d_model, num_class, 1)

       # 0: background, 1: foreground
        self._reset_parameters()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x_spans, condition, time_cond,noisy):
        x_spans = torch.clamp(x_spans, min=-1 * self.scale, max=self.scale)
        x_spans = ((x_spans / self.scale) + 1) / 2
        # print(x_spans.shape, condition.shape, time_cond.shape)
        main_fea = self.denoise_network(x_spans, condition, time_cond,noisy)  # b, d
        # import pdb; pdb.set_trace()
        main_cood = self.diff_span_embed(main_fea)  # b, n
        # main_cood = main_cood.sigmoid()

        x_start = main_cood  # (batch, num_proposals, 2)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x_spans, time_cond, x_start)

        return pred_noise, x_start, main_fea, main_cood

    


    @torch.no_grad()
    def ddim_sample(self, batched_moments,noisy):
        if isinstance(batched_moments, tuple):
            batch = batched_moments[0].shape[0]
        else:
            batch = batched_moments.shape[0]  # b
        shape = (batch, self.num_class)  # b, n
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None
        x_spans = torch.randn(shape).cuda()  # b, n
        # import pdb;pdb.set_trace()
        step_num = 0

        all_hs = [x_spans]

        for time, time_next in time_pairs:  # T-1->T-2 .....

            step_num = step_num + 1

            time_cond = torch.full((batch,), time).cuda().long()  # b
            pred_noise, pred_start, hs, hs_cood = self.model_predictions(x_spans, batched_moments,
                                                                         time_cond,noisy)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            cc = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(pred_start)
            x_spans = pred_start * alpha_next.sqrt() + cc * pred_noise + sigma * noise  # b, n
            all_hs.append(hs_cood)
        return hs_cood  # b, n

    # @torch.no_grad()#p_sample
    # def ddim_sample(self, batched_moments,noisy):
    #     if isinstance(batched_moments, tuple):
    #         batch = batched_moments[0].shape[0]
    #     else:
    #         batch = batched_moments.shape[0] # b
    #     shape = (batch, self.num_class) # b, n
    #     total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
    #     # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    #
    #     x_start = None
    #     x_spans = torch.randn(shape).cuda() # b, n
    #     step_num = 0
    #
    #     for time, time_next in time_pairs: #T-1->T-2 .....
    #
    #         step_num = step_num+1
    #
    #         time_cond = torch.full((batch,), time).cuda().long() # b
    #         pred_noise, pred_start, hs, hs_cood = self.model_predictions(x_spans, batched_moments,
    #                                                     time_cond,noisy)
    #
    #         alpha = self.alphas[time]
    #         alpha_next = self.alphas_cumprod[time]
    #         beta = self.betas[time]
    #         # sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         # cc = (1 - alpha_next - sigma ** 2).sqrt()
    #         noise = torch.randn_like(pred_start)
    #         # x_spans = pred_start * alpha_next.sqrt() + cc * pred_noise + sigma * noise # b, n
    #         x_spans = 1 / torch.sqrt(alpha) * (
    #             x_spans - ((1-alpha) / (torch.sqrt(1-alpha_next))) * pred_noise) + torch.sqrt(
    #                 beta) * noise
    #
    #     # hs_class = self.diff_class_embed(hs)
    #     return hs_cood # b, n









    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        # n, 2     1     n, 2
        # b, n     b     b, n
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.triu(noise)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, x_start):
        """
        param gt_boxes: (center, long) # b, n
        """
        time = torch.randint(0, self.num_timesteps, (x_start.size(0),)).long().cuda()  # 随机时间步 b
        noise = torch.randn_like(x_start.float()).cuda()  # 随机噪声  b, n

        x_start = (x_start * 2. - 1.) * self.scale  # 归一化 [0, 1] --> [-2, 2]
        # noise sample
        x_t = self.q_sample(x_start=x_start, t=time, noise=noise)  # 加噪 # b, n
        x_t = torch.clamp(x_t, min=-1 * self.scale, max=self.scale)  # 限制 [-2, 2]
        x_t = ((x_t / self.scale) + 1) / 2.  # 归一化回来 [-2, 2] --> [0, 1]

        return x_t, noise, time

    def prepare_targets(self, targets):
        # b, n
        d_spans, d_noise, d_t = self.prepare_diffusion_concat(targets)  # 得到 x_t, noise, t   # b, d
        # b, n    b, n    b
        return d_spans, d_noise, d_t

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, condition, gt_target,noisy):
        """
        Args:
            condition: (batch_size, L, d)
            gt_target: (b, d)

        Returns:

        """
        #condition=self.encoder(condition)+condition

        if gt_target is not None:
            ### span normalization
            x_t_spans, noises, time_t = self.prepare_targets(gt_target)  # b, d   b, d    b
            # time_t = time_t.squeeze(-1) # b
            ### Span embedding + Intensity-aware attention
            hs = self.denoise_network(x_t_spans, condition, time_t,noisy)  # b, d
            ### out
            # ### predicted spans
            # hs_class = self.diff_class_embed(hs) # linear # 1, b, 2
            # ### scores
            hs_class = self.diff_span_embed(hs)  # MLP # b, n
            # outputs_coord = hs_cood.sigmoid()
        else:
            hs_class = self.ddim_sample(condition,noisy)  # b, n
            # hs_class, outputs_coord, hs = self.ddim_sample(tgt, memory_local, txt_src, memory_key_padding_mask=mask_local, pos=pos_embed_local)

        # memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d) condition
        return hs_class

class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        # self.linear = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        # self.linearrep= nn.Linear(768*2, 768)
        # # self.linearrep2 = nn.Linear(23, 23)
        self.fc = nn.Linear(768, num_class)
        #self.fc=MLP(768, 768, num_class, 1)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.scale=0.8#0.8
        self.steps=80#80
        self.dropout=0.1#0.1
        self.denoise_network = DenoiseNetwork(module_dim=768, module_id=0, layer_num=3, num_class=23,dropout=self.dropout)#3
        self.transformer = Transformer(
            d_model=768,
            dropout=0.5,
            num_queries=1,
            nhead=8,
            dim_feedforward=768,
            num_encoder_layers=1,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=True,
            activation='prelu',
            num_class=23,
            scale=self.scale,
            denoise_network=self.denoise_network,
            steps=self.steps
        )

        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    def majority_vote(self,predictions):

    # 确保所有预测张量具有相同的形状
        preds_tensor = torch.stack(predictions)  # shape: [num_models, bs]

    # 使用 mode() 函数获取每列（每个样本）出现次数最多的类别和其计数
        majority_class_indices, _ = torch.mode(preds_tensor, dim=0)

        return majority_class_indices

    def forward(self,*args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """

        rep_old, rep_new, label,noisy,ground,ground_label, outaaa,outbbb = self.sentence_encoder(*args)
        gt_target = F.one_hot(label, 23)
        rep_old1 = self.drop(rep_old)
        rep_old1 = self.fc(rep_old1)  # bsz,23
 

        if self.training:
            rep_new = self.transformer(rep_new, gt_target,noisy) 

        else:  
            rep_new = self.transformer(rep_new, None,noisy) 

        logits=rep_new
        rep=rep_new
        all_right=None


        
        return rep_new, rep_new,rep_new,rep_old1,all_right
