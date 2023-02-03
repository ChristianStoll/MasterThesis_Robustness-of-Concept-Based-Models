"""
Slot attention model based on code of tkipf and the corresponding paper Locatello et al. 2020
"""
import math
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import numpy as np
from torchsummary import summary


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128, return_attn=False):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.return_attn = return_attn
        self.attention_result = None

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim, eps=1e-05)
        self.norm_slots = nn.LayerNorm(dim, eps=1e-05)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-05)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_inputs(inputs)
        k, v = self.project_k(inputs), self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            self.attention_result = attn.clone()
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        if self.return_attn:
            return slots, attn, updates
        else:
            return slots


class SlotAttention_encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SlotAttention_encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class BigSlot_encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(BigSlot_encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (7, 7), stride=(1, 1), padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.network(x)


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution, device="cuda"):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.grid = torch.FloatTensor(build_grid(resolution))
        self.grid = self.grid.to(device)
        self.resolution = resolution[0]
        self.hidden_size = hidden_size

    def forward(self, inputs):
        return inputs + self.dense(self.grid).view((-1, self.hidden_size, self.resolution, self.resolution))


############
# Transformers #
############
"""
Code largely from https://github.com/juho-lee/set_transformer by yoonholee.
"""
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    """
    Set Transformer used for the Neuro-Symbolic Concept Learner.
    """

    def __init__(self, dim_input=3, dim_output=40, dim_hidden=128, num_heads=4, ln=False):
        """
        Builds the Set Transformer.
        :param dim_input: Integer, input dimensions
        :param dim_output: Integer, output dimensions
        :param dim_hidden: Integer, hidden layer dimensions
        :param num_heads: Integer, number of attention heads
        :param ln: Boolean, whether to use Layer Norm
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_input, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
            SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=1, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze()


class Bottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_attributes, use_sigmoid=False):
        super(Bottleneck, self).__init__()
        self.n_attributes = n_attributes + 1
        self.use_sigmoid = use_sigmoid

        self.network = nn.Sequential(
            # N x num_slots x hidden_channels
            nn.Linear(in_channels, hidden_channels),
            # N x num_slots x hidden_channels
            nn.ReLU(inplace=True),
            # N x num_slots x hidden_channels
            nn.Linear(hidden_channels, n_attributes),
            # N x num_slots x 112+1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# TODO this is the Adapted ConceptModel
class HybridResNet(nn.Module):
    def __init__(self, n_slots, n_iters, n_attr, img_size=128, in_channels=3,
                 encoder_hidden_channels=64,
                 attention_hidden_channels=128,
                 bn_hidden_channels=256,
                 return_attn=False,
                 device="cuda"):
        super(HybridResNet, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.n_attributes = n_attr + 1  # additional slot to indicate if it is a object or empty (background) slot
        self.img_size = img_size
        self.return_attn = return_attn
        self.device = device

        # load ResNet18 from torchvision.models as encoder
        self.encoder_cnn = tv_models.resnet18(pretrained=True)
        # set last layer encoder_hidden_channels neurons
        self.encoder_cnn.fc = nn.Linear(self.encoder_cnn.fc.in_features, encoder_hidden_channels)

        self.encoder_pos = SoftPositionEmbed(encoder_hidden_channels, resolution=(img_size // 4, img_size // 4),
                                             device=device)
        self.layer_norm = nn.LayerNorm(encoder_hidden_channels, eps=1e-05)
        self.mlp = MLP(hidden_channels=encoder_hidden_channels)
        self.slot_attention = SlotAttention(num_slots=n_slots, dim=encoder_hidden_channels, iters=n_iters, eps=1e-8,
                                            hidden_dim=attention_hidden_channels, return_attn=self.return_attn)

        # first attempt for a bottleneck model
        self.bottleneck = Bottleneck(in_channels=encoder_hidden_channels, hidden_channels=bn_hidden_channels,
                                     n_attributes=self.n_attributes)



    def forward(self, x):
        x = self.encoder_cnn(x)
        print(f'cnn: {x.shape}')
        x = self.encoder_pos(x)

        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm(x)
        x = self.mlp(x)
        if self.return_attn:
            x, attn, updates = self.slot_attention(x)
        else:
            x = self.slot_attention(x)
        x = self.bottleneck(x)
        if self.return_attn:
            return x, attn, updates
        else:
            return x


# TODO this is the full SlotAttention Model - every other class is a building block of it
class HybridConceptModel_Large(nn.Module):
    def __init__(self, n_slots, n_iters, n_attr, img_size=128,
                 in_channels=3,
                 encoder_hidden_channels=256,
                 attention_hidden_channels=512,
                 bn_hidden_channels=256,
                 return_attn=False,
                 device="cuda"):
        super(HybridConceptModel_Large, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.n_attributes = n_attr + 1  # additional slot to indicate if it is a object or empty (background) slot
        self.img_size = img_size
        self.return_attn = return_attn
        self.device = device

        # first attempt on the encoder cnn
        self.encoder_cnn = BigSlot_encoder(in_channels=in_channels, hidden_channels=encoder_hidden_channels)
        self.encoder_pos = SoftPositionEmbed(encoder_hidden_channels, resolution=(img_size // 4, img_size // 4),
                                             device=device)
        self.layer_norm = nn.LayerNorm(encoder_hidden_channels, eps=1e-05)
        self.mlp = MLP(hidden_channels=encoder_hidden_channels)
        self.slot_attention = SlotAttention(num_slots=n_slots, dim=encoder_hidden_channels, iters=n_iters, eps=1e-8,
                                            hidden_dim=attention_hidden_channels, return_attn=self.return_attn)

        # first attempt for a bottleneck model
        self.bottleneck = Bottleneck(in_channels=encoder_hidden_channels, hidden_channels=bn_hidden_channels,
                                     n_attributes=self.n_attributes)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_pos(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm(x)
        x = self.mlp(x)
        if self.return_attn:
            x, attn, updates = self.slot_attention(x)
        else:
            x = self.slot_attention(x)
        x = self.bottleneck(x)
        if self.return_attn:
            return x, attn, updates
        else:
            return x


# TODO this is the full SlotAttention Model - every other class is a building block of it
class HybridConceptModel(nn.Module):
    def __init__(self, n_slots, n_iters, n_attr, img_size=128,
                 in_channels=3,
                 encoder_hidden_channels=256,
                 attention_hidden_channels=512,
                 bn_hidden_channels=256,
                 return_attn=False,
                 device="cuda"):
        super(HybridConceptModel, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.n_attributes = n_attr + 1  # additional slot to indicate if it is a object or empty (background) slot
        self.img_size = img_size
        self.return_attn = return_attn
        self.device = device

        # first attempt on the encoder cnn
        self.encoder_cnn = SlotAttention_encoder(in_channels=in_channels, hidden_channels=encoder_hidden_channels)
        self.encoder_pos = SoftPositionEmbed(encoder_hidden_channels, resolution=(img_size // 4, img_size // 4),
                                             device=device)
        self.layer_norm = nn.LayerNorm(encoder_hidden_channels, eps=1e-05)
        self.mlp = MLP(hidden_channels=encoder_hidden_channels)
        self.slot_attention = SlotAttention(num_slots=n_slots, dim=encoder_hidden_channels, iters=n_iters, eps=1e-8,
                                            hidden_dim=attention_hidden_channels, return_attn=self.return_attn)

        # first attempt for a bottleneck model
        self.bottleneck = Bottleneck(in_channels=encoder_hidden_channels, hidden_channels=bn_hidden_channels,
                                     n_attributes=self.n_attributes)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_pos(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm(x)
        x = self.mlp(x)
        if self.return_attn:
            x, attn, updates = self.slot_attention(x)
        else:
            x = self.slot_attention(x)
        x = self.bottleneck(x)
        if self.return_attn:
            return x, attn, updates
        else:
            return x


class JointHybrid_Model(nn.Module):
    def __init__(self, n_attr, n_classes, img_size, n_slots, n_iters, return_concepts,
                 in_channels=3,
                 encoder_hidden_channels=64,
                 attention_hidden_channels=128,
                 bn_hidden_channels=256,
                 device="cuda"):
        super(JointHybrid_Model, self).__init__()
        self.n_attributes = n_attr  # additional attribute to model object / background(empty)
        self.n_classes = n_classes
        self.return_concepts = return_concepts
        self.concept_model = HybridConceptModel(n_slots=n_slots, n_iters=n_iters, n_attr=self.n_attributes,
                                                img_size=img_size, in_channels=in_channels,
                                                encoder_hidden_channels=encoder_hidden_channels,
                                                attention_hidden_channels=attention_hidden_channels,
                                                bn_hidden_channels=bn_hidden_channels, device=device)
        self.set_classifier = SetTransformer(dim_input=self.n_attributes+1, dim_output=self.n_classes, dim_hidden=128,
                                             num_heads=4)

    def forward(self, x):
        c = self.concept_model(x)
        y = self.set_classifier(c)
        if self.return_concepts:
            return y, c
        else:
            return y


def mkresnet():
    encoder_hidden_channels=128
    net = tv_models.resnet18(pretrained=True)

    # set last layer to be Bottleneck having n_attr
    net.fc = nn.Linear(512, encoder_hidden_channels)

    return net


if __name__ == "__main__":
    resolution = 224
    n_slots = 2
    print('concept model')
    x = torch.rand(1, 3, resolution, resolution)
    cmodel = HybridConceptModel(n_slots=n_slots, n_iters=3, n_attr=112, img_size=resolution,
                                encoder_hidden_channels=256, attention_hidden_channels=512, device='cpu')
    output = cmodel(x)
    print(f'concept output: {output.shape}')
    summary(cmodel, (3, resolution, resolution), device='cpu')

    """model = SetTransformer(dim_input=113, dim_hidden=128, num_heads=4, dim_output=200, ln=True)
    print(model)
    print()
    summary(model, (2, 113), device='cpu')
    x = torch.rand(5, 2, 113)
    output = model(x)"""

    """print('resnet')
    encoder_hidden_channels = 128
    net = tv_models.resnet18(pretrained=True)

    # set last layer to be Bottleneck having n_attr
    net.fc = nn.Linear(net.fc.in_features, encoder_hidden_channels)
    net.eval()
    summary(net, (3, resolution, resolution), device='cpu')


    print('resnet model')
    model = HybridResNet(n_slots=n_slots, n_iters=3, n_attr=112, img_size=resolution,
                         encoder_hidden_channels=128, attention_hidden_channels=256, device='cpu')
    summary(model, (3, resolution, resolution), device='cpu')"""

