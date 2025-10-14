import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import Attention

if not hasattr(Attention, "head_dim"):
    def head_dim(self):
        return self.qkv.weight.shape[1] // 3 // self.num_heads
    Attention.head_dim = property(head_dim)

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

def random_indexes_block(size : int, ratio : float):

    h = w = int(np.sqrt(size))
    forward_indexes, backward_indexes = [], []

    M = set()
    while len(M) <= int(ratio * size):

        s = np.random.randint(16, max(16, int(ratio * size) - len(M)) + 1)
        r = np.random.uniform(0.3, 1 / 0.3)
        a = int(np.sqrt(s * r))
        b = int(np.sqrt(s / r))

        t = np.random.randint(0, h - a + 1) if h - a > 0 else 0
        l = np.random.randint(0, w - b + 1) if w - b > 0 else 0

        for i in range(t, min(t + a, h)):
            for j in range(l, min(l + b, w)):
                M.add(i * w + j)

    M = np.array(sorted(list(M)))
    mask = np.zeros(size, dtype=bool)
    mask[M] = True

    visible = np.where(~mask)[0]
    masked = np.where(mask)[0]

    forward_indexes = np.concatenate([visible, masked])
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def random_indexes_grid(size: int):

    h = w = int(np.sqrt(size))
    visible = []
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            visible.append(i * w + j)

    visible = np.array(visible)
    mask = np.ones(size, dtype=bool)
    mask[visible] = False

    masked = np.where(mask)[0]

    forward_indexes = np.concatenate([visible, masked])
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio, sampling) -> None:
        super().__init__()
        self.ratio = ratio
        self.sampling = sampling

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        if self.sampling == 'random':
            indexes = [random_indexes(T) for _ in range(B)]
        elif self.sampling == 'block':
            indexes = [random_indexes_block(T, self.ratio) for _ in range(B)]
        elif self.sampling == 'grid':
            indexes = [random_indexes_grid(T) for _ in range(B)]
        else:
            raise NotImplementedError(f'Sampling method {self.sampling} is not implemented.')
        
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 mask_encoder=False,
                 sampling='random'
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio, sampling)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.mask_encoder = mask_encoder
        if self.mask_encoder:
            self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
        if self.mask_encoder:
            trunc_normal_(self.mask_token, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        T_full = forward_indexes.shape[0]
        T_visible = patches.shape[0]

        if self.mask_encoder:
            mask_tokens = self.mask_token.expand(T_full - T_visible, patches.shape[1], -1)
            patches = torch.cat([patches, mask_tokens], dim=0)
            patches = take_indexes(patches, backward_indexes)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes, T_visible

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 mask_encoder=False
                 ) -> None:
        super().__init__()

        self.mask_encoder = mask_encoder
        if not self.mask_encoder:
            self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        if not self.mask_encoder:
            trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes, T_visible):

        if not self.mask_encoder:
            cls_backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
            features = torch.cat([features, self.mask_token.expand(cls_backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
            features = take_indexes(features, cls_backward_indexes)
        
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T_visible:] = 1
        mask = take_indexes(mask, backward_indexes)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 mask_encoder=False,
                 sampling='random'
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, mask_encoder, sampling)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, mask_encoder)

    def forward(self, img):
        features, backward_indexes, T_visible = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes, T_visible)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes, T_visible = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes, T_visible)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
