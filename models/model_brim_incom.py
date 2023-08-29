import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import join
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 1,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            # dropout=0.25
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class TransLayer1(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 1,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            # dropout=0.25
            dropout=0.1
        )

    def forward(self, x):
        temp = x
        x,attn =self.attn(self.norm(x),return_attn = True)
        x = x + temp
        print("attn.size:",attn.size())

        return x,attn
    
class TransLayer2(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            # dropout=0.25
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            # [b, 256] => [b, 64]
            nn.Linear(256, 64),
            nn.ReLU(),
            # [b, 64] => [b, 20]
            nn.Linear(64, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            nn.Linear(20, 64),
            nn.ReLU(),
            # [b, 64] => [b, 256]
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        # batchsz = x.size(0)
        # # flatten
        # x = x.view(batchsz, -1)
        # encode
        z = self.encoder(x)
        # decode
        out = self.decoder(z)
        # reshape
        # x = x.view(batchsz, 1, 28, 28)

        return out


class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.1):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())
        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

def SNN_Block(dim1, dim2, dropout=0.1):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))






### MMF (in the PORPOISE Paper)
class Brim_incom(nn.Module):
    def __init__(self, 
        omic_input_dim,
        path_input_dim=1024, 
        fusion='bilinear', 
        dropout=0.1,
        n_classes=4, 
        scale_dim1=8, 
        scale_dim2=8, 
        gate_path=1, 
        gate_omic=1, 
        skip=True, 
        dropinput=0.10,
        use_mlp=False,
        size_arg = "small",
        ):
        super(Brim_incom, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim,256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.n_classes = n_classes

        size = self.size_dict_path[size_arg]
        if dropinput:
            fc = [nn.Dropout(dropinput), nn.Linear(size[0], 512), nn.ELU(), nn.Dropout(dropout),nn.Linear(512, size[1]), nn.ELU(), nn.Dropout(dropout)]
        else:
            fc = [nn.Linear(size[0], 256), nn.ReLU(), nn.Dropout(dropout)]


        self.wsi_net = nn.Sequential(*fc)
        self.pos_layer = PPEG(dim=256)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.layer1 = TransLayer(dim=256)
        self.layer2 = TransLayer(dim=256)
        self.norm = nn.LayerNorm(256)
        self.AE_path_gene = AE()
        self.AE_gene_path = AE()
        ### Constructing Genomic SNN
        if self.fusion is not None:

            Block = SNN_Block
            hidden = self.size_dict_omic['small']
            fc_omic = [Block(dim1=omic_input_dim, dim2=256, dropout=dropout)]  ##0.1
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout)) ##0.1
            self.fc_omic = nn.Sequential(*fc_omic)
            
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=256)
            

        self.classifier_mm = nn.Linear(256, n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.wsi_net = self.wsi_net.to(device)
        self.pos_layer = self.pos_layer.to(device)
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)
        self.norm = self.norm.to(device)
        
        self.classifier_mm = self.classifier_mm.to(device)
        self.AE_gene_path = self.AE_gene_path.to(device)
        self.AE_path_gene = self.AE_path_gene.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer
        h_path_bag = torch.transpose(h_path_bag,1,0)
        h_path = torch.cat([h_path_bag],dim=1) #[B,N,256]
        H = h_path.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h_path = torch.cat([h_path, h_path[:,:add_length,:]],dim = 1) #[B, N, 512]
        B = h_path.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h_path = torch.cat((cls_tokens, h_path), dim=1)
        #---->Translayer x1
        h_path = self.layer1(h_path) #[B, N, 512]

        #---->PPEG
        h_path = self.pos_layer(h_path, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h_path = self.layer2(h_path) #[B, N, 512]

        #---->cls_token
        h_path = self.norm(h_path)[:,0]
        
        validing = kwargs['validing']
        if validing == "Training":
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic)
            ae_path = self.AE_gene_path(h_omic)
        ae_omic = self.AE_path_gene(h_path)
        
        
        
        if validing == "pseudo_gene":
            h_omic = ae_omic
            ae_path = None
        if validing == "pseudo_path":
            h_path = ae_path
        if self.fusion == 'bilinear':
            h_mm = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))
        elif self.fusion == 'lrb':
            h_mm  = self.mm(h_path, h_omic) # logits needs to be a [1 x 4] vector 
            return h_mm

        h_mm  = self.classifier_mm(h_mm) # logits needs to be a [B x 4] vector      
        assert len(h_mm.shape) == 2 and h_mm.shape[1] == self.n_classes


        # return h_path, h_omic, h_mm
        return h_mm,h_omic,h_path,ae_omic,ae_path
