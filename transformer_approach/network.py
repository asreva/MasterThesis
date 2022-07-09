"""
Aim: Implement the network for the MI prediction from patches along the artery
TR-Net copied and adapted from https://github.com/XinghuaMa/TRNet/tree/main/method
Author: Ivan-Daniel Sievering for the LTS4 Lab (EPFL)
"""

# --- Settings --- #
#Libraries
import sys
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
# from libauc.losses import CompositionalLoss
# from libauc.optimizers import PDSCA
from loss import FocalLoss

# --- Functions --- #
def count_parameters(model):
    """ 
        Aim: Returns the number of trainable parameters
        
        Parameters:
            - model: the model
            
        Output: the number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def xavier_uniform_init(m):
    """ 
        Aim: Apply Xavier Uniform initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def xavier_normal_init(m):
    """ 
        Aim: Apply Xavier Normal initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
def kaiming_uniform_init(m):
    """ 
        Aim: Apply He Uniform initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def kaiming_normal_init(m):
    """ 
        Aim: Apply He Normal initialization to a layer Linar or Conv2d
        
        Parameters:
            - m: the layer
    """
    
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# --- Classes --- #
class Residual_Connection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Layer_Normal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MSA_Block(nn.Module):
    def __init__(self, dim_seq, num_heads, dim_head):
        super().__init__()
        dim_inner = dim_head * num_heads

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim_seq, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_seq)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        b, n_l, _, h = *x.shape, self.num_heads
        q, k, v = map(lambda t: rearrange(t, 'b nw_l (h d) -> b h nw_l d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h nw_l d -> b nw_l (h d)')
        out = self.to_out(out)

        return out


class transformer_block(nn.Module):
    def __init__(self, dim_seq, dim_mlp, num_heads, dim_head):
        super().__init__()

        self.attention_block = Residual_Connection(
            Layer_Normal(dim_seq, MSA_Block(dim_seq=dim_seq, num_heads=num_heads, dim_head=dim_head)))

        self.mlp_block = Residual_Connection(Layer_Normal(dim_seq, MLP_Block(dim=dim_seq, hidden_dim=dim_mlp)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)

        return x


class Transformer_Structure(nn.Module):
    def __init__(self, dim_seq=3456, num_heads=3, dim_head=18, num_encoders=8, num_regions=4):
        super().__init__()

        self.order_embedding = nn.Parameter(torch.randn(1, num_regions, dim_seq))
        self.to_trans, self.to_seq = nn.Linear(dim_seq, dim_head), nn.Linear(dim_head, dim_seq)

        self.layers = nn.ModuleList([])
        for _ in range(num_encoders):
            self.layers.append(transformer_block(dim_seq=dim_head, num_heads=num_heads,
                                                 dim_mlp=dim_seq * 2, dim_head=dim_head))

    def forward(self, img):

        x = img + self.order_embedding
        x = self.to_trans(x)
        for layer in self.layers:
            x = layer(x)
        x = self.to_seq(x)
        return x


class Softmax_Classify(nn.Module):
    def __init__(self, hidden_size, num_linear, num_class):
        super().__init__()

        tmp_hidden_size = hidden_size

        self.layers = nn.ModuleList([])
        for _ in range(num_linear - 1):
            self.layers.append(nn.Linear(int(tmp_hidden_size), int(tmp_hidden_size / 2)))
            tmp_hidden_size /= 2

        self.layers.append(nn.Linear(int(tmp_hidden_size), num_class))

        self.soft_max = nn.Softmax(dim=0)

    def forward(self, x):
        b, l, n = x.shape
        
        x = rearrange(x, 'b l n -> (b l) n')
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.soft_max(x)
        
        x = rearrange(x, '(b l) n -> b l n', b=b)
        
        return x


def conv2(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Conv2d(nn.Module):
    def __init__(self, train_configuration, in_channels, num_levels=4, f_maps=16, dropout=None):
        super().__init__()

        self.in_channels = in_channels

        self.layers = nn.ModuleList([])
        for i in range(num_levels):
            self.layers.append(conv2(self.in_channels, f_maps * (2 ** i), stride=1))
            self.layers.append(nn.BatchNorm2d(f_maps * (2 ** i)))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=1))
            self.in_channels = f_maps * (2 ** i)
            
        self.dropout = torch.nn.Dropout2d(dropout)

    def forward(self, x):
        b, l, c, n_h, n_w = x.shape

        x = rearrange(x, 'b l c n_h n_w -> (b l) c n_h n_w')

        for layer in self.layers:
            x = layer(x)
        
        x = rearrange(x, '(b l) c n_h n_w  -> b l (c n_h n_w)', l=l)
        
        x = self.dropout(x)
        
        

        return x


class transformer_network(nn.Module):
    """ 
        Aim: Get all the patches extracted from an image and extract from it the structure
        
        Structure:
            - 2D CNN: each patch goes through a CNN feature analysis
            - Transformer: all the patches go to a transformer structure
            - Softmax classifier: the result of the transformer is converted to classification at each patch level
    """

    def __init__(self, train_configuration, in_channels=1, num_levels=4, f_maps=16, dim_hidden=3456, num_heads=3, dim_head=18,
                 num_encoders=8, num_linear=2, num_class=2, num_regions=4, verbose=False):
        super().__init__()

        self._2dcnn = Conv2d(train_configuration, in_channels=in_channels, num_levels=num_levels, f_maps=f_maps, dropout=train_configuration["dropout"])

        self.transformer_structure = Transformer_Structure(dim_seq=dim_hidden, num_heads=num_heads,
                                                           dim_head=dim_head, num_encoders=num_encoders,
                                                          num_regions=num_regions)

        self.softmax_classify = Softmax_Classify(hidden_size=dim_hidden, num_linear=num_linear, num_class=num_class)
        
        self.verbose=verbose

    def forward(self, x):
        
        if self.verbose:
            print("Input: {}".format(x.shape))
            
        x = self._2dcnn(x) 
        if self.verbose:
            print("CNN: {}".format(x.shape))
                  
        x = self.transformer_structure(x)
        if self.verbose:
            print("Transformer: {}".format(x.shape))
            
        x = self.softmax_classify(x)
        if self.verbose:
            print("Softmax: {}".format(x.shape))

        return x
    
class ArteryLevelDNN(nn.Module):
    """ 
        Aim: Get the analysis at artery level of the two views
        
        Structure:
            - Analyse of the two views: the two views go to the transformer_network class
            - Concat: their result is concatenated
            - Return:   - this result is sent to next level
                        - the analysis is sent for further siamese comparison
            - Artery prediction: a prediction is also done at the artery level
    """
    
    def __init__(self, train_configuration, in_channels=1, num_levels=4, f_maps=16, dim_hidden=3456, num_heads=3, dim_head=18,
                 num_encoders=8, num_linear=2, num_class=2, num_regions=4, verbose=False):
        super().__init__()

        self.artery_analyser = transformer_network(train_configuration, in_channels=in_channels, num_levels=num_levels, f_maps=f_maps, dim_hidden=dim_hidden, num_heads=num_heads, dim_head=dim_head,
                 num_encoders=num_encoders, num_linear=num_linear, num_class=num_class, num_regions=num_regions, verbose=verbose)
        
        self.verbose=verbose

    def forward(self, img_view1, img_view2):
        
        # siamese analysis
        if self.verbose:
            print("Shape image 1 {} Shape image 2 {}".format(img_view1.shape, img_view2.shape))
            
        analyse_view1 = self.artery_analyser(img_view1)
        analyse_view2 = self.artery_analyser(img_view2)
        
        if self.verbose:
            print("Analysis image 1 {} Analysis image 2 {}".format(analyse_view1.shape, analyse_view2.shape))
        
        artery_analyse = torch.concat([analyse_view1, analyse_view2], dim=1)
        
        if self.verbose:
            print("Analysis together {}".format(artery_analyse.shape))

        artery_pred = torch.max(artery_analyse, dim=1).values
        
        # return analyse to compute then siamese loss
        return artery_analyse, artery_pred, analyse_view1.detach(), analyse_view2.detach()
    
class PatientLevelDNN(nn.Module):
    """ 
        Aim: Get the prediction at the patient level
        
        Structure:
            - Analyse of the three arteries: the three arteries data go to the ArteryLevelDNN class
            - Concat: their result is concatenated
            - Prediction: the prediction is done based on the concatenated result
    """
    def __init__(self, train_configuration, in_channels=1, num_levels=4, f_maps=16, dim_hidden=3200, num_heads=3, dim_head=18,
                 num_encoders=8, num_linear=2, num_class=1, num_regions=192, verbose=False):
        super().__init__()

        num_regions = 0
        for nb_patch in train_configuration["nb_patch_l"]:
            num_regions += nb_patch
        
        self.lad_analyser = ArteryLevelDNN(train_configuration, in_channels, num_levels, f_maps, dim_hidden, num_heads, dim_head,
                 num_encoders, num_linear, num_class, num_regions, verbose)
        self.lcx_analyser = ArteryLevelDNN(train_configuration, in_channels, num_levels, f_maps, dim_hidden, num_heads, dim_head,
                 num_encoders, num_linear, num_class, num_regions, verbose)
        self.rca_analyser = ArteryLevelDNN(train_configuration, in_channels, num_levels, f_maps, dim_hidden, num_heads, dim_head,
                 num_encoders, num_linear, num_class, num_regions, verbose)

        self.verbose=verbose

    def forward(self, imgs):
        lad_imgs, lcx_imgs, rca_imgs = imgs
        
        if self.verbose:
            print("Shape LAD image view 0 {}".format(lad_imgs[0].shape))
        
        lad_analysis, lad_pred, res_lad_view1, res_lad_view2 = self.lad_analyser(lad_imgs[0], lad_imgs[1])
        lcx_analysis, lcx_pred, res_lcx_view1, res_lcx_view2 = self.lcx_analyser(lcx_imgs[0], lcx_imgs[1])
        rca_analysis, rca_pred, res_rca_view1, res_rca_view2 = self.rca_analyser(rca_imgs[0], rca_imgs[1])
        
        if self.verbose:
            print("Shape LAD anlaysis {}".format(lad_analysis.shape))
        
        patient_analysis = torch.concat([lad_analysis, lcx_analysis, rca_analysis], dim=1)
        
        if self.verbose:
            print("Shape full anlaysis {}".format(patient_analysis.shape))
        
        pred = torch.max(patient_analysis, dim=1).values
        
        if self.verbose:
            print("Shape output {}".format(pred.shape))
        
        return pred, lad_pred, lcx_pred, rca_pred, \
                    (res_lad_view1, res_lad_view2), (res_lcx_view1, res_lcx_view2), (res_rca_view1, res_rca_view2)

    
def init_net(train_configuration):
    """ 
        Aim: Initialise a network and its optimisation (optimiser, ...) based on the given train configuration
        
        Parameters:
            - train_configuration: dictionnary defining the parameters of the run (see configuration_dict.py)
        
        Output: the network, the train scheduler, the optimizer
    """
    
    # Create the network
    if train_configuration["load_network"] is None:
        net = train_configuration["network_class"](train_configuration)
    else:
        print("Loading existing network {}".format(train_configuration["load_network"]))
        net = train_configuration["network_class"](train_configuration)
        net.load_state_dict(torch.load(train_configuration["load_network"]))
    print("Nb of parameters is {}".format(count_parameters(net)))
    
    criterion_l = []
    optimizer_l = []
    scheduler_l = []
    
    for i in range(len(train_configuration["criterion_type"])):
        # Choose criteraion
        if train_configuration["criterion_type"][i] == "BCE":
            criterion_l.append(torch.nn.BCELoss())
        elif train_configuration["criterion_type"][i] == "AUC":
            criterion_l.append(AUCMLoss(imratio=train_configuration["PESG_imratio"]))
        elif train_configuration["criterion_type"][i] == "Focal":
            criterion_l.append(FocalLoss(train_configuration["focal_alpha"], train_configuration["focal_gamma"], train_configuration["focal_reduction"]))
        elif train_configuration["criterion_type"][i] == "CompositionalLoss":
            criterion_l.append(CompositionalLoss(imratio=train_configuration["Compo_imratio"]))
        else:
            print("Criterion not found. Exit code (network.py).")
            sys.exit()

        # Choose the optimizer and its scheduler 
        if train_configuration["optimizer_type"][i] == "SGD":
            optimizer_l.append(torch.optim.SGD(net.parameters(), lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"], momentum=train_configuration["SGD_momentum"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))

        elif train_configuration["optimizer_type"][i] == "Adam":
            optimizer_l.append(torch.optim.Adam(net.parameters(), lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"]))
            scheduler_l.append(None) # no scheduler for adam

        elif train_configuration["optimizer_type"][i] == "PESG":
            # imratio is percentage of positive cases, gamma and margin from their example
            optimizer_l.append(PESG(net, lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"],
                             a=criterion_l[i].a, b=criterion_l[i].b, alpha=criterion_l[i].alpha, imratio=train_configuration["PESG_imratio"], gamma=train_configuration["PESG_gamma"], margin=train_configuration["PESG_margin"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))
            
        elif train_configuration["optimizer_type"][i] == "PDSCA":
            # imratio is percentage of positive cases, gamma and margin from their example
            optimizer_l.append(PDSCA(net, lr=train_configuration["learning_rate"][i], weight_decay=train_configuration["weight_decay"],
                             a=criterion_l[i].a, b=criterion_l[i].b, alpha=criterion_l[i].alpha, imratio=train_configuration["Compo_imratio"], gamma=train_configuration["Compo_gamma"], margin=train_configuration["Compo_margin"], beta1=train_configuration["Compo_beta1"], beta2=train_configuration["Compo_beta2"]))
            scheduler_l.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l[i], mode='min', factor=train_configuration["scheduler_factor"], patience=train_configuration["scheduler_patience"], verbose=True))

        else:
            print("Optimizer not found. Exit code (network.py).")
            sys.exit()

    # Apply weights and biases initialisation
    if train_configuration["load_network"] is None:
        if train_configuration["init"] == "Xavier Uniform":
            net.apply(xavier_uniform_init)
        elif train_configuration["init"] == "Xavier Normal":
            net.apply(xavier_normal_init)
        elif train_configuration["init"] == "Kaiming Uniform":
            net.apply(kaiming_uniform_init)
        elif train_configuration["init"] == "Kaiming Normal":
            net.apply(kaiming_normal_init)
        else:
            print("Unknown Initialisation")
            sys.exit()

    return net, criterion_l, scheduler_l, optimizer_l