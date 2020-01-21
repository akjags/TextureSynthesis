
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature, **kwargs):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class StyleLossPCA(nn.Module):
    def __init__(self, target_feature, layerName, which_pc = None, pc_step_size=None, pc_layer = 'pool2'):
        super(StyleLossPCA, self).__init__()
        self.pca = np.load('/scratch/groups/jlg/texpca/{}_dims_histmatch.npy'.format(layerName)).item()['pca']
        self.target = gram_matrix(target_feature).detach().to(device)
        self.components = torch.from_numpy(self.pca.components_.T).to(device) # nFeatures x nComponents
        self.pca_mean = torch.from_numpy(self.pca.mean_).to(device).view(1,-1) # (nFeatures,)
        self.target_pca = self.transform(self.target.view(1,-1))
        if which_pc is not None and layerName == pc_layer:
            self.target_pca[0,which_pc] = self.target_pca[0,which_pc] + pc_step_size

    def forward(self, input):
        G = self.transform(gram_matrix(input).view(1,-1))
        self.loss = F.mse_loss(G, self.target_pca)
        return input

    def transform(self, mtx):
        mtx_pca = torch.mm(mtx - self.pca_mean, self.components)
        return mtx_pca

class StyleLossNMF(nn.Module):
    def __init__(self, target_feature, layerName, which_pc = None, pc_step_size=None, pc_layer='pool2', **kwargs):
        super(StyleLossNMF, self).__init__()
        self.nmf = np.load('/scratch/groups/jlg/texpca/{}_dims_nmf.npy'.format(layerName)).item()['nmf']
        self.target = gram_matrix(target_feature).detach().to(device)
        self.components = torch.from_numpy(self.nmf.components_.T).to(device) # nFeatures x nDimensions
        self.target_nmf = self.transform(self.target.view(1,-1))
        print(self.target_nmf.shape)
        if which_pc is not None and layerName == pc_layer:
          self.target_nmf[0,which_pc] = self.target_nmf[0,which_pc] + pc_step_size

    def forward(self, input):
        G = self.transform(gram_matrix(input).view(1,-1))
        self.loss = F.mse_loss(G, self.target_nmf)
        return input

    def transform(self, mtx):
        mtx_lda = torch.mm(torch.abs(mtx), self.components.float())
        return mtx_lda

class StyleLossDiag(nn.Module):
    def __init__(self, target_feature, **kwargs):
        super(StyleLossDiag, self).__init__()
        self.target = torch.diag(gram_matrix(target_feature).detach())

    def forward(self, input):
        G = torch.diag(gram_matrix(input))
        self.loss = F.mse_loss(G, self.target)
        return input

class StyleLossPool2(nn.Module):
    def __init__(self, target_feature, layerName=None):
        super(StyleLossPool2, self).__init__()
        if layerName == 'pool2':
            self.target = 10000*torch.ones_like(target_feature)
        else:
            self.target = torch.zeros_like(target_feature)

    def forward(self, input):
        pdb.set_trace()
        assert (input >= 0. & input <= 1.).all()

        self.loss = F.mse_loss(input, self.target)
        return input


