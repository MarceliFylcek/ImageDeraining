import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision.models as models


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net



##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg16().cuda()
        self.criterion = nn.L1Loss()

        # Normalized weights
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf, n_layers):
        """ID MultiScale Patch Discriminator

        Args:
            input_nc (int): number of input channels
            output_nc (int): number of channels from the generator
            ndf (int): number of convolutions/filters
            n_layers (int): number of discriminator layers
        """        
                
        super(MultiScaleDiscriminator, self).__init__()

        layers = [
            # Test + generated channels
            # start with ndf number of features
            nn.Conv2d(input_nc + output_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult_prev = 1
        nf_mult = 1

        for n in range(n_layers - 1):
            nf_mult_prev = nf_mult

            # Increse number of features
            nf_mult = min(2**n, 8)
            layers.extend([
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ])
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.extend([
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1),
            nn.Sigmoid()
        ])
        
        self.netD = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.netD(x)


class IDGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        """ ID generator

        Args:
            input_nc (_type_): number of input channels
            output_nc (_type_): 
            ngf (_type_): _description_
        """        

        super(IDGenerator, self).__init__()

        # Encoder part

        #! No downsamling
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1)
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf // 2)
        )
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf // 2, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1)
        )
        self.d1_ = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(1, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf // 2)
        )
        self.d2_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf // 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.d3_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.d4_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.d5_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.d6_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)
        )
        self.o1 = nn.Tanh()


    """
                                                    d6 ---> o1
        e1                                      d5
            e2 ------------------------------>d4
                e3                      d3
                    e4  -------->   d2
                        e5      d1
                            e6
    
    """

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        d1_ = self.d1_(e6)
        d2_ = self.d2_(d1_)
        d2 = torch.add(d2_, e4)
        d3_ = self.d3_(d2)
        d4_ = self.d4_(d3_)
        d4 = torch.add(d4_, e2)
        d5_ = self.d5_(d4)
        d6_ = self.d6_(d5_)
        d6 = nn.Identity()(d6_)
        o1 = self.o1(d6)
        return o1


class Vgg16(torch.nn.Module):
    """

    Returns features from VGG16

    VGG layers:
    0. Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    1. ReLU(inplace=True)
    2. Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    3. ReLU(inplace=True)
    4. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    5. Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    6. ReLU(inplace=True)
    7. Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    8. ReLU(inplace=True)
    9. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    10. Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    11. ReLU(inplace=True)
    12. Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    13. ReLU(inplace=True)
    14. Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    15. ReLU(inplace=True)
    16. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    17. Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    18. ReLU(inplace=True)
    19. Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    20. ReLU(inplace=True)
    21. Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    22. ReLU(inplace=True)
    23. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    24. Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    25. ReLU(inplace=True)
    26. Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    27. ReLU(inplace=True)
    28. Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    29. ReLU(inplace=True)
    30. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    """

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()

        # Get features from pretrained model
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        # Divide into 5 slices
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
