import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
import numpy as np
import itertools
import torch_optimizer as new_optim
from torchvision.utils import save_image

from MultiScaleNet import MultiScaleDiscriminator
from MultiScaleNet import MultiScaleGenerator
from MultiScaleNet import DeGenerator
from ImageDataset import ImageDatasetFolder
from Visualization import VisdomPlotter

import datetime
import time
import os

'''
references
'wasserstein-div' https://arxiv.org/abs/1712.01026 "Wasserstein Divergence for GANs"
'wasserstein-gp' https://arxiv.org/abs/1704.00028 "Improved Training of Wasserstein GANs"
'wasserstein' https://arxiv.org/abs/1701.07875 "Wasserstein GAN"

spectral norm https://arxiv.org/abs/1802.05957 "Spectral Normalization for Generative Adversarial Networks"
self attention layer https://arxiv.org/abs/1805.08318 "Self-Attention Generative Adversarial Networks"

pixel norm and minibatch std https://arxiv.org/abs/1710.10196 "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
filter response norm https://arxiv.org/abs/1911.09737 "Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks"

network deconv https://arxiv.org/abs/1905.11926 "Network Deconvolution" WARNING this thing eat gpu memory like there's no tomorrow

the net structure used is
multi scale gradient net https://arxiv.org/abs/1903.06048 "MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis"

rADAM from https://github.com/jettify/pytorch-optimizer

relativistic discriminator gans https://arxiv.org/pdf/1807.00734.pdf and https://medium.com/@jonathan_hui/gan-rsgan-ragan-a-new-generation-of-cost-function-84c5374d3c6e
'''

# general
resolution = 128  # final image resolution
batch_size = 16  # real gpu batch size
aggregated_batch_size = 1  # batch accumulation (if the real batch size you want doesn't fit in memory). effective batch size is aggregated_batch_size * batch_size. 1 default
use_cuda = True
num_epochs = 10000
load_checkpoint = True
checkpoint_location = "./temp/"
checkpoint_filename = "checkpoint_"  # actual saved files are checkpoint_filename + 'generator.pth' or 'discriminator.pth'
checkpoint_to_load_discriminator = "checkpoint_discriminator_at_epoc_400.pth"  # filename to load
checkpoint_to_load_generator = "checkpoint_generator_at_epoc_400.pth"
do_rgb = False  # whatever to process RGB or grayscale images
generator_loss_type = 'rsgan-gp'  # 'sgan' 'rsgan' 'rsgan-gp' 'rasgan' 'rasgan-gp' 'lsgan' 'ralsgan' 'hingegan' 'rahingegan' 'wgan' 'wgan-gp' 'wgan-div'
discriminator_loss_type = 'rsgan-gp'  # 'sgan' 'rsgan' 'rsgan-gp' 'rasgan' 'rasgan-gp' 'lsgan' 'ralsgan' 'hingegan' 'rahingegan' 'wgan' 'wgan-gp' 'wgan-div'
d_loss_reguralizer = 1.0  # 0.0001 discriminator loss regularizer
g_loss_reguralizer = 1.0  # 0.0001 generator loss regularizer
latent_size = 512
wgan_lambda_gp = 5.0  # reguralizer term for gradient penalty (10.0) or divergence penalty (1.0?)
step_per_epoch = 10  # minimum step per epoch. Actual steps are max between this number and dataset len/batch size
force_use_wasserstein_gradient_penalty = False  # spectral norm disable wasserstein gradient penalty unless you set this to True
force_use_wasserstein_divergence_penalty = False  # spectral norm disable wasserstein divergence penalty unless you set this to True

random_noise_before_discriminator = False  # add random noise to discriminator inputs
random_noise_add_before_discriminator_divisor = 40.0  # divisor to noise variance, increase to attenuate

use_mode_collapse_prevention = True
mode_collapse_lambda = 1.0

# net
num_internal_layers = 2  # number of internal layers for each resolution block in the generator. 2 default.
internal_size_generator = 512
internal_size_discriminator = 512
use_spectral_norm = True  # True globally enable spectral norm
use_spectral_norm_discriminator = True and use_spectral_norm  # True enable spectral norm for the discriminator
use_spectral_norm_generator = False and use_spectral_norm  # True enable spectral norm for the generator
cat_mode = 'simple'  # 'simple' 'complex' multi scale gradient net concatenates images at different scales between generator and discriminator, simple mode just concatenate RGB values, complex mode pass them through 1x1 filter conv
generator_norm_mode = 'batchnorm'  # 'none' 'pixelnorm' 'batchnorm' 'groupnorm' 'syncbatchnorm' 'instancenorm' 'localresponse' 'filterresponse' 'layernorm'
discriminator_norm_mode = 'none'  # 'none' 'pixelnorm' 'batchnorm' 'groupnorm' 'syncbatchnorm' 'instancenorm' 'localresponse' 'filterresponse' 'layernorm'
discriminator_nonlinearity_mode = 'relu'  # 'elu' 'relu' 'selu' 'celu' 'leaky_relu' 'tanh' 'none'
discriminator_use_bias = False  # False internal convs don't use bias. output convs always use bias regardless of this
generator_nonlinearity_mode = 'relu'  # 'elu' 'relu' 'selu' 'celu' 'leaky_relu' 'tanh' 'none'
generator_use_bias = False # False internal convs don't use bias. output convs always use bias regardless of this
latent_layers = 2  # number of fully connected layers from input vector to conv input
use_self_attention = False  # use self attention layer. If set resolution must be at least 64
use_minbatch_std = True  # use minibatch std
generator_conv_type = 'conv'  # 'conv' 'deconv' 'channelwise'  # deconv means Network Deconvolutions not transposed convolutions see https://arxiv.org/abs/1905.11926
discriminator_conv_type = 'conv'  # 'conv' 'deconv' 'channelwise' # deconv means Network Deconvolutions not transposed convolutions see https://arxiv.org/abs/1905.11926

# sgd
optimizer_generator = 'radam'  # 'adamax' 'rmsprop' 'sgd' 'adam' 'radam'
optimizer_discriminator = 'radam'
generator_learning_rate = 0.0001 #  0.0001
discriminator_learning_rate = 0.0002 #  0.0004
adam_beta1 = 0.5
adam_beta2 = 0.999
weight_decay = 0.0
adam_amsgrad = True
eps = 1e-8
rmsprop_alpha = 0.99
momentum = 0.9
rmsprop_centered = True
dampening = 0.0
nesterov = False

# dataset
datasets_location = 'D:/Development/Datasets/'  #  D:\ArtCollection\MiniDataset     'D:/Development/Datasets/'
dataset_artist_dir = 'giger/'  # 'giger/' 'ZdzislawBeksinski/' 'mnist_png/training/' 'img_align_celeba/'
preload_dataset = True
dataset_nonpreserving_scale = True
dataset_center_crop_not_random = False  # if True use center crop, not random. Useful for img_align_celeba
dataset_workers = 8

# visualization
use_visdom = True
vis_title = 'dbxGAN'
sample_to_viz = 4
viz_plotter = VisdomPlotter()

# help functions

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_device():
    global use_cuda
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            use_cuda = False
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("device={}".format(device))
    if use_cuda:
        cudnn.enabled = False
        cudnn.benchmark = False
    return device

def load_checkpoints(load_checkpoint, net, use_cuda, checkpoint_location, checkpoint_to_load):
    if (load_checkpoint is True) and (use_cuda is True):
        return net.load_state_dict(torch.load(checkpoint_location + checkpoint_to_load))
    if (load_checkpoint is True) and (use_cuda is False):
        return net.load_state_dict(torch.load(checkpoint_location + checkpoint_to_load, map_location='cpu'))
    return net

def select_optimizer(optimizer, net, learning_rate):
    global adam_beta1, adam_beta2, weight_decay, rmsprop_alpha, momentum, rmsprop_centered, \
        adam_amsgrad, nesterov, dampening

    if optimizer == 'adamax':
        opt = optim.Adamax(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,
                           betas=(adam_beta1, adam_beta2),
                           weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, alpha=rmsprop_alpha,
                            weight_decay=weight_decay,
                            momentum=momentum, centered=rmsprop_centered)
    elif optimizer == 'adam':
        opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,
                         betas=(adam_beta1, adam_beta2), weight_decay=weight_decay,
                         amsgrad=adam_amsgrad, eps=eps)
    elif optimizer == 'radam':
        opt = new_optim.RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,
            betas=(adam_beta1, adam_beta2), eps=eps, weight_decay=weight_decay)
    else:  # sgd
        opt = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
    return opt

    #if optimizer == 'adamax':
    #    opt = optim.Adamax(net.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2),
    #                       weight_decay=weight_decay)
    #elif optimizer == 'rmsprop':
    #    opt = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=rmsprop_alpha, weight_decay=weight_decay,
    #                        momentum=momentum, centered=rmsprop_centered)
    #elif optimizer == 'adam':
    #    opt = optim.Adam(net.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay,
    #                     amsgrad=adam_amsgrad, eps=eps)
    #else:  # sgd
    #    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
    #                    weight_decay=weight_decay, nesterov=nesterov)
    #return opt

def generate_image_array(image, resolution):
    arr = [image]
    levels = int(np.log2(resolution)) - 2
    for i in range(levels):
        arr.insert(0, nn.AvgPool2d(2)(arr[0]))
    return arr

def generate_image_array_variable(image, resolution):
    arr = [torch.autograd.Variable(image, requires_grad=True)]
    levels = int(np.log2(resolution)) - 2
    for i in range(levels):
        arr.insert(0,torch.autograd.Variable(nn.AvgPool2d(2)(arr[0]), requires_grad=True))
    return arr

def inf_data_gen(dataloader):
    while True:
        for images_batch in dataloader:
            yield images_batch

# program

'''
def discriminator_loss_real(batch_size, output_real, device):
    global discriminator_loss_type
    # loss on real data
    if discriminator_loss_type == 'hinge':
        disc_loss_real = nn.functional.relu(1.0 - output_real).mean()
    elif discriminator_loss_type == 'wasserstein' or discriminator_loss_type == 'wasserstein-gp' or discriminator_loss_type == 'wasserstein-div':
        disc_loss_real = -output_real.mean()
    elif discriminator_loss_type == 'mse':
        gt_real = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)  # 0.6 1.0
        gt_real = gt_real.to(device)
        disc_loss_real = nn.functional.mse_loss(torch.sigmoid(output_real), gt_real)
    else:  # bce
        gt_real = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)  # 0.6 1.0
        gt_real = gt_real.to(device)
        disc_loss_real = nn.BCEWithLogitsLoss()(output_real, gt_real)
    return disc_loss_real

def discriminator_loss_fake(batch_size, output_fake, device):
    global discriminator_loss_type
    if discriminator_loss_type == 'hinge':
        disc_loss_fake = nn.functional.relu(1.0 + output_fake).mean()
    elif discriminator_loss_type == 'wasserstein' or discriminator_loss_type == 'wasserstein-gp' or discriminator_loss_type == 'wasserstein-div':
        disc_loss_fake = output_fake.mean()
    elif discriminator_loss_type == 'mse':
        gt_fake = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2)  # -1.0 -0.6
        gt_fake = gt_fake.to(device)
        disc_loss_fake = nn.functional.mse_loss(torch.sigmoid(output_fake), gt_fake)
    else:  # bce
        gt_fake = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2)  # -1.0 -0.6
        gt_fake = gt_fake.to(device)
        disc_loss_fake = nn.BCEWithLogitsLoss()(output_fake, gt_fake)
    return disc_loss_fake
'''
def discriminator_loss(batch_size, output_real, output_fake, device):
    global discriminator_loss_type
    # loss on real data
    if discriminator_loss_type == 'sgan':
        disc_loss = -torch.log(torch.sigmoid(output_real)).mean() - torch.log(1.0 - torch.sigmoid(output_fake)).mean()
    elif discriminator_loss_type == 'rsgan' or discriminator_loss_type == 'rsgan-gp':
        disc_loss = -torch.log(torch.sigmoid(output_real-output_fake)).mean()
    elif discriminator_loss_type == 'rasgan' or discriminator_loss_type == 'rasgan-gp':
        dxr = torch.sigmoid(output_real - output_fake.mean())
        dxf = torch.sigmoid(output_fake - output_real.mean())
        disc_loss = -torch.log(dxr).mean() - torch.log(1.0-dxf).mean()
    elif discriminator_loss_type == 'lsgan':
        zero = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2)  # -1.0 -0.6
        zero = zero.to(device)
        one = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)  # 0.6 1.0
        one = one.to(device)
        disc_loss = ((output_real - zero) ** 2).mean() + ((output_fake - one) ** 2).mean()
    elif discriminator_loss_type == 'ralsgan':
        one = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)  # 0.6 1.0
        one = one.to(device)
        disc_loss = ((output_real - output_fake.mean() - one) ** 2).mean() + ((output_fake - output_real.mean() + one) ** 2).mean()
    elif discriminator_loss_type == 'hingegan':
        disc_loss = torch.relu(1.0 - output_real).mean() + torch.relu(1.0 + output_fake).mean()
    elif discriminator_loss_type == 'rahingegan':
        dxr = output_real - output_fake.mean()
        dxf = output_fake - output_real.mean()
        disc_loss = torch.relu(1.0 - dxr).mean() + torch.relu(1.0 + dxf).mean()
    elif discriminator_loss_type == 'wgan' or discriminator_loss_type == 'wgan-gp' or discriminator_loss_type == 'wgan-div':
        disc_loss = output_fake.mean() - output_real.mean()
    return disc_loss

def generator_loss(batch_size, output_fake_gen, output_real, device):
    global generator_loss_type
    if generator_loss_type == 'sgan':
        gen_loss = -torch.log(torch.sigmoid(output_fake_gen)).mean()
    elif generator_loss_type == 'rsgan' or discriminator_loss_type == 'rsgan-gp':
        gen_loss = -torch.log(torch.sigmoid(output_fake_gen-output_real)).mean()
    elif generator_loss_type == 'rasgan' or discriminator_loss_type == 'rasgan-gp':
        dxr = torch.sigmoid(output_real - output_fake_gen.mean())
        dxf = torch.sigmoid(output_fake_gen - output_real.mean())
        gen_loss = -torch.log(dxf).mean() - torch.log(1.0-dxr).mean()
    elif generator_loss_type == 'lsgan':
        zero = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2)  # -1.0 -0.6
        zero = zero.to(device)
        gen_loss = ((output_fake_gen - zero) ** 2).mean()
    elif generator_loss_type == 'ralsgan':
        one = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)  # 0.6 1.0
        one = one.to(device)
        gen_loss = ((output_fake_gen - output_real.mean() - one) ** 2).mean() + ((output_real - output_fake_gen.mean() + one) ** 2).mean()
    elif generator_loss_type == 'hingegan':
        gen_loss = -output_fake_gen.mean()
    elif generator_loss_type == 'rahingegan':
        dxr = output_real - output_fake_gen.mean()
        dxf = output_fake_gen - output_real.mean()
        gen_loss = torch.relu(1.0 - dxf).mean() + torch.relu(1.0 + dxr).mean()
    elif generator_loss_type == 'wgan' or discriminator_loss_type == 'wgan-gp' or discriminator_loss_type == 'wgan-div':
        gen_loss = -output_fake_gen.mean()
    return gen_loss


def train_epoch(dataset_artist, total_num_batches, epoch, generator, discriminator, device, discriminator_optimizer, generator_optimizer):
    global batch_size, checkpoint_location, checkpoint_filename, dataset_workers, wgan_lambda_gp, \
        aggregated_batch_size, d_loss_reguralizer, latent_size, g_loss_reguralizer, resolution, \
        random_noise_add_before_discriminator_divisor, generator_loss_type, discriminator_loss_type, \
        random_noise_before_discriminator, sample_to_viz, viz_plotter, force_use_wasserstein_gradient_penalty, \
        force_use_wasserstein_divergence_penalty, use_autoencoder_constraint, use_mode_collapse_prevention, \
        mode_collapse_lambda

    last_time = time.time()

    # loss accumulators
    epoch_generator_loss = 0.0
    epoch_discriminator_loss = 0.0

    # dataset management
    dataset_artist_iterator = inf_data_gen(dataset_artist)

    discriminator.train()
    generator.train()

    global_z = torch.randn(sample_to_viz, latent_size).to(device)  # common latent for visualization

    for num in range(total_num_batches):
        discriminator.zero_grad()
        generator.zero_grad()
        running_discriminator_loss = 0.0
        for i in range(int(aggregated_batch_size)):
            artist_data = next(dataset_artist_iterator)
            artist_image = artist_data[0]
            artist_image = artist_image.to(device)
            artist_images = generate_image_array_variable(artist_image, resolution)

            z = torch.randn(artist_image.shape[0], latent_size).to(device)

            # train the discriminator

            if random_noise_before_discriminator is True:
                for j in range(len(artist_images)-4):
                    rnd = torch.randn(artist_images[j+4].shape).to(device) / random_noise_add_before_discriminator_divisor
                    artist_images[j+4] = artist_images[j+4] + rnd

            output_real = discriminator(artist_images)  # run the discriminator with real data

            if discriminator_loss_type == 'wgan-div' or force_use_wasserstein_divergence_penalty:
                fakes = generator(z)  # need gradients
            else:
                with torch.no_grad():
                    fakes = generator(z)  # generate fake
                for j in range(len(fakes)):
                    fakes[j] = fakes[j].detach()

            if random_noise_before_discriminator is True:
                for j in range(len(fakes)):
                    rnd = torch.randn(fakes[j].shape).to(device) / (random_noise_add_before_discriminator_divisor*(len(fakes)-j))
                    fakes[j] = fakes[j] + rnd

            output_fake = discriminator(fakes)  # run the discriminator on fake data.

            # loss on real data
            #disc_loss_real = discriminator_loss_real(artist_image.shape[0], output_real, device)
            #disc_loss_fake = discriminator_loss_fake(artist_image.shape[0], output_fake, device)
            #disc_loss_real = d_loss_reguralizer * (disc_loss_real / float(aggregated_batch_size))
            #disc_loss_fake = d_loss_reguralizer * (disc_loss_fake / float(aggregated_batch_size))

            disc_loss = discriminator_loss(artist_image.shape[0], output_real, output_fake, device)
            disc_loss = d_loss_reguralizer * (disc_loss / float(aggregated_batch_size))

            #running_discriminator_loss = running_discriminator_loss + disc_loss_fake.item() + disc_loss_real.item()
            running_discriminator_loss = running_discriminator_loss + disc_loss.item()

            # Compute gradient penalty
            d1_loss = 0.0
            d2_loss = 0.0
            if ((discriminator_loss_type == 'wgan-gp' or discriminator_loss_type == 'rsgan-gp' or discriminator_loss_type == 'rasgan-gp') and use_spectral_norm_discriminator is not True) or force_use_wasserstein_gradient_penalty is True:
                batch_size = artist_image.size(0)
                epsilon = torch.rand(batch_size, 1, 1, 1)
                epsilon = epsilon.expand_as(artist_image)
                epsilon = epsilon.to(device)
                interpolation = epsilon * artist_image.data + (1.0 - epsilon) * fakes[-1].data
                interpolateds = generate_image_array_variable(interpolation, resolution)
                out = discriminator(interpolateds)
                gradients = torch.autograd.grad(outputs=out,
                                           inputs=interpolateds,
                                           grad_outputs=torch.ones(out.size()).to(device),
                                           retain_graph=True,
                                           create_graph=True)[0]  # only_inputs=True allow_unused=True
                gradients = gradients.view(batch_size, -1)
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                d_loss_gp = ((gradients_norm - 1) ** 2).mean()
                d1_loss = wgan_lambda_gp * d_loss_gp
                d1_loss = d1_loss / aggregated_batch_size
                running_discriminator_loss = running_discriminator_loss + d1_loss.item()
            if (discriminator_loss_type == 'wgan-div' and use_spectral_norm_discriminator is not True) or force_use_wasserstein_divergence_penalty is True:
                k = 2
                p = 6
                real_gradients = torch.autograd.grad(outputs=output_real,
                                                inputs=artist_images,
                                                grad_outputs=torch.ones(output_real.size()).to(device),
                                                retain_graph=True,
                                                create_graph=True,
                                                only_inputs=True)[0]  # only_inputs=True allow_unused=True
                real_grad_norm = real_gradients.view(real_gradients.size(0), -1).pow(2).sum(1) ** (p / 2)
                fake_gradients = torch.autograd.grad(outputs=output_fake,
                                                 inputs=fakes,
                                                 grad_outputs=torch.ones(output_fake.size()).to(device),
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 only_inputs=True)[0]  # only_inputs=True allow_unused=True
                fake_grad_norm = fake_gradients.view(fake_gradients.size(0), -1).pow(2).sum(1) ** (p / 2)
                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
                d2_loss = wgan_lambda_gp * (div_gp / aggregated_batch_size)
                running_discriminator_loss = running_discriminator_loss + d2_loss.item()

            #total_loss = disc_loss_real + disc_loss_fake + d1_loss + d2_loss
            total_loss = disc_loss + d1_loss + d2_loss
            total_loss.backward()

        discriminator_optimizer.step()

        # Weight clipping
        if discriminator_loss_type == 'wgan' and use_spectral_norm_discriminator is not True:
           for p in discriminator.parameters():
               p.data.clamp_(-0.001, 0.001)

        discriminator.zero_grad()
        generator.zero_grad()
        running_generator_loss = 0.0
        for i in range(int(aggregated_batch_size)):
            # train the generator
            if generator_loss_type == 'rsgan' or discriminator_loss_type == 'rsgan-gp' or \
                    generator_loss_type == 'rasgan' or discriminator_loss_type == 'rasgan-gp' or \
                    generator_loss_type == 'ralsgan' or generator_loss_type == 'rahingegan':
                artist_data = next(dataset_artist_iterator)
                artist_image = artist_data[0]
                artist_image = artist_image.to(device)
                artist_images = generate_image_array_variable(artist_image, resolution)
                if random_noise_before_discriminator is True:
                    for j in range(len(artist_images)-4):
                        rnd = torch.randn(artist_images[j+4].shape).to(device) / random_noise_add_before_discriminator_divisor
                        artist_images[j+4] = artist_images[j+4] + rnd
                output_real = discriminator(artist_images)  # run the discriminator with real data

            z = torch.randn(artist_image.shape[0], latent_size).to(device)
            fakes = generator(z)
            if random_noise_before_discriminator is True:
                for j in range(len(fakes)):
                    rnd = torch.randn(fakes[j].shape).to(device) / (random_noise_add_before_discriminator_divisor*(len(fakes)-j))
                    fakes[j] = fakes[j] + rnd
            output_fake_gen = discriminator(fakes)

            if generator_loss_type == 'rsgan' or discriminator_loss_type == 'rsgan-gp' or \
                    generator_loss_type == 'rasgan' or discriminator_loss_type == 'rasgan-gp' or \
                    generator_loss_type == 'ralsgan' or generator_loss_type == 'rahingegan':
                gen_loss = generator_loss(artist_image.shape[0], output_fake_gen, output_real, device)
            else:
                gen_loss = generator_loss(artist_image.shape[0], output_fake_gen, None, device)

            if use_mode_collapse_prevention is True:
                z2 = torch.randn(artist_image.shape[0], latent_size).to(device)
                fakes2 = generator(z2)
                mode_loss = 0
                for j in range(len(fakes2)):
                    mode_loss = mode_loss + torch.relu(((fakes2[j] - fakes[j]) ** 2).mean() / ((z2-z)**2).mean())
                gen_loss = gen_loss - mode_collapse_lambda * mode_loss/float(len(fakes2))

            gen_loss = g_loss_reguralizer * (gen_loss / float(aggregated_batch_size))

            gen_loss.backward()

            running_generator_loss = running_generator_loss + gen_loss.item()

        generator_optimizer.step()

        time_spent = time.time() - last_time
        batch_time = time_spent/(num+1)
        time_to_finish = batch_time * (total_num_batches - num)
        print("epoch {} of {} at {}%, to finish epoch: {}".format(epoch, num_epochs, (num / total_num_batches) * 100,
                                                                  str(datetime.timedelta(seconds=time_to_finish))))
        print("generator loss {}, discriminator loss {}".format(running_generator_loss, running_discriminator_loss))

        epoch_generator_loss += running_generator_loss
        epoch_discriminator_loss += running_discriminator_loss
        if use_visdom:
            viz_plotter.plot_line("running_generator_loss", running_generator_loss, "Generator Loss")
            viz_plotter.plot_line("running_discriminator_loss", running_discriminator_loss, "Discriminator Loss")
        if (num % 10) == 9:
            if use_visdom:
                generator.eval()
                with torch.no_grad():
                    #z = torch.randn(artist_image.shape[0], latent_size).to(device)
                    fakes = generator(global_z)
                    viz_plotter.plot_images("fake_images_{}".format(resolution), fakes[-1].cpu(), "Generated Images {}".format(resolution), False)
                    viz_plotter.plot_images("fake_images_{}".format(resolution//2), fakes[-2].cpu(), "Generated Images {}".format(resolution//2), False)
                    viz_plotter.plot_images("fake_images_{}".format(resolution//4), fakes[-3].cpu(), "Generated Images {}".format(resolution//4), False)
                    viz_plotter.plot_images("fake_images_{}".format(resolution//8), fakes[-4].cpu(), "Generated Images {}".format(resolution//8), False)
                    viz_plotter.plot_images("real_images_{}".format(resolution), artist_images[-1][0:sample_to_viz,:,:,:].cpu(), "Real Artist Images {}".format(resolution), False)
                    viz_plotter.plot_images("real_images_{}".format(resolution//2), artist_images[-2][0:sample_to_viz,:,:,:].cpu(), "Real Artist Images {}".format(resolution//2), False)
                    viz_plotter.plot_images("real_images_{}".format(resolution//4), artist_images[-3][0:sample_to_viz,:,:,:].cpu(), "Real Artist Images {}".format(resolution//4), False)
                    viz_plotter.plot_images("real_images_{}".format(resolution//8), artist_images[-4][0:sample_to_viz,:,:,:].cpu(), "Real Artist Images {}".format(resolution//8), False)
                generator.train()
            torch.save(generator.state_dict(), checkpoint_location + checkpoint_filename + 'generator.pth')
            torch.save(discriminator.state_dict(), checkpoint_location + checkpoint_filename + 'discriminator.pth')

    epoch_generator_loss = epoch_generator_loss / total_num_batches
    epoch_discriminator_loss = epoch_discriminator_loss / total_num_batches
    dataset_artist_iterator.__del__()

    return epoch_generator_loss, epoch_discriminator_loss

def run():
    global datasets_location, dataset_artist_dir, preload_dataset, vis_title, use_visdom, use_cuda, do_rgb, \
        num_epochs, load_checkpoint, checkpoint_location, checkpoint_to_load_discriminator, \
        checkpoint_to_load_generator, generator_learning_rate, discriminator_learning_rate, adam_beta1, adam_beta2, \
        weight_decay, adam_amsgrad, eps, batch_size, dataset_workers, cat_mode, discriminator_norm_mode, \
        aggregated_batch_size, latent_size, use_self_attention, generator_norm_mode, resolution, \
        discriminator_nonlinearity_mode, discriminator_use_bias, generator_nonlinearity_mode, generator_use_bias, \
        latent_layers, dataset_center_crop_not_random, step_per_epoch, generator_conv_type, discriminator_conv_type, \
        viz_plotter, num_internal_layers, dataset_nonpreserving_scale, optimizer_discriminator, optimizer_generator, \
        internal_size_generator, internal_size_discriminator, use_spectral_norm_generator, \
        use_spectral_norm_discriminator, load_checkpoint

    if do_rgb:
        image_channels = 3
    else:
        image_channels = 1

    #select GPU if enabled and available
    device = set_device()

    vis_title = vis_title + ' on ' + dataset_artist_dir

    # create dataset
    dataset_artist = ImageDatasetFolder(datasets_location + dataset_artist_dir, start_size=resolution, do_rgb=do_rgb, preload=preload_dataset, center_crop=dataset_center_crop_not_random,
                                        nonpreserving_scale=dataset_nonpreserving_scale)
    dataset_artist_loader = torchdata.DataLoader(dataset_artist, batch_size=batch_size, shuffle=True, num_workers=dataset_workers, pin_memory=True)

    total_num_batches = max(step_per_epoch, len(dataset_artist_loader))
    # create nets
    discriminator = MultiScaleDiscriminator(image_channels, resolution, 1, use_spectral_norm_discriminator, discriminator_conv_type,
                                            discriminator_norm_mode, cat_mode, use_self_attention, use_minbatch_std,
                                            discriminator_nonlinearity_mode, discriminator_use_bias, internal_size_discriminator).to(device)
    generator = MultiScaleGenerator(latent_size, latent_layers, resolution, image_channels, use_spectral_norm_generator, generator_conv_type,
                                    generator_norm_mode, use_self_attention, generator_nonlinearity_mode,
                                    generator_use_bias, num_internal_layers, internal_size_generator).to(device)

    # load latest checkpoint
    load_checkpoints(load_checkpoint, discriminator, use_cuda, checkpoint_location, checkpoint_to_load_discriminator)
    load_checkpoints(load_checkpoint, generator, use_cuda, checkpoint_location, checkpoint_to_load_generator)
    # optimizers
    discriminator_optimizer = select_optimizer(optimizer_discriminator, discriminator, discriminator_learning_rate)
    generator_optimizer = select_optimizer(optimizer_generator, generator, generator_learning_rate)
    # learning rate schedulers
    #scheduler_d = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=learning_rate_scheduler_step_size,
    #                                              gamma=learning_rate_scheduler_gamma)
    #scheduler_g = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=learning_rate_scheduler_step_size,
    #                                              gamma=learning_rate_scheduler_gamma)
    for epoch in range(num_epochs):
        epoch_generator_loss, epoch_discriminator_loss = train_epoch(dataset_artist=dataset_artist_loader,
                                                                     total_num_batches=total_num_batches,
                                                                     epoch=epoch,
                                                                     generator=generator,
                                                                     discriminator=discriminator,
                                                                     device=device,
                                                                     discriminator_optimizer=discriminator_optimizer,
                                                                     generator_optimizer=generator_optimizer)

        print("EPOCH generator loss {}, discriminator loss {}".format(epoch_generator_loss, epoch_discriminator_loss))
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), checkpoint_location + checkpoint_filename + 'generator_at_epoc_{}.pth'.format(epoch))
            torch.save(discriminator.state_dict(), checkpoint_location + checkpoint_filename + 'discriminator_at_epoc_{}.pth'.format(epoch))
        if use_visdom:
            if epoch_discriminator_loss>2.0:
                epoch_discriminator_loss = 2.0
            if epoch_generator_loss>2.0:
                epoch_generator_loss = 2.0
            viz_plotter.plot_line("epoch_discriminator_loss", epoch_discriminator_loss,"Epoch Discriminator Loss", do_clip_vals=False)
            viz_plotter.plot_line("epoch_generator_loss", epoch_generator_loss, "Epoch Generator Loss", do_clip_vals=False)

def test():
    global use_cuda, do_rgb, latent_size, use_self_attention, generator_norm_mode, resolution, \
        generator_nonlinearity_mode, generator_use_bias, latent_layers, generator_conv_type, \
        num_internal_layers, internal_size_generator, use_spectral_norm_generator
    if do_rgb:
        image_channels = 3
    else:
        image_channels = 1

    #select GPU if enabled and available
    device = set_device()

    generator = MultiScaleGenerator(latent_size, latent_layers, resolution, image_channels, use_spectral_norm_generator,
                                    generator_conv_type,
                                    generator_norm_mode, use_self_attention, generator_nonlinearity_mode,
                                    generator_use_bias, num_internal_layers, internal_size_generator).to(device)

    load_checkpoints(True, generator, use_cuda, "./temp/", "checkpoint_generator_at_epoc_100.pth")

    generator.eval()

    for num in range(500):
        z = torch.randn(1, latent_size).to(device)
        with torch.no_grad():
            fakes = generator(z)
            img = fakes[-1].cpu()[0]
            img = (img + 1.0) / 2.0
            img = torch.clamp(img, 0.0, 1.0)
            save_image(img, './images/img_{}.png'.format(num))

if __name__ == "__main__":
    run()

'''
    if use_autoencoder_constraint is True:
        degenerator = DeGenerator(image_channels, resolution, latent_size, use_spectral_norm_generator,
                                  generator_conv_type,
                                  generator_norm_mode, generator_nonlinearity_mode, generator_use_bias,
                                  internal_size_generator).to(device)
        load_checkpoints(load_checkpoint, degenerator, use_cuda, checkpoint_location, checkpoint_to_load_degenerator)
        degenerator_optimizer = new_optim.RAdam(itertools.chain(generator.parameters(), degenerator.parameters()), lr=generator_learning_rate,
            betas=(adam_beta1, adam_beta2), eps=eps, weight_decay=weight_decay)
    else:
        degenerator = None
        degenerator_optimizer = None
        '''
'''if use_autoencoder_constraint is True:
            discriminator.zero_grad()
            generator.zero_grad()
            degenerator.zero_grad()
            for i in range(int(aggregated_batch_size)):
                artist_data = next(dataset_artist_iterator)
                artist_image = artist_data[0]
                artist_image = artist_image.to(device)
                real_latent = degenerator(artist_image)
                real_reconstruction = generator(real_latent)

                z = torch.randn(artist_image.shape[0], latent_size).to(device)
                fakes = generator(z)
                fakes_latent = degenerator(fakes[-1])

                ae_loss = ((artist_image-real_reconstruction[-1]) ** 4).mean() + ((z-fakes_latent) ** 2).mean()
                ae_loss = ae_loss / float(aggregated_batch_size)
                ae_loss.backward()
            degenerator_optimizer.step()
        else:'''