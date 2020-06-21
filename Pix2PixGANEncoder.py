import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
import numpy as np
import torch_optimizer as new_optim

from StyleTransferNet import DBX_UNET
from StyleTransferNet import DBX_Discriminator
from StyleTransferNet import DBX_SmallerDiscriminator
from ImageDataset import ImageDatasetFolder
from Visualization import VisdomPlotter

import datetime
import time
import os
from random import randint

# general
image_size = 128
batch_size = 4  # real gpu batch size
aggregated_batch_size = 4  # simulated batch size (if the real batch size you want doesn't fit in memory)
use_cuda = True
steps_per_epoch = 1000
num_epochs = 10000
load_checkpoint = False
checkpoint_location = "./temp/"
checkpoint_filename = "checkpoint_"
checkpoint_to_load_discriminator_T = ""
checkpoint_to_load_discriminator_S = ""
checkpoint_to_load_generator = ""

discriminator_loss_type = 'rsgan-gp'
generator_loss_type = 'rsgan-gp'
enalbe_gradient_penalty = True
lambda_gp = 1.0
lambda_gan = 1.0
lambda_autoenc = 1.0
lambda_cyc = 1.0
lambda_sem = 0.1

# net
discriminator_levels = 4
unet_levels = 4

# sgd
optimizer = 'adam'  # 'adamax' 'rmsprop' 'sgd' 'adam'
generator_learning_rate = 0.001
discriminator_learning_rate = 0.004
adam_beta1 = 0.5
adam_beta2 = 0.99
weight_decay = 0.0
adam_amsgrad = True
eps = 1e-8
rmsprop_alpha = 0.99
momentum = 0.9
rmsprop_centered = True
dampening = 0.0
nesterov = False

# dataset
datasets_location = 'D:/Development/Datasets/'
dataset_source_dir = 'img_align_celeba/'
dataset_target_dir = 'animefaces/'
preload_dataset = False
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

def set_device(use_cuda):
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
        cudnn.enabled = True
        cudnn.benchmark = True
    return device

def load_checkpoints(net, use_cuda, checkpoint_location, checkpoint_to_load):
    global load_checkpoint
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

def discriminator_loss(output_real, output_fake, device):
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
        zero = torch.FloatTensor(output_real.shape[0], 1).uniform_(0.0, 0.2)  # -1.0 -0.6
        zero = zero.to(device)
        one = torch.FloatTensor(output_real.shape[0], 1).uniform_(0.8, 1.0)  # 0.6 1.0
        one = one.to(device)
        disc_loss = ((output_real - zero) ** 2).mean() + ((output_fake - one) ** 2).mean()
    elif discriminator_loss_type == 'ralsgan':
        one = torch.FloatTensor(output_real.shape[0], 1).uniform_(0.8, 1.0)  # 0.6 1.0
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

def lsgan_discriminator_loss(output_real, output_fake, device):
    zero = torch.FloatTensor(output_real.shape[0], 1).uniform_(0.0, 0.2)  # -1.0 -0.6
    zero = zero.to(device)
    one = torch.FloatTensor(output_fake.shape[0], 1).uniform_(0.8, 1.0)  # 0.6 1.0
    one = one.to(device)
    disc_loss = ((torch.sigmoid(output_real) - zero) ** 2).mean() + ((torch.sigmoid(output_fake) - one) ** 2).mean()
    return disc_loss

def generator_loss(output_fake, output_real, device):
    global generator_loss_type
    if generator_loss_type == 'sgan':
        gen_loss = -torch.log(torch.sigmoid(output_fake)).mean()
    elif generator_loss_type == 'rsgan' or discriminator_loss_type == 'rsgan-gp':
        gen_loss = -torch.log(torch.sigmoid(output_fake-output_real)).mean()
    elif generator_loss_type == 'rasgan' or discriminator_loss_type == 'rasgan-gp':
        dxr = torch.sigmoid(output_real - output_fake.mean())
        dxf = torch.sigmoid(output_fake - output_real.mean())
        gen_loss = -torch.log(dxf).mean() - torch.log(1.0-dxr).mean()
    elif generator_loss_type == 'lsgan':
        zero = torch.FloatTensor(output_fake.shape[0], 1).uniform_(0.0, 0.2)  # -1.0 -0.6
        zero = zero.to(device)
        gen_loss = ((output_fake - zero) ** 2).mean()
    elif generator_loss_type == 'ralsgan':
        one = torch.FloatTensor(output_fake.shape[0], 1).uniform_(0.8, 1.0)  # 0.6 1.0
        one = one.to(device)
        gen_loss = ((output_fake - output_real.mean() - one) ** 2).mean() + ((output_real - output_fake.mean() + one) ** 2).mean()
    elif generator_loss_type == 'hingegan':
        gen_loss = -output_fake.mean()
    elif generator_loss_type == 'rahingegan':
        dxr = output_real - output_fake.mean()
        dxf = output_fake - output_real.mean()
        gen_loss = torch.relu(1.0 - dxf).mean() + torch.relu(1.0 + dxr).mean()
    elif generator_loss_type == 'wgan' or discriminator_loss_type == 'wgan-gp' or discriminator_loss_type == 'wgan-div':
        gen_loss = -output_fake.mean()
    return gen_loss

def lsgan_generator_loss(output_fake, device):
    zero = torch.FloatTensor(output_fake.shape[0], 1).uniform_(0.0, 0.2)  # -1.0 -0.6
    zero = zero.to(device)
    gen_loss = ((torch.sigmoid(output_fake) - zero) ** 2).mean()
    return gen_loss

def l1_loss(A, B):
    loss = (torch.abs(A-B)).mean()
    return loss

def l2_loss(A, B):
    loss = ((A-B) ** 2).mean()
    return loss

def l4_loss(A, B):
    loss = ((A-B) ** 4).mean()
    return loss

def build_sample(size, dataset):
    imgid = randint(0,dataset.__len__()-1)
    tensor, _ = dataset.__getitem__(imgid)
    tensor = tensor.view(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])
    for i in range(size-1):
        imgid = randint(0, dataset.__len__() - 1)
        temp, _ = dataset.__getitem__(imgid)
        temp = temp.view(1, temp.shape[0], temp.shape[1], temp.shape[2])
        tensor = torch.cat([tensor, temp], 0)
    return tensor

def get_batch(dataset_source_loader, dataset_target_loader):
    while True:
        try:
            real_S, _ = get_batch.source_iter.next()
        except:
            get_batch.source_iter = iter(dataset_source_loader)
            real_S, _ = get_batch.source_iter.next()

        try:
            real_T, _ = get_batch.target_iter.next()
        except:
            get_batch.target_iter = iter(dataset_target_loader)
            real_T, _ = get_batch.target_iter.next()
        if real_T.shape[0] == real_S.shape[0]:
            break;
    return real_S, real_T
get_batch.source_iter = None
get_batch.target_iter = None

def compute_gradient_penalty(input_real, input_generated, discriminator, device):
    global enalbe_gradient_penalty, lambda_gp
    gp_loss = 0.0
    if enalbe_gradient_penalty is True:
        batch_size = input_real.shape[0]
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(input_real)
        epsilon = epsilon.to(device)
        interpolation = epsilon * input_real.data + (1.0 - epsilon) * input_generated.data
        interpolateds = torch.autograd.Variable(interpolation, requires_grad=True)
        out = discriminator(interpolateds)
        gradients = torch.autograd.grad(outputs=out,
                                        inputs=interpolateds,
                                        grad_outputs=torch.ones(out.size()).to(device),
                                        retain_graph=True,
                                        create_graph=True)[0]  # only_inputs=True allow_unused=True
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        d_loss_gp = ((gradients_norm - 1) ** 2).mean()
        gp_loss = lambda_gp * d_loss_gp
    return gp_loss

# program

def train_epoch(dataset_source, dataset_target,
                dataset_source_loader, dataset_target_loader,
                iterations, epoch,
                generator, discriminator_S, discriminator_T,
                device,
                discriminator_S_optimizer, discriminator_T_optimizer, generator_optimizer):

    global batch_size, aggregated_batch_size, checkpoint_location, checkpoint_filename, num_epochs, sample_to_viz, \
        lambda_gan, lambda_autoenc, lambda_cyc, lambda_sem, enalbe_gradient_penalty

    last_time = time.time()
    # loss accumulators
    epoch_generator_loss = 0.0
    epoch_discriminator_loss = 0.0

    discriminator_S.train()
    discriminator_T.train()
    generator.train()

    viz_sample_S = build_sample(sample_to_viz, dataset_source).to(device)
    viz_sample_T = build_sample(sample_to_viz, dataset_target).to(device)

    for num in range(iterations):
        # update D
        discriminator_S.zero_grad()
        discriminator_T.zero_grad()
        generator.zero_grad()
        running_discriminator_loss = 0.0
        running_disc_loss = 0.0
        running_gp_loss_target = 0.0
        running_gp_loss_source = 0.0
        for i in range(int(aggregated_batch_size)):
            real_S, real_T = get_batch(dataset_source_loader, dataset_target_loader)

            real_S, real_T = real_S.to(device), real_T.to(device)

            real_S_encoded, real_S_outs = generator.forward_encoder_S(real_S)
            #reconstructed_S = generator.forward_decoder_S(real_S_encoded, real_S_outs)
            generated_T = generator.forward_decoder_T(real_S_encoded, real_S_outs)

            real_T_encoded, real_T_outs = generator.forward_encoder_T(real_T)
            #reconstructed_T = generator.forward_decoder_T(real_T_encoded, real_T_outs)
            generated_S = generator.forward_decoder_S(real_T_encoded, real_T_outs)

            output_D_real_S = discriminator_S(real_S)
            output_D_real_T = discriminator_T(real_T)
            output_D_generated_S = discriminator_S(generated_S)
            output_D_generated_T = discriminator_T(generated_T)

            # compute D losses
            discriminator_S_loss = discriminator_loss(output_D_real_S, output_D_generated_S, device)
            discriminator_T_loss = discriminator_loss(output_D_real_T, output_D_generated_T, device)
            loss = (discriminator_S_loss + discriminator_T_loss) / float(aggregated_batch_size)
            loss.backward()
            running_disc_loss += loss.item()
            running_discriminator_loss += loss.item()

            # Compute gradient penalty
            gp_loss_target = compute_gradient_penalty(real_T, generated_T, discriminator_T, device) / aggregated_batch_size
            gp_loss_source = compute_gradient_penalty(real_S, generated_S, discriminator_S, device) / aggregated_batch_size
            loss = gp_loss_target + gp_loss_source
            loss.backward()

            running_gp_loss_target += gp_loss_target.item()
            running_gp_loss_source += gp_loss_source.item()
            running_discriminator_loss += loss.item()

        discriminator_S_optimizer.step()
        discriminator_T_optimizer.step()

        # update G
        discriminator_S.zero_grad()
        discriminator_T.zero_grad()
        generator.zero_grad()
        running_generator_loss = 0.0
        running_gan_loss = 0.0
        running_autoencoder_loss = 0.0
        running_cycle_loss = 0.0
        running_sem_loss = 0.0
        for i in range(int(aggregated_batch_size)):
            real_S, real_T = get_batch(dataset_source_loader, dataset_target_loader)

            real_S, real_T = real_S.to(device), real_T.to(device)

            real_S_encoded, real_S_outs = generator.forward_encoder_S(real_S)
            reconstructed_S = generator.forward_decoder_S(real_S_encoded, real_S_outs)
            generated_T = generator.forward_decoder_T(real_S_encoded, real_S_outs)
            generated_T_encoded, generated_T_outs = generator.forward_encoder_T(generated_T)
            cycle_generated_S = generator.forward_decoder_S(generated_T_encoded, generated_T_outs)

            real_T_encoded, real_T_outs = generator.forward_encoder_T(real_T)
            reconstructed_T = generator.forward_decoder_T(real_T_encoded, real_T_outs)
            generated_S = generator.forward_decoder_S(real_T_encoded, real_T_outs)
            generated_S_encoded, generated_S_outs = generator.forward_encoder_S(generated_S)
            cycle_generated_T = generator.forward_decoder_T(generated_S_encoded, generated_S_outs)

            output_D_generated_S = discriminator_S(generated_S)
            output_D_generated_T = discriminator_T(generated_T)

            output_D_real_S = discriminator_S(real_S)
            output_D_real_T = discriminator_T(real_T)

            # compute D losses
            generator_S_loss = generator_loss(output_D_generated_S, output_D_real_S, device)
            generator_T_loss = generator_loss(output_D_generated_T, output_D_real_T, device)
            gan_loss = generator_S_loss + generator_T_loss

            S_autoenc_loss = l1_loss(real_S, reconstructed_S)
            T_eutoenc_loss = l1_loss(real_T, reconstructed_T)
            autoencoder_loss = S_autoenc_loss + T_eutoenc_loss

            S_cyc_loss = l1_loss(real_S, cycle_generated_S)
            T_cyc_loss = l1_loss(real_T, cycle_generated_T)
            cycle_loss = S_cyc_loss + T_cyc_loss

            S_sem_loss = l1_loss(real_S_encoded, generated_T_encoded)
            T_sem_loss = l1_loss(real_T_encoded, generated_S_encoded)
            sem_loss = S_sem_loss + T_sem_loss

            loss = (lambda_gan * gan_loss + lambda_autoenc * autoencoder_loss +
                    lambda_cyc * cycle_loss + lambda_sem * sem_loss) / float(aggregated_batch_size)
            #loss = (gan_loss + cycle_loss + sem_loss) / float(aggregated_batch_size)
            running_generator_loss += loss.item()

            running_gan_loss += gan_loss.item()/float(aggregated_batch_size)
            running_autoencoder_loss += autoencoder_loss.item()/float(aggregated_batch_size)
            running_cycle_loss += cycle_loss.item()/float(aggregated_batch_size)
            running_sem_loss += sem_loss.item()/float(aggregated_batch_size)

            loss.backward()
        generator_optimizer.step()

        time_spent = time.time() - last_time
        batch_time = time_spent/(num+1)
        time_to_finish = batch_time * (iterations - num)
        print("epoch {} of {} at {}%, to finish epoch: {}".format(epoch, num_epochs, (num / iterations) * 100, str(datetime.timedelta(seconds=time_to_finish))))
        print("generator loss {}, discriminator loss {}".format(running_generator_loss, running_discriminator_loss))
        print("disc loss {}, gan loss {}".format(running_disc_loss, running_gan_loss))
        print("autoencoder loss {}, cycle loss {} sem {}".format(running_autoencoder_loss, running_cycle_loss, running_sem_loss))
        print("gp source loss {}, gp terget loss {}".format(running_gp_loss_source, running_gp_loss_target))

        epoch_generator_loss += running_generator_loss
        epoch_discriminator_loss += running_discriminator_loss
        if use_visdom:
            #viz_plotter.plot_line("running_generator_loss", running_generator_loss, "Generator Loss", do_clip_vals=False)
            #viz_plotter.plot_line("running_discriminator_loss", running_discriminator_loss, "Discriminator Loss", do_clip_vals=False)
            viz_plotter.plot_lines("running_losses","Main Losses", iterations*epoch+num, [running_generator_loss, running_discriminator_loss],
                                   ["Generator Loss","Discriminator Loss"],[[255,0,0],[0,0,255]])

            viz_plotter.plot_lines("running_sub_losses", "Sub Losses", iterations * epoch + num,
                                   [running_gan_loss, running_autoencoder_loss, running_cycle_loss, running_sem_loss],
                                   ["Gen Gan Loss", "Autoenc Loss", "Cycle Loss", "Sem Loss"],
                                   [[255, 0, 0], [0, 0, 255], [0, 255, 255], [0, 255, 0]])
        if (num % 10) == 9:
            if use_visdom:
                generator.eval()
                with torch.no_grad():
                    temp_S_encoded, temp_S_outs = generator.forward_encoder_S(viz_sample_S)
                    temp_generated_T = generator.forward_decoder_T(temp_S_encoded, temp_S_outs)
                    temp_T_encoded, temp_T_outs = generator.forward_encoder_T(viz_sample_T)
                    temp_generated_S = generator.forward_decoder_S(temp_T_encoded, temp_T_outs)
                    viz_plotter.plot_images("original_source", viz_sample_S.cpu(), "Real Source Images", False)
                    viz_plotter.plot_images("original_target", viz_sample_T.cpu(), "Real Target Images", False)
                    viz_plotter.plot_images("generated_source", temp_generated_S.cpu(), "Generated Source Images", True)
                    viz_plotter.plot_images("generated_target", temp_generated_T.cpu(), "Generated Target Images", True)
                generator.train()
            torch.save(generator.state_dict(), checkpoint_location + checkpoint_filename + 'generator.pth')
            torch.save(discriminator_S.state_dict(), checkpoint_location + checkpoint_filename + 'discriminator_S.pth')
            torch.save(discriminator_T.state_dict(), checkpoint_location + checkpoint_filename + 'discriminator_T.pth')

    epoch_generator_loss = epoch_generator_loss / iterations
    epoch_discriminator_loss = epoch_discriminator_loss / iterations

    return epoch_generator_loss, epoch_discriminator_loss

def run():
    global datasets_location, dataset_source_dir, dataset_target_dir, preload_dataset, dataset_workers,\
        vis_title, use_visdom, viz_plotter, \
        image_size, batch_size, use_cuda, num_epochs, load_checkpoint, checkpoint_location, steps_per_epoch, \
        checkpoint_to_load_discriminator_S, checkpoint_to_load_discriminator_T, checkpoint_to_load_generator, \
        generator_learning_rate, discriminator_learning_rate, \
        discriminator_levels, unet_levels, checkpoint_filename

    #select GPU if enabled and available
    device = set_device(use_cuda)

    vis_title = vis_title + ' on ' + dataset_target_dir

    # create dataset
    dataset_source = ImageDatasetFolder(datasets_location + dataset_source_dir, start_size=image_size, do_rgb=True,
                                        preload=preload_dataset, center_crop=False, nonpreserving_scale=True)
    dataset_source_loader = torchdata.DataLoader(dataset_source, batch_size=batch_size, shuffle=True,
                                                 num_workers=dataset_workers, pin_memory=True)

    dataset_target = ImageDatasetFolder(datasets_location + dataset_target_dir, start_size=image_size, do_rgb=True,
                                        preload=preload_dataset, center_crop=False, nonpreserving_scale=True)
    dataset_target_loader = torchdata.DataLoader(dataset_target, batch_size=batch_size, shuffle=True,
                                                 num_workers=dataset_workers, pin_memory=True)

    # create nets
    discriminator_S = DBX_SmallerDiscriminator(image_size, discriminator_levels).to(device)
    discriminator_T = DBX_SmallerDiscriminator(image_size, discriminator_levels).to(device)
    generator = DBX_UNET(unet_levels).to(device)
    # load latest checkpoint
    load_checkpoints(discriminator_S, use_cuda, checkpoint_location, checkpoint_to_load_discriminator_S)
    load_checkpoints(discriminator_T, use_cuda, checkpoint_location, checkpoint_to_load_discriminator_T)
    load_checkpoints(generator, use_cuda, checkpoint_location, checkpoint_to_load_generator)
    # optimizers
    discriminator_T_optimizer = select_optimizer(optimizer, discriminator_T, discriminator_learning_rate)
    discriminator_S_optimizer = select_optimizer(optimizer, discriminator_S, discriminator_learning_rate)
    generator_optimizer = select_optimizer(optimizer, generator, generator_learning_rate)
    # learning rate schedulers
    #scheduler_d = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=learning_rate_scheduler_step_size,
    #                                              gamma=learning_rate_scheduler_gamma)
    #scheduler_g = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=learning_rate_scheduler_step_size,
    #                                              gamma=learning_rate_scheduler_gamma)
    for epoch in range(num_epochs):
        epoch_generator_loss, epoch_discriminator_loss = train_epoch(dataset_source=dataset_source,
                                                                     dataset_target=dataset_target,
                                                                     dataset_source_loader=dataset_source_loader,
                                                                     dataset_target_loader=dataset_target_loader,
                                                                     iterations=steps_per_epoch,
                                                                     epoch=epoch,
                                                                     generator=generator,
                                                                     discriminator_S=discriminator_S,
                                                                     discriminator_T=discriminator_T,
                                                                     device=device,
                                                                     discriminator_S_optimizer=discriminator_S_optimizer,
                                                                     discriminator_T_optimizer=discriminator_T_optimizer,
                                                                     generator_optimizer=generator_optimizer)

        print("EPOCH generator loss {}, discriminator loss {}".format(epoch_generator_loss, epoch_discriminator_loss))
        torch.save(generator.state_dict(), checkpoint_location + checkpoint_filename + 'generator_at_epoc_{}.pth'.format(epoch))
        torch.save(discriminator_T.state_dict(), checkpoint_location + checkpoint_filename + 'discriminatorT_at_epoc_{}.pth'.format(epoch))
        torch.save(discriminator_S.state_dict(), checkpoint_location + checkpoint_filename + 'discriminatorS_at_epoc_{}.pth'.format(epoch))
        if use_visdom:
            #viz_plotter.plot_line("epoch_discriminator_loss", epoch_discriminator_loss,"Epoch Discriminator Loss", do_clip_vals=False)
            #viz_plotter.plot_line("epoch_generator_loss", epoch_generator_loss, "Epoch Generator Loss", do_clip_vals=False)
            viz_plotter.plot_lines("epoch_losses", "Epoch Losses", epoch,
                                   [epoch_discriminator_loss, epoch_generator_loss],
                                   ["Generator Loss", "Discriminator Loss"], [[255, 0, 0], [0, 0, 255]])

if __name__ == "__main__":
    run()
