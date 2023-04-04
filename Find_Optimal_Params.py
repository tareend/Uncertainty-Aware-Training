# **********************************************************************************************************************
# Imports
# **********************************************************************************************************************

import pandas as pd
import numpy as np
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.metrics import roc_curve, confusion_matrix, auc
from torch.utils.tensorboard import \
    SummaryWriter

# **********************************************************************************************************************
# Functions
# **********************************************************************************************************************
def get_model_weights_filepath(folder, model_type):
    now = time.localtime(time.time())
    time_string = '{}_{}_{}_{}_{}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    model_weights_filepath = os.path.join(folder, '{}_{}.pt'.format(time_string, model_type))
    return model_weights_filepath


def calc_label_weights(Y):
    """ Y_train goes in """
    num_pos = Y.sum(dim=0)
    num_total = Y.shape[0]
    num_neg = num_total - num_pos
    weights = num_neg / num_pos
    return weights


def overlap(a, b):
    # return the indices in a that overlap with b, also returns
    # the corresponding index in b only works if both a and b are unique!
    # This is not very efficient but it works
    bool_a = np.in1d(a, b)
    ind_a = np.arange(len(a))
    ind_a = ind_a[bool_a]

    ind_b = np.array([np.argwhere(b == a[x]) for x in ind_a]).flatten()
    return ind_a, ind_b


def max_estimate(X):
    return np.argmax(X, axis=1)

def my_dice(pred, true, eps=1e-7):
    max_pred = max_estimate(pred)
    max_pred = max_pred.reshape((-1, 80, 80))
    true = true.reshape((-1, 80, 80))
    N = true.shape[0]
    num_classes = 4
    dice_scores = np.zeros((N, num_classes))
    for n in range(N):
        for c in range(num_classes):
            pred_n_c = (max_pred[n] == c).flatten()
            true_n_c = (true[n] == c).flatten()
            intersection = (pred_n_c * true_n_c).sum()
            sum_pred = pred_n_c.sum()
            sum_true = true_n_c.sum()
            dice_scores[n, c] = 2 * intersection / (sum_pred + sum_true + eps)
    return np.mean(dice_scores, axis=0)

class ResBlock2D(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, resolution=None):
        """ ResNet like block

        Parameters
        ----------
        num_filters_in - int - number of filters input
        num_filters_out - int - number of filters output
        downsample - bool - if True, downsample in first conv

        Notes
        -----
        If num_filters_in = num_filters_out then residual activation is identity
        If num_filters_in > num_filters_out then the residual activation has a 1x1 conv to down-filter
        If num_filters_in < num_filters_out then the filters are padded with 0s

        In the main block of the res-layer use num_filters_in
        """
        super(ResBlock2D, self).__init__()
        self.num_filters_in = num_filters_in
        self.num_filters_out = num_filters_out
        self.resolution = resolution

        self.conv1 = nn.Conv2d(self.num_filters_in, self.num_filters_in, 3, stride=1, padding=1)

        if resolution == 'downsample':
            self.conv2 = nn.Conv2d(self.num_filters_in, self.num_filters_out, 2, stride=2, padding=0)
        elif resolution == 'upsample':
            self.conv2 = nn.ConvTranspose2d(self.num_filters_in, self.num_filters_out, 4, stride=2, padding=1)
        else:
            self.conv2 = nn.Conv2d(self.num_filters_in, self.num_filters_out, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_filters_in)
        self.bn2 = nn.BatchNorm2d(self.num_filters_out)

        if num_filters_in > num_filters_out:
            # create 1x1 conv layer for down-filtering the residual
            self.conv_1x1 = nn.Conv2d(self.num_filters_in, self.num_filters_out, 1, stride=1, padding=0)
            self.bn_conv_1x1 = nn.BatchNorm2d(self.num_filters_out)

    def forward(self, x_in):
        res = nn.LeakyReLU(0.2)(self.bn1(self.conv1(x_in)))
        res = nn.LeakyReLU(0.2)(self.bn2(self.conv2(res)))

        if self.resolution == 'downsample':
            # if downsampling (and maybe adding filters) downsample the input
            x = F.interpolate(x_in, scale_factor=0.5)
        else:
            x = x_in

        if self.num_filters_in > self.num_filters_out:
            # reducing filters, so pass input through 1x1 conv
            x = F.relu(self.bn_conv_1x1(self.conv_1x1(x)))
        elif self.num_filters_in < self.num_filters_out:
            # adding filters, so pad inout with 0s
            x = torch.cat((x, torch.zeros_like(x)), 1)

        if self.resolution == 'upsample':
            x = F.interpolate(x, scale_factor=2)

        return res + x


class Encoder2D(nn.Module):
    def __init__(self, img_dim, code_dim, num_filters, num_input_channels=1):
        """ Infer probability distribution p(z|x) from data x.

        Parameters
        ----------
        img_dim : int
            Input image is square patch of this side.
        code_dim : int
            Output code is vector of this length.
        num_filters : int
            Number of filters in first convolutional layer, doubling thereafter.

        """
        super(Encoder2D, self).__init__()
        self.img_dim = img_dim
        self.code_dim = code_dim
        self.num_filters = num_filters
        self.num_input_channels = num_input_channels
        self.epsilon = 1e-6

        self.num_downsample = 4
        self.max_num_filters = 128
        self.low_res_img_dim = img_dim // (2 ** self.num_downsample)

        self.conv1 = nn.Conv2d(num_input_channels, self.num_filters, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_filters)

        num_filters_in = num_filters
        num_filters_out = num_filters
        res_layers = []

        for n in range(self.num_downsample):
            if num_filters_out < self.max_num_filters:
                num_filters_out *= 2
            res_layers.append(ResBlock2D(num_filters_in, num_filters_out, resolution='downsample'))
            num_filters_in = num_filters_out
            res_layers.append(ResBlock2D(num_filters_in, num_filters_out))
            res_layers.append(ResBlock2D(num_filters_in, num_filters_out))

        self.resnet = nn.Sequential(*res_layers)

        downsampled_size = img_dim // (2 ** self.num_downsample)
        self.final_conv_size = downsampled_size ** 2 * num_filters_out

        self.encode_fc_mu = nn.Linear(self.final_conv_size, self.code_dim)
        self.encode_fc_logvar = nn.Linear(self.final_conv_size, self.code_dim)

        self.bn_mu = nn.BatchNorm1d(self.code_dim)
        self.bn_logvar = nn.BatchNorm1d(self.code_dim)

    def forward(self, x):
        x = x.reshape(-1, self.num_input_channels, self.img_dim, self.img_dim)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resnet(x)
        x = x.reshape(-1, self.final_conv_size)
        mu = self.bn_mu(self.encode_fc_mu(x))
        logvar = F.softplus(self.bn_logvar(self.encode_fc_logvar(x))) + self.epsilon
        return mu, logvar


class Decoder2D(nn.Module):
    def __init__(self, img_dim, code_dim, num_filters, num_output_channels=1, num_output_classes=1):
        """ """
        super(Decoder2D, self).__init__()
        self.img_dim = img_dim
        self.code_dim = code_dim
        self.num_filters = num_filters
        self.num_output_channels = num_output_channels
        self.num_output_classes = num_output_classes

        self.num_upsample = 4
        self.low_res_img_dim = img_dim // (2 ** self.num_upsample)
        self.initial_num_filters = num_filters * (2 ** self.num_upsample)
        self.initial_fc_size = self.initial_num_filters * self.low_res_img_dim ** 2

        self.fc_1 = nn.Linear(self.code_dim, self.initial_fc_size)
        self.bn_1 = nn.BatchNorm1d(self.initial_fc_size)

        res_layers = []
        num_filters_in = self.initial_num_filters
        num_filters_out = self.initial_num_filters
        for n in range(self.num_upsample):
            num_filters_out //= 2
            res_layers.append(ResBlock2D(num_filters_in, num_filters_out, resolution='upsample'))
            num_filters_in = num_filters_out
            res_layers.append(ResBlock2D(num_filters_in, num_filters_out))
            res_layers.append(ResBlock2D(num_filters_in, num_filters_out))

        self.resnet = nn.Sequential(*res_layers)

        assert num_filters_out == num_filters, '{} : {}'.format(num_filters_out, num_filters)
        self.conv_final = nn.Conv2d(num_filters_out, num_output_channels * num_output_classes, 1, stride=1, padding=0)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.bn_1(self.fc_1(x)))
        x = x.reshape(-1, self.initial_num_filters, self.low_res_img_dim, self.low_res_img_dim)
        x = self.resnet(x)
        x = self.conv_final(x)
        return x.reshape(-1, self.num_output_classes, self.num_output_channels, self.img_dim, self.img_dim)


class LatentSpaceMLP_Conv(nn.Module):
    def __init__(self, t_dim, code_dim, z_concat_dim):
        """ Maps from 50 latent vectors (one per image) to 1 latent vector (one per subject) """
        super(LatentSpaceMLP_Conv, self).__init__()
        self.t_dim = t_dim
        self.code_dim = code_dim
        self.z_concat_dim = z_concat_dim

        self.fc0 = nn.Linear(self.code_dim, self.code_dim)
        self.bn0 = nn.BatchNorm1d(self.code_dim)
        self.fc1 = nn.Linear(self.code_dim, self.code_dim)
        self.bn1 = nn.BatchNorm1d(self.code_dim)
        self.fc2 = nn.Linear(self.code_dim, self.code_dim)
        self.bn2 = nn.BatchNorm1d(self.code_dim)
        self.conv1 = nn.Conv1d(self.code_dim, self.code_dim, 5, padding=1)  # t_dim -> 48 -> 24
        self.bn3 = nn.BatchNorm1d(self.code_dim)
        self.conv2 = nn.Conv1d(self.code_dim, self.code_dim, 5, padding=2)  # t_dim -> 24
        self.bn4 = nn.BatchNorm1d(self.code_dim)
        self.conv3 = nn.Conv1d(self.code_dim, self.code_dim, 5, padding=2)  # t_dim -> 12
        self.bn5 = nn.BatchNorm1d(self.code_dim)
        self.conv4 = nn.Conv1d(self.code_dim, self.code_dim, 5)  # t_dim -> 8
        self.bn6 = nn.BatchNorm1d(self.code_dim)
        self.fc3 = nn.Linear(self.code_dim * 8, self.z_concat_dim)

    def forward(self, x):
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = x.reshape(-1, self.t_dim, self.code_dim)
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(F.relu(self.bn3(self.conv1(x))), kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv2(x)))
        x = F.max_pool1d(F.relu(self.bn5(self.conv3(x))), kernel_size=2, stride=2)
        x = F.relu(self.bn6(self.conv4(x)))
        x = x.reshape(-1, self.code_dim * 8)
        x = self.fc3(x)
        return x

class LatentSpaceMLP_Concat(nn.Module):
    def __init__(self, t_dim, code_dim, z_concat_dim):
        """ Classify sequence of latent space vectors with MLP """
        super(LatentSpaceMLP_Concat, self).__init__()
        self.t_dim = t_dim
        self.code_dim = code_dim
        self.z_concat_dim = z_concat_dim

        self.fc0 = nn.Linear(self.code_dim, self.code_dim)
        self.bn0 = nn.BatchNorm1d(self.code_dim)
        self.fc1 = nn.Linear(self.code_dim, self.z_concat_dim)
        self.bn1 = nn.BatchNorm1d(self.z_concat_dim)
        self.fc2 = nn.Linear(self.z_concat_dim, self.z_concat_dim)
        self.bn2 = nn.BatchNorm1d(self.z_concat_dim)
        # concat all vectors in timeseries
        self.fc3 = nn.Linear(self.z_concat_dim * t_dim, self.z_concat_dim)
        self.bn3 = nn.BatchNorm1d(self.z_concat_dim)

    def forward(self, x):
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = x.reshape(-1, (self.t_dim) * self.z_concat_dim)
        x = F.relu(self.bn3(self.fc3(x)))
        return x

class LatentSpaceMLP_Classifier(nn.Module):
    def __init__(self, z_concat_dim, num_class):
        """ Classify sequence of latent space vectors with MLP """
        super(LatentSpaceMLP_Classifier, self).__init__()
        self.z_concat_dim = z_concat_dim
        self.num_class = num_class

        self.fc1 = nn.Linear(self.z_concat_dim, self.z_concat_dim)
        self.bn1 = nn.BatchNorm1d(self.z_concat_dim)
        self.fc2 = nn.Linear(self.z_concat_dim, self.z_concat_dim)
        self.bn2 = nn.BatchNorm1d(self.z_concat_dim)
        self.fc_final = nn.Linear(self.z_concat_dim, self.num_class + self.num_class)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc_final(x)
        return x

# **********************************************************************************************************************
# Image parameters
# **********************************************************************************************************************
img_dim = 80
t_dim = 25
NImg = 10

# **********************************************************************************************************************
# DL hyper parameters
# **********************************************************************************************************************
num_epochs = 300
batch_size = 8
lr = 1e-4
num_filters = 32
code_dim = 128
LOG_INTERVAL = 10
num_classes = 4
epsilon = 1e-6

## Optimal params ##

z_concat_dim = 32 ## Hidden layers
kld_weight = [0.001, 0.01, 0.1] ## Example of range input
confid_weighting_scale = [1.5, 1, 0.8, 0.6, 1.2, 2, 2.5, 3, 3.5, 4] ## Example of range input
classifier_weight = [0.5, 0.8, 1.5, 1.2, 1.5, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 3] ## Example of range input
sample = 20 ## Number of samples to draw from latent space

# **********************************************************************************************************************
# Root directories
# **********************************************************************************************************************
# Paths

Date = 'Date'

root_save_dir = (
    'Data_Path_/Find_Optimal/'  # This one uses the CI with Epistemic sampling as UC
    '{0}_Epochs_{1}_Batch_Size_{2}_t_dim_{3}'.format(Date, num_epochs, batch_size, t_dim))

if not os.path.exists(root_save_dir):
    os.system('mkdir -p {0}'.format(root_save_dir))

# **********************************************************************************************************************
# Load data
# **********************************************************************************************************************

Q = myArr = np.zeros((73, 3, 25, 80, 80))  ## Load your data here ##
Q = np.transpose(Q, (0, 1, 4, 2, 3))
num_input_channels = Q.shape[1]
Q = torch.tensor(Q)

eid_metadata = np.empty((73,))  ## Load your data here ##
Y_CRT = np.zeros((73,))  ## Load your data here ##

pre_trained_encoder = 'Place file path here'
pre_trained_decoder = 'Place file path here'

# **********************************************************************************************************************
# CUDA parameters
# **********************************************************************************************************************
torch.backends.cudnn.version()
torch.manual_seed(0)
device = torch.device("cuda")
bce_loss_function = nn.CrossEntropyLoss()

# **********************************************************************************************************************
# TRAIN + TEST VAE
# **********************************************************************************************************************
train_loss_epoch = []
bce_train_loss_epoch = []
kld_train_loss_epoch = []
crt_train_loss_epoch = []
Test_number = 0

# **************************************************************************************************
# Split in train and test
# **************************************************************************************************
indices = np.arange(eid_metadata.shape[0])
skf = StratifiedKFold(n_splits=2)
outer_skf = StratifiedKFold(n_splits=5)
outer_kskf = 0

Y_true_all = []
Y_pred_all = []
X_index = []

for train_outer_indices, test_outer_indices in outer_skf.split(indices, Y_CRT):
    data_test_outer = Q[test_outer_indices]
    Y_CRT_test_outer = Y_CRT[test_outer_indices]
    Y_true_inner = []
    Y_pred_inner = []

    for kld in kld_weight:

        for w in confid_weighting_scale:

            for cw in classifier_weight:

                Test_number = Test_number + 1
                kskf = 0

                save_dir = '{0}/Test_{1}/saved_weights_and_data_lr_{2}_nbF_{3}_codeDim_{4}_z_concat_dim_{5}_kld_{6}_classif_weight_{7}_loss_weight_{8}/'.format(
                    root_save_dir, Test_number, lr,
                    num_filters,
                    code_dim,
                    z_concat_dim,
                    kld,
                    cw, w)

                if not os.path.exists(save_dir):
                    os.system('mkdir -p {0}'.format(save_dir))

                image_dir = '{0}/Test_{1}/images_lr_{2}_nbF_{3}_codeDim_{4}_z_concat_dim_{5}_kld_{6}_classif_weight_{7}_loss_weight_{8}'.format(
                    root_save_dir, Test_number, lr, num_filters,
                    code_dim, z_concat_dim, kld, cw, w)
                if not os.path.exists(image_dir):
                    os.system('mkdir -p {0}'.format(image_dir))

                for train_indices, test_indices in skf.split(train_outer_indices, Y_CRT[train_outer_indices]):

                    writer = SummaryWriter(
                        '{0}/Test_{1}/Runs/_TensorBoard_Inner_Fold_{2}'.format(root_save_dir, Test_number, kskf))  # save logs/runs

                    encoder = torch.load('')  ## Load pre-trained encoder ##
                    decoder = torch.load('')  ## Load pre-trained decoder ##

                    classifier_1 = LatentSpaceMLP_Concat(t_dim=t_dim, code_dim=code_dim, z_concat_dim=z_concat_dim).to(
                        device)
                    classifier_CRT = LatentSpaceMLP_Classifier(z_concat_dim=z_concat_dim, num_class=1).to(device)

                    lr = 1e-4
                    lr_c = 1e-8
                    enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
                    dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
                    cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr_c)
                    cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr_c)

                    data_train = Q[train_indices]
                    data_test = Q[test_indices]
                    label_weights_crt = calc_label_weights(Y_CRT[train_indices])

                    Y_CRT_train = Y_CRT[train_indices].clone().detach().float()
                    Y_CRT_test = Y_CRT[test_indices].clone().detach().float()

                    N = data_train.shape[0]
                    num_batches = N // batch_size
                    N_test = data_test.shape[0]
                    num_batches_test = N_test // batch_size

                    for epoch in range(num_epochs):
                        # **********************************************************************************************
                        # Train
                        # **********************************************************************************************

                        encoder.train()
                        decoder.train()
                        classifier_1.train()
                        classifier_CRT.train()

                        total_loss_epoch = 0.
                        kld_loss_epoch = 0.
                        bce_loss_epoch = 0.
                        classifier_loss_crt_epoch = 0.
                        dice_list = []
                        out_confid_loss = []

                        batch_indices = np.arange(N, dtype=np.int)
                        np.random.shuffle(batch_indices)
                        pred_labels_CRT = np.zeros(N)
                        confidences = np.zeros(N)

                        for batch_idx in range(num_batches):
                            this_batch_indices = batch_indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                            X = data_train[this_batch_indices]
                            X_in = np.transpose(X, (0, 2, 1, 3, 4))
                            X = X.reshape(-1, num_input_channels, img_dim, img_dim)
                            X = torch.tensor(X)
                            X_sample = X
                            X_sample = X_sample.long().to(device)
                            X = X.long().to(device)
                            Ycrt = Y_CRT_train[this_batch_indices].to(device)
                            GT_samples = Y_CRT_train[this_batch_indices].float().cpu().data.squeeze().numpy()

                            # Train VAE
                            enc_optimizer.zero_grad()
                            dec_optimizer.zero_grad()
                            cla_1_optimizer.zero_grad()
                            cla_optimizer_CRT.zero_grad()

                            # reparameterisation inside the VAE
                            mu, logvar = encoder(X.float())
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z = eps.mul(std).add_(mu)
                            ww = classifier_1(mu)
                            Y_CRT_pred = classifier_CRT(ww)

                            # Decode VAE
                            X_decoded = decoder(z)

                            # Loss functions
                            # KLD loss
                            kld_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight) / (
                                    code_dim * batch_size * t_dim)

                            # decoder loss
                            bce_loss = bce_loss_function(
                                X_decoded.reshape(-1, num_classes, num_input_channels, img_dim ** 2),
                                X.reshape(-1, num_input_channels, img_dim ** 2))

                            dice_list.append(
                                my_dice(X_decoded.detach().cpu().numpy(), X.float().detach().cpu().numpy()))

                            predicted_probs = torch.softmax(Y_CRT_pred, dim=1)
                            confidences[this_batch_indices] = np.maximum(
                                predicted_probs.cpu().data.squeeze().numpy()[:, 0],
                                predicted_probs.cpu().data.squeeze().numpy()[:, 1])

                            pred_labels_CRT[this_batch_indices] = np.maximum(
                                Y_CRT_pred.cpu().data.squeeze().numpy()[:, 0],
                                Y_CRT_pred.cpu().data.squeeze().numpy()[:, 1])
                            predictions = torch.tensor(pred_labels_CRT[this_batch_indices]).to(device)

                            ###############################################################################################################################################

                            pred_labels_CRT_sample = []
                            probs_CRT_sample = []

                            with torch.no_grad():

                                Y_GT_sample = np.repeat(GT_samples, sample)
                                mu_sample, logvar_sample = encoder(X_sample.float())
                                std_sample = torch.exp(0.5 * logvar_sample)
                                eps_sample = torch.randn_like(std_sample)
                                z_sample = eps_sample.mul(std_sample).add_(mu_sample)
                                ww_sample = classifier_1(mu_sample)
                                Y_CRT_pred_sample = classifier_CRT(ww_sample)
                                probs = torch.softmax((classifier_CRT(ww_sample)), dim=1)
                                confidences_probs = np.maximum(probs.cpu().data.squeeze().numpy()[:, 0],
                                                               probs.cpu().data.squeeze().numpy()[:, 1])
                                pred_labels_CRT_pos_one = np.argmax(probs.cpu().data.squeeze().numpy(),
                                                                    axis=1)

                                sampled_preds = np.zeros((X_in.shape[0], sample))
                                sampled_out_probs = np.zeros((X_in.shape[0], sample))

                                sampled_out_probs[:, 0] = confidences_probs
                                sampled_preds[:, 0] = pred_labels_CRT_pos_one

                                q = torch.distributions.Normal(mu_sample, std_sample)
                                for i in range(1, sample):
                                    ##  Random sampling ##
                                    z_random_sample = q.rsample()
                                    difference_z = torch.cdist(z_sample, z_random_sample)
                                    #################################
                                    w_val_z_random = classifier_1(z_random_sample)
                                    Y_pred_val_CRT_random = classifier_CRT(w_val_z_random)

                                    sampled_probs = torch.softmax(Y_pred_val_CRT_random, dim=1)
                                    sampled_confidences_probs = np.maximum(
                                        sampled_probs.cpu().data.squeeze().numpy()[:, 0],
                                        sampled_probs.cpu().data.squeeze().numpy()[:, 1])

                                    sampled_out_probs[:, i] = sampled_confidences_probs
                                    sampled_preds[:, i] = np.argmax(sampled_probs.cpu().data.squeeze().numpy(), axis=1)

                            pred_labels_CRT_sample_all = np.concatenate(sampled_preds)
                            probs_CRT_sample_all = np.concatenate(sampled_out_probs)

                            x = 0  # if 1 dont include original segmentation into CI bands
                            y = sample
                            CI_patient = np.zeros_like(GT_samples).astype(float)
                            CI_patient_0 = np.zeros_like(GT_samples).astype(float)

                            for ii in range(len(Ycrt)):
                                CI_patient[ii] = (sum(pred_labels_CRT_sample_all[x:y]) / sample)
                                CI_patient_0[ii] = (1 - CI_patient[ii])
                                x = x + 20  # 20
                                y = y + 20  # 20

                            confid_weight_train = GT_samples * (CI_patient_0) + (1 - GT_samples) * CI_patient
                            confid_weight_train = (1 - confid_weighting_scale) * confid_weight_train + confid_weighting_scale
                            confid_weight = torch.tensor(confid_weight_train)

                            #################################################################################################################################
                            max_elements, max_idxs = torch.max(Y_CRT_pred, dim=1)
                            classifier_loss_function_crt = nn.BCEWithLogitsLoss(weight=confid_weight.to(device),
                                                                                pos_weight=label_weights_crt.to(device))
                            classifier_loss_crt = classifier_loss_function_crt(max_elements, Ycrt) * classifier_weight

                            total_loss = kld_loss + bce_loss + classifier_loss_crt

                            print('Training Loss: ', total_loss)

                            total_loss.backward()
                            enc_optimizer.step()
                            dec_optimizer.step()
                            cla_1_optimizer.step()
                            cla_optimizer_CRT.step()

                            total_loss_epoch += total_loss.item()
                            kld_loss_epoch += kld_loss.item()
                            bce_loss_epoch += bce_loss.item()
                            classifier_loss_crt_epoch += classifier_loss_crt.item()

                        if epoch == 100:

                            torch.save(obj=encoder,
                                       f='{0}/encoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                            torch.save(obj=decoder,
                                       f='{0}/decoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                            torch.save(obj=classifier_1,
                                       f='{0}/classifier_1_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                            torch.save(obj=classifier_CRT,
                                       f='{0}/classifier_CRT_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))

                        elif epoch == 200:

                            torch.save(obj=encoder,
                                       f='{0}/encoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                            torch.save(obj=decoder,
                                       f='{0}/decoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                            torch.save(obj=classifier_1,
                                       f='{0}/classifier_1_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                            torch.save(obj=classifier_CRT,
                                       f='{0}/classifier_CRT_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))

                    torch.save(obj=encoder,
                               f='{0}/encoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                    torch.save(obj=decoder,
                               f='{0}/decoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                    torch.save(obj=classifier_1,
                               f='{0}/classifier_1_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                    torch.save(obj=classifier_CRT,
                               f='{0}/classifier_CRT_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(save_dir, z_concat_dim,
                                                                                                kld,
                                                                                                epoch, kskf))
                    kskf = kskf + 1

                    # **********************************************************************************************
                    #  VALIDATION
                    # **********************************************************************************************
                    ## Print the Validation accuracy per EPOCH ##

                    encoder.eval()
                    decoder.eval()
                    classifier_1.eval()
                    classifier_CRT.eval()

                    Y_pred_val_CRT = np.zeros(Y_CRT_test.shape[0])
                    Y_CRT_testing = np.zeros(Y_CRT_test.shape[0])
                    N_val = Y_CRT_test.shape[0]
                    batch_size_test = batch_size
                    num_batches_test = N_val // batch_size_test
                    batch_indices_test = np.arange(N_val, dtype=np.int)
                    batch_index = 0
                    test_index = np.zeros(Y_CRT_test.shape[0])

                    with torch.no_grad():
                        X_v = data_test
                        X_v = X_v.reshape(-1, num_input_channels, img_dim, img_dim)
                        X_v = X_v.long().to(device)
                        mu, logvar = encoder(X_v.float())
                        X_decoded_val = decoder(mu)  # torch.Size([250, 4, 3, 80, 80])
                        w_val = classifier_1(mu)
                        Y_pred_val_CRT = classifier_CRT(w_val).detach().cpu().numpy()

                    predicted_probs_outer = torch.softmax(torch.tensor(Y_pred_val_CRT.cpu().data.squeeze().numpy()),
                                                          dim=1)
                    confidences = np.maximum(predicted_probs_outer.cpu().data.squeeze().numpy()[:, 0],
                                             predicted_probs_outer.cpu().data.squeeze().numpy()[:, 1])

                    pred_labels_CRT_outer = np.maximum(Y_pred_val_CRT.cpu().data.squeeze().numpy()[:, 0],
                                                       Y_pred_val_CRT.cpu().data.squeeze().numpy()[:, 1])

                # Outside the inner loop
                Y_true_all = np.concatenate(Y_true_inner)
                Y_pred_all = np.concatenate(Y_CRT_test)
                fpr, tpr, thresholds = roc_curve(Y_true_all, Y_pred_all)
                auc_val = auc(fpr, tpr)
                idx = np.argmax(tpr-fpr)
                optimal_th = thresholds[idx]
                y_CRT_pred_zo = np.zeros_like(Y_true_all)
                y_CRT_pred_zo[Y_pred_all > optimal_th] = 1

                tn, fp, fn, tp = confusion_matrix(Y_true_all, y_CRT_pred_zo).ravel()
                sen = tp / (tp + fn)
                spe = tn / (tn + fp)
                bacc = 0.5 * (sen + spe)

                np.savetxt('{0}/Thresholds_w_{1}_cw_{2}_outer{3}.txt'.format(save_dir, w, cw, outer_kskf), [optimal_th])
                outcomes = pd.DataFrame({"Sen_Sen_Acc": [sen, spe, bacc]})
                outcomes.to_csv(r'{0}/Sen_Sen_Acc_Outer_w_{1}_cw_{2}_outer{3}.csv'.format(save_dir, w, cw, outer_kskf),header=True, index=False)

                print(
                    'Test CRT \t w {:.1f}  \t cw {:.3f}  \t BACC: \t {:.3f} \t SEN: {:.3f} \t SPE: \t {:.3f} '.format(w, cw, bacc, sen, spe))

                ##################################################################################################

        test_set = pd.DataFrame({"Test Set Outer": test_outer_indices})
        test_set.to_csv(r'{0}/Test_Set_Outer_Fold_{1}.csv'.format(save_dir, outer_kskf), header=True, index=False)

        outer_kskf = outer_kskf + 1
        torch.cuda.empty_cache()
