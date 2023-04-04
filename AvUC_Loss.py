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
from sklearn.metrics import roc_curve, confusion_matrix
from torch.utils.tensorboard import \
    SummaryWriter
import torchio as tio

# **********************************************************************************************************************
# Functions
# **********************************************************************************************************************

def augment(image):
    flip = tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.1)
    flipped = flip(image)
    to_ras = tio.ToCanonical()
    flipped_canonical = to_ras(flipped)
    flip2 = tio.RandomFlip(axes=['anterior-inferior'], flip_probability=0.1)
    final_image = flip2(flipped_canonical)

    return (final_image)

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
            # if upsampling (and maybe reducing filters) interpolate newly created filters
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
        # force logvar to be positive and > epsilon
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
        # concat all vectors in timeseries then do 1d convs on them
        self.conv1 = nn.Conv1d(self.code_dim, self.code_dim, 5, padding=1)  # t_dim -> 48 -> 24
        self.bn3 = nn.BatchNorm1d(self.code_dim)
        self.conv2 = nn.Conv1d(self.code_dim, self.code_dim, 5, padding=2)  # t_dim -> 24
        self.bn4 = nn.BatchNorm1d(self.code_dim)
        self.conv3 = nn.Conv1d(self.code_dim, self.code_dim, 5, padding=2)  # t_dim -> 12
        self.bn5 = nn.BatchNorm1d(self.code_dim)
        self.conv4 = nn.Conv1d(self.code_dim, self.code_dim, 5)  # t_dim -> 8
        self.bn6 = nn.BatchNorm1d(self.code_dim)
        # vectorise into FC layer
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

class AvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default);
    1: model uncertainty]
    """
    def __init__(self, beta=1):
        super(AvULoss, self).__init__()
        self.beta = alpha
        self.eps = 1e-5

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        # number of samples accurate and certain
        n_ac = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and certain
        n_ic = torch.zeros(1, device=true_label.device)
        # number of samples accurate and uncertain
        n_au = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device)

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels, optimal_uncertainty_threshold, type=0):

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        unc_th = torch.tensor(optimal_uncertainty_threshold,
                              device=logits.device)

        n_ac = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=logits.device)
        avu_loss = torch.zeros(1, device=logits.device)

        for i in range(len(labels)):
            if ((labels[i].item() == predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                """ accurate and certain """
                n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() == predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ accurate and uncertain """
                n_au += confidences[i] * torch.tanh(unc[i])
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() <= unc_th.item()):
                """ inaccurate and certain """
                n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ inaccurate and uncertain """
                n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        p_ac = (n_ac) / (n_ac + n_ic)
        p_ui = (n_iu) / (n_iu + n_ic)
        avu_loss = -1 * self.beta * torch.log(avu + self.eps)
        expected_unc = torch.mean(unc, dim=0)

        return (avu_loss, expected_unc)

# **********************************************************************************************************************
# Image parameters
# **********************************************************************************************************************
img_dim = 80  # res of stored high-res images
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

z_concat_dim = 32
kld_weight = 0.001
classifier_weight = 2
alpha = 2

uncertainty_epoch_min = 3
optimal_threshold = 1

# Best params from find optimal #
Test_number = 1
type = '_AvUC_Thresh_Test_Set'
Date = 'Date'

Iteration_Runs = 5

# **********************************************************************************************************************
# Root directories
# **********************************************************************************************************************
# Paths

results_main_path = os.path.join('./Results/', 'Orig_Images_{0}{1}_Iterations_{2}'.format(Date, type, Iteration_Runs))
if not os.path.exists(results_main_path):
    os.system('mkdir -p {0}'.format(results_main_path))

root_save_dir = os.path.join(results_main_path, 'Test_{0}_Epochs_{1}_Batch_Size_{2}_t_dim_{3}_weight_{4}_CW_{5}'.format(Test_number, num_epochs, batch_size, t_dim, kld_weight, classifier_weight))
if not os.path.exists(root_save_dir):
    os.system('mkdir -p {0}'.format(root_save_dir))

Y_iteration_accuracy = []
iteration_number = []
epoch_number = []
fold_number = []

for iteration in range(Iteration_Runs):

    # **********************************************************************************************************************
    # Load data
    # **********************************************************************************************************************

    Q = myArr = np.zeros((73, 3, 25, 80, 80)) ## Load your data here ##
    Q = np.transpose(Q, (0, 1, 4, 2, 3))
    num_input_channels = Q.shape[1]
    Q = torch.tensor(Q)

    eid_metadata = np.empty((73,))           ## Load your data here ##
    Y_CRT = np.zeros((73,))                  ## Load your data here ##

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
    Y_true_all = []
    Y_pred_all = []
    Y_GT_All = []
    Y_Pred = []

    # **************************************************************************************************
    # Split in train and test
    # **************************************************************************************************

    indices = np.arange(eid_metadata.shape[0])
    skf = StratifiedKFold(n_splits=5)
    kskf = 0

    Y_CRT = torch.tensor(Y_CRT).float()

    root_save_dir_iter = os.path.join(root_save_dir, '{0}'.format(iteration))
    if not os.path.exists(root_save_dir_iter):
        os.system('mkdir -p {0}'.format(root_save_dir_iter))

    save_dir = os.path.join(root_save_dir_iter,
                            'saved_weights_and_data_lr_{0}_nbF_{1}_codeDim_{2}_z_concat_dim_{3}_kld_{4}_classif_weight_{5}_alpha_{6}'.format(
                                lr,
                                num_filters,
                                code_dim,
                                z_concat_dim,
                                kld_weight,
                                classifier_weight, alpha))

    image_dir = os.path.join(root_save_dir_iter,
                             'images_lr_{0}_nbF_{1}_codeDim_{2}_z_concat_dim_{3}_kld_{4}_classif_weight_{5}_alpha_{6}'.format(
                                 lr, num_filters,
                                 code_dim, z_concat_dim, kld_weight, classifier_weight, alpha))

    if not os.path.exists(save_dir):
        os.system('mkdir -p {0}'.format(save_dir))

    if not os.path.exists(image_dir):
        os.system('mkdir -p {0}'.format(image_dir))

    for train_indices, test_indices in skf.split(indices, Y_CRT):

        data_train = Q[train_indices]
        data_test = Q[test_indices]
        label_weights_crt = calc_label_weights(Y_CRT[train_indices])
        Y_CRT_train = Y_CRT[train_indices].clone().detach().float()
        Y_CRT_test = Y_CRT[test_indices].clone().detach().float()

        N = data_train.shape[0]
        num_batches = N // batch_size
        N_test = data_test.shape[0]
        num_batches_test = N_test // batch_size

        model_path = os.path.join(save_dir, 'K_Fold_{0}_num_hidden_{1}_kld_{2}'.format(kskf, z_concat_dim, kld_weight))
        if not os.path.exists(model_path):
            os.system('mkdir -p {0}'.format(model_path))

        writer = SummaryWriter(
            '{0}/Runs/TensorBoard_Fold_{2}'.format(root_save_dir_iter, Test_number, kskf))  # save logs/runs

        encoder = torch.load('') ## Load pre-trained encoder ##
        decoder = torch.load('') ## Load pre-trained decoder ##

        classifier_1 = LatentSpaceMLP_Concat(t_dim=t_dim, code_dim=code_dim, z_concat_dim=z_concat_dim).to(
            device)
        classifier_CRT = LatentSpaceMLP_Classifier(z_concat_dim=z_concat_dim, num_class=1).to(device)

        enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
        cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr)
        cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr)

        if kskf == 0:
            lr = 1e-2
            lr_c = 1e-8
            enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
            dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
            cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr_c)
            cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr_c)
            cla_optimizer_CRT_CI = cla_optimizer_CRT
        elif kskf == 1:
            lr = 1e-6
            enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
            dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
            cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr)
            cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr)
            cla_optimizer_CRT_CI = cla_optimizer_CRT
        elif kskf == 2:
            lr = 1e-6
            enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
            dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
            cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr)
            cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr)
            cla_optimizer_CRT_CI = cla_optimizer_CRT
        elif kskf == 3:
            lr = 1e-2
            lr_c = 1e-4
            enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
            dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
            cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr_c)
            cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr_c)
            cla_optimizer_CRT_CI = cla_optimizer_CRT
        elif kskf == 4:
            lr = 1e-6
            enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
            dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
            cla_1_optimizer = optim.Adam(classifier_1.parameters(), lr=lr)
            cla_optimizer_CRT = optim.Adam(classifier_CRT.parameters(), lr=lr)
            cla_optimizer_CRT_CI = cla_optimizer_CRT

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
            avuc_loss_epoch = 0.
            dice_list = []
            out_confid_loss = []

            batch_indices = np.arange(N, dtype=np.int)
            np.random.shuffle(batch_indices)
            pred_labels_CRT = np.zeros(N)
            pred_labels_CRT_arg = np.zeros(N)
            confidences = np.zeros(N)

            for batch_idx in range(num_batches):
                this_batch_indices = batch_indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                X = data_train[this_batch_indices]
                X_in = np.transpose(X, (0, 2, 1, 3, 4))
                X = X.reshape(-1, num_input_channels, img_dim, img_dim)
                X = augment(X)
                X = torch.tensor(X)
                X = X.long().to(device)
                Ycrt = Y_CRT_train[this_batch_indices].to(device)

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

                dice_list.append(my_dice(X_decoded.detach().cpu().numpy(), X.float().detach().cpu().numpy()))

                predicted_probs = torch.softmax(Y_CRT_pred, dim=1)
                confidences[this_batch_indices] = np.maximum(predicted_probs.cpu().data.squeeze().numpy()[:,0], predicted_probs.cpu().data.squeeze().numpy()[:,1])
                pred_labels_CRT_arg[this_batch_indices] = np.argmax(predicted_probs.cpu().data.squeeze().numpy(), axis=1)

                pred_labels_CRT[this_batch_indices] = np.maximum(Y_CRT_pred.cpu().data.squeeze().numpy()[:, 0], Y_CRT_pred.cpu().data.squeeze().numpy()[:, 1])
                predictions = torch.tensor(pred_labels_CRT[this_batch_indices]).to(device)

                ###############################################################################################################################################

                avUC = AvULoss(alpha)
                (AvUC_loss, unc) = avUC(Y_CRT_pred, Ycrt.float().cpu().data.squeeze().numpy(), optimal_threshold)

                #################################################################################################################################
                max_elements, max_idxs = torch.max(Y_CRT_pred, dim=1)
                classifier_loss_function_crt = nn.BCEWithLogitsLoss(pos_weight=label_weights_crt.to(device))
                classifier_loss_crt = classifier_loss_function_crt(max_elements, Ycrt) * classifier_weight

                total_loss = kld_loss + bce_loss + classifier_loss_crt + AvUC_loss

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
                avuc_loss_epoch += AvUC_loss.item()

            ######################## Tensorboard #################################
            writer.add_scalar('Total Loss/train', total_loss_epoch / num_batches, epoch)
            writer.add_scalar('KLD Loss/train', kld_loss_epoch / num_batches, epoch)
            writer.add_scalar('BCE Loss/train_Fold', bce_loss_epoch / num_batches, epoch)
            writer.add_scalar('Classifier Loss/train', classifier_loss_crt / num_batches, epoch)
            writer.add_scalar('AvUC_loss Loss/train', avuc_loss_epoch / num_batches, epoch)
            ######################################################################

            if epoch >= uncertainty_epoch_min:
                optimal_threshold = unc
            else:
                optimal_threshold = optimal_threshold

        torch.save(obj=encoder,
                   f='{0}/encoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(model_path, z_concat_dim,
                                                                                    kld_weight,
                                                                                    epoch, kskf))
        torch.save(obj=decoder,
                   f='{0}/decoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(model_path, z_concat_dim,
                                                                                    kld_weight,
                                                                                    epoch, kskf))
        torch.save(obj=classifier_1,
                   f='{0}/classifier_1_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(model_path, z_concat_dim,
                                                                                    kld_weight,
                                                                                    epoch, kskf))
        torch.save(obj=classifier_CRT,
                   f='{0}/classifier_CRT_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{4}.pt'.format(model_path, z_concat_dim,
                                                                                    kld_weight,
                                                                                    epoch, kskf))

        # **********************************************************************************************
        #  VALIDATION
        # **********************************************************************************************
        encoder.eval()
        decoder.eval()
        classifier_1.eval()
        classifier_CRT.eval()

        with torch.no_grad():
            X_v = data_test
            X_v = X_v.reshape(-1, num_input_channels, img_dim, img_dim)
            X_v = X_v.long().to(device)
            mu, logvar = encoder(X_v.float())
            X_decoded_val = decoder(mu)  # torch.Size([250, 4, 3, 80, 80])
            w_val = classifier_1(mu)
            Y_pred_val_CRT = classifier_CRT(w_val)

        predicted_probs_outer = torch.softmax(torch.tensor(Y_pred_val_CRT.cpu().data.squeeze().numpy()), dim=1)
        confidences = np.maximum(predicted_probs.cpu().data.squeeze().numpy()[:, 0],
                                                     predicted_probs.cpu().data.squeeze().numpy()[:, 1])

        pred_labels_CRT_outer = np.maximum(Y_pred_val_CRT.cpu().data.squeeze().numpy()[:, 0], Y_pred_val_CRT.cpu().data.squeeze().numpy()[:, 1])

        fpr, tpr, thresholds = roc_curve(Y_CRT_test.cpu().data.squeeze().numpy(), pred_labels_CRT_outer)
        idx = np.argmax(tpr - fpr)
        optimal_th = thresholds[idx]
        y_CRT_pred_zo = np.zeros_like(pred_labels_CRT_outer)
        y_CRT_pred_zo[pred_labels_CRT_outer > optimal_th] = 1

        tn, fp, fn, tp = confusion_matrix(Y_CRT_test.cpu().data.squeeze().numpy(), y_CRT_pred_zo).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        bacc = 0.5 * (sen + spe)

        outcomes = pd.DataFrame({"Sen_Spec_Acc": [sen, spe, bacc]})
        outcomes.to_csv(r'{0}/Sen_Spec_Acc_Outer_{1}_{2}_{3}.csv'.format(save_dir, z_concat_dim, kld_weight, kskf),
                        header=True, index=False)

        Y_iteration_accuracy.append(bacc)
        iteration_number.append(iteration)
        epoch_number.append(epoch+1)
        fold_number.append(kskf)

        print(
            'Test CRT \t num_hidden {:.1f}  \t kld_weight {:.3f}  \t BACC: \t {:.3f} \t SEN: {:.3f} \t SPE: \t {:.3f} '.format(
                z_concat_dim, kld_weight, bacc, sen, spe))

        test_set = pd.DataFrame({"Test Set Outer": test_indices})
        test_set.to_csv(r'{0}/Test_Set_Outer_Fold_{1}.csv'.format(save_dir, kskf), header=True, index=False)


        kskf = kskf + 1
        torch.cuda.empty_cache()

df_Y_iteration_accuracy = pd.DataFrame({"Y_iteration_accuracy": Y_iteration_accuracy})
df_iteration_number = pd.DataFrame({"iteration_number": iteration_number})
df_epoch_number = pd.DataFrame({"epoch_number": epoch_number})
df_fold_number = pd.DataFrame({"fold_number": fold_number})

av_accuracy =  [((sum(Y_iteration_accuracy[i::5]))/5) for i in range(len(Y_iteration_accuracy) // 5)]
df_av_iterations = pd.DataFrame({"Average": av_accuracy})

final = pd.concat([df_Y_iteration_accuracy, df_iteration_number, df_epoch_number, df_fold_number, df_av_iterations], axis=1)
final.to_csv(
    r'{0}/_Three_Iterations.csv'.format(root_save_dir), header=True)