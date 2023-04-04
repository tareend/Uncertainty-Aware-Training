# **********************************************************************************************************************
# Imports
# **********************************************************************************************************************

import pandas as pd
import os
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, confusion_matrix
import math
import numpy as np
from sklearn.metrics import brier_score_loss

# **********************************************************************************************************************
# Functions
# **********************************************************************************************************************

def AdaptiveBinning(infer_results, show_reliability_diagram=True):
    '''
    This function implement adaptive binning. It returns AECE, AMCE and some other useful values.
    Arguements:
    infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample. res[0] is the confidence score r and res[1] is the correctness score c. Since c is either 1 or 0, here res[1] is True if the prediction is correctd and False otherwise.
    show_reliability_diagram (boolean): a boolean value to denote wheather to plot a Reliability Diagram.
    Return Values:
    AECE (float): expected calibration error based on adaptive binning.
    AMCE (float): maximum calibration error based on adaptive binning.
    cofidence (list): average confidence in each bin.
    accuracy (list): average accuracy in each bin.
    cof_min (list): minimum of confidence in each bin.
    cof_max (list): maximum of confidence in each bin.
    '''

    # Intialize.

    infer_results = infer_results.values.tolist()

    infer_results.sort(key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    assert infer_results[0][0] <= 1 and infer_results[1][0] >= 0, 'Confidence score should be in [0,1]'

    z = 1.645
    num = [0 for i in range(n_total_sample)]
    final_num = [0 for i in range(n_total_sample)]
    correct = [0 for i in range(n_total_sample)]
    confidence = [0 for i in range(n_total_sample)]
    cof_min = [1 for i in range(n_total_sample)]
    cof_max = [0 for i in range(n_total_sample)]
    accuracy = [0 for i in range(n_total_sample)]

    ind = 0
    target_number_samples = float('inf')

    # Traverse all samples for a initial binning.
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[2]
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 40 and cof_min[ind] - infer_results[-1][0] > 0.05:
                ind += 1
                target_number_samples = float('inf')
        num[ind] += 1
        confidence[ind] += confidence_score
        try:
            if correctness == 1:
                correct[ind] += 1
        except:

            'Expect boolean value for correctness!'

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float('inf')
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    n_bins = ind + 1

    # Get final binning.
    if target_number_samples - num[ind] > 0:
        needed = target_number_samples - num[ind]
        extract = [0 for i in range(n_bins - 1)]
        final_num[n_bins - 1] = num[n_bins - 1]
        for i in range(n_bins - 1):
            extract[i] = int(needed * num[ind] / n_total_sample)
            final_num[i] = num[i] - extract[i]
            final_num[n_bins - 1] += extract[i]
    else:
        final_num = num
    final_num = final_num[:n_bins]

    # Re-intialize.
    num = [0 for i in range(n_bins)]
    correct = [0 for i in range(n_bins)]
    confidence = [0 for i in range(n_bins)]
    cof_min = [1 for i in range(n_bins)]
    cof_max = [0 for i in range(n_bins)]
    accuracy = [0 for i in range(n_bins)]
    gap = [0 for i in range(n_bins)]
    neg_gap = [0 for i in range(n_bins)]
    # Bar location and width.
    x_location = [0 for i in range(n_bins)]
    width = [0 for i in range(n_bins)]

    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):

        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[2]
        num[ind] += 1
        confidence[ind] += confidence_score

        if correctness == True:
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - accuracy[ind] > 0:
                gap[ind] = confidence[ind] - accuracy[ind]
            else:
                neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    # Get AECE and AMCE based on the binning.
    AMCE = 0
    AECE = 0
    for i in range(n_bins):
        AECE += abs((accuracy[i] - confidence[i])) * final_num[i] / n_total_sample
        AMCE = max(AMCE, abs((accuracy[i] - confidence[i])))

    print('ECE based on adaptive binning: {}'.format(AECE))
    print('MCE based on adaptive binning: {}'.format(AMCE))

    # Plot the Reliability Diagram if needed.
    if show_reliability_diagram:
        f1, ax = plt.subplots()
        plt.bar(x_location, accuracy, width)
        plt.bar(x_location, gap, width, bottom=accuracy)
        plt.bar(x_location, neg_gap, width, bottom=accuracy)
        plt.legend(['Accuracy', 'Overconfident', 'Underconfident'], fontsize=18, loc=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Confidence', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.show()

        f1.savefig('{0}/AECE_Diag_{1}.png'.format(save_dir, model_name))

    return AECE, AMCE, cof_min, cof_max, confidence, accuracy, n_bins

def compute_overconfidence_calibration_ece(true_labels, pred_labels, confidences, num_bins):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)

    bin_uncertainty = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])

            bin_uncertainty[b] = -1 * np.sum(
                confidences[selected] * np.log(confidences[selected] + 1e-15))
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)

    oce_values = bin_confidences - bin_accuracies
    max_values = np.zeros(oce_values.shape[0])
    mce = np.max(gaps)

    for i in range(oce_values.shape[0]):
        if oce_values[i] > 0:
            max_values[i] = oce_values[i]
        else:
            max_values[i] = 0

    oce = np.sum((bin_confidences * max_values) * bin_counts) / np.sum(bin_counts)

    return {"over_confidence_calibration_error": oce,
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "uncertainty": bin_uncertainty,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce}

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

# **********************************************************************************************************************
# Image parameters
# **********************************************************************************************************************
img_dim = 80  # res of stored high-res images
t_dim = 25
NImg = 10
bin_size = 15
# **********************************************************************************************************************
# DL hyper parameters
# **********************************************************************************************************************
num_epochs = 200
batch_size = 8
lr = 1e-4
num_filters = 32
code_dim = 128  # 256 #128
LOG_INTERVAL = 10
num_classes = 4
epsilon = 1e-4
samples = 20

# **********************************************************************************************************************
# Load data
# **********************************************************************************************************************

Q = np.zeros((1460, 3, 25, 80, 80))  ## Load your data here ## 20 i.e 73x20 (segmentations) = 1460
num_input_channels = Q.shape[1]
Q_new = torch.tensor(Q)

Q_orig = myArr = np.zeros((73, 3, 25, 80, 80))  ## Load your data here ##
num_input_channels = Q_orig.shape[1]
Q_orig = torch.tensor(Q_orig)

eid_metadata = np.empty((73,))  ## Load your data here ##
Y_CRT = np.zeros((73,))  ## Load your data here ##

# **********************************************************************************************************************
# CUDA parameters
# **********************************************************************************************************************
torch.backends.cudnn.version()
torch.manual_seed(0)
device = torch.device("cuda")

## Journal ##

## Depends on the pre trained models you load:
num_hidden = 32  # same as z_concat_dim
kld_weight = 0.1
classifier_weight = 2

Date = 'Place_Date_Here'
Test_Number = 1
model_name = 'Base_Evi'
folds = 5

root_save_dir = './Epistemic_Eval/' \
                '/{0}_Test_{1}_{2}'.format(model_name, Test_Number, Date)
########################################################################################################################################################################################################################################################################################

if not os.path.exists(root_save_dir):
    os.system('mkdir -p {0}'.format(root_save_dir))
save_dir = '{0}/UC_Test_Number_{1}_Epochs_{2}_lr_{3}_nbF_{4}_codeDim_{5}_classif_weight_{6}_Testing_Nested/'.format(
    root_save_dir, Test_Number, num_epochs, lr, num_filters,
    code_dim, classifier_weight)
if not os.path.exists(save_dir):
    os.system('mkdir -p {0}'.format(save_dir))

root_dir = 'Load optimal model from path'

########################################################################################################################################################################################################################################################################################

y_prob_all = []
Probs_positive_responder_all = []
Probs_negative_responder_all = []

y_true_all = []
y_pred_all = []
CI_patient_all = []
CI_patient_0_all = []
First_prediction_baseline_model_all = []
First_GT_baseline_model_all = []
Y_sample_GT_all = []

probs_all = []
eid_all = []
Volume_All = []
EF_All = []
y_CRT_pred_orig = []
Ypred_z0 = []

pred_labels_CRT_sample = []
probs_CRT_sample = []
predicted_probs_orig_all = []
predicted_preds_orig_all = []
predicted_probs_responders = []
predicted_probs_confid = []
probs_CRT_sample_all_total = []

for kfold in range(folds):

    test_index = np.squeeze(pd.read_csv('{0}/Test_Set_Outer_Fold_{1}.csv'.format(root_dir, kfold))).values
    GT_samples = Y_CRT[test_index]
    X_v = Q[test_index]
    X_v = X_v.reshape(-1, num_input_channels, img_dim, img_dim)
    X_v = X_v.long().to(device)

    encoder = torch.load(os.path.join(root_dir,
                                      'K_Fold_{0}_num_hidden_{1}_kld_{2}/encoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{0}.pt'.format(
                                          kfold, num_hidden, kld_weight, num_epochs-1)))
    decoder = torch.load(os.path.join(root_dir,
                                      'K_Fold_{0}_num_hidden_{1}_kld_{2}/decoder_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{0}.pt'.format(
                                          kfold, num_hidden, kld_weight, num_epochs-1)))
    classifier_1 = torch.load(os.path.join(root_dir,
                                           'K_Fold_{0}_num_hidden_{1}_kld_{2}/classifier_1_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{0}.pt'.format(
                                          kfold, num_hidden, kld_weight, num_epochs-1)))
    classifier_CRT = torch.load(os.path.join(root_dir,
                                             'K_Fold_{0}_num_hidden_{1}_kld_{2}/classifier_CRT_num_hidden_{1}_kld_{2}_epoch_{3}_inner_{0}.pt'.format(
                                          kfold, num_hidden, kld_weight, num_epochs-1)))

    encoder.eval()
    decoder.eval()
    classifier_1.eval()
    classifier_CRT.eval()

    eid_all.append(eid_metadata[test_index])

    with torch.no_grad():

        Y_GT_sample = np.repeat(GT_samples, samples)
        mu_sample, logvar_sample = encoder(X_v.float())
        std_sample = torch.exp(0.5 * logvar_sample)
        eps_sample = torch.randn_like(std_sample)
        z_sample = eps_sample.mul(std_sample).add_(mu_sample)
        ww_sample = classifier_1(mu_sample)
        Y_CRT_pred = classifier_CRT(ww_sample)
        probs = torch.softmax((classifier_CRT(ww_sample)), dim=1)
        predicted_probs_responders.append(probs.cpu().data.squeeze().numpy())

        confidences_probs = np.maximum(probs.cpu().data.squeeze().numpy()[:, 0],
                                       probs.cpu().data.squeeze().numpy()[:, 1])
        pred_labels_CRT_pos_one = np.argmax(probs.cpu().data.squeeze().numpy(),
                                            axis=1)

        pred_labels_CRT_outer = np.maximum(Y_CRT_pred.cpu().data.squeeze().numpy()[:, 0],
                                           Y_CRT_pred.cpu().data.squeeze().numpy()[:, 1])

        sampled_preds = np.zeros((Q[test_index].shape[0], samples))
        sampled_out_probs = np.zeros((Q[test_index].shape[0], samples))
        logits_out = np.zeros((Q[test_index].shape[0], samples))

        sampled_out_probs[:, 0] = confidences_probs
        sampled_preds[:, 0] = pred_labels_CRT_pos_one
        logits_out[:, 0] = pred_labels_CRT_outer

        q = torch.distributions.Normal(mu_sample, std_sample)
        for i in range(1, samples):
            z_random_sample = q.rsample()
            difference_z = torch.cdist(z_sample, z_random_sample)
            w_val_z_random = classifier_1(z_random_sample)
            Y_pred_val_CRT_random = classifier_CRT(w_val_z_random)

            sampled_probs = torch.softmax(Y_pred_val_CRT_random, dim=1)
            sampled_confidences_probs = np.maximum(sampled_probs.cpu().data.squeeze().numpy()[:, 0],
                                                   sampled_probs.cpu().data.squeeze().numpy()[:, 1])

            sampled_out_probs[:, i] = sampled_confidences_probs
            sampled_preds[:, i] = np.argmax(sampled_probs.cpu().data.squeeze().numpy(), axis=1)
            logits_out[:, i] = np.maximum(Y_pred_val_CRT_random.cpu().data.squeeze().numpy()[:, 0],
                                          Y_pred_val_CRT_random.cpu().data.squeeze().numpy()[:, 1])

    predicted_probs_confid.append(confidences_probs)
    pred_labels_CRT_sample_all = np.concatenate(sampled_preds)
    probs_CRT_sample_all = np.concatenate(sampled_out_probs)
    logits_out_all = np.concatenate(logits_out)

    fpr, tpr, thresholds = roc_curve(Y_GT_sample, logits_out_all)
    idx = np.argmax(tpr - fpr)
    optimal_th = thresholds[idx]
    y_CRT_pred_zo = np.zeros_like(logits_out_all)
    y_CRT_pred_zo[logits_out_all > optimal_th] = 1

    tn, fp, fn, tp = confusion_matrix(Y_GT_sample, y_CRT_pred_zo).ravel()
    sen_epi = tp / (tp + fn)
    spe_epi = tn / (tn + fp)
    bacc_epi_uc = 0.5 * (sen_epi + spe_epi)

    x = 0  # if 1 dont include original segmentation into CI bands
    y = samples
    CI_patient = (np.zeros_like(GT_samples)).astype(float)
    CI_patient_0 = np.zeros_like(GT_samples).astype(float)

    for ii in range(len(Y_CRT[test_index])):
        CI_patient[ii] = (sum(y_CRT_pred_zo[x:y]))/samples
        CI_patient_0[ii] = (100 - CI_patient[ii])
        x = x + 20
        y = y + 20

    fpr, tpr, thresholds = roc_curve(Y_CRT[test_index], logits_out[:, 0])
    idx = np.argmax(tpr - fpr)
    optimal_th = thresholds[idx]
    y_CRT_pred_zo_orig = np.zeros_like(logits_out[:, 0])
    y_CRT_pred_zo_orig[logits_out[:, 0] > optimal_th] = 1

    tn, fp, fn, tp = confusion_matrix(Y_CRT[test_index], y_CRT_pred_zo_orig).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    bacc_orig = 0.5 * (sen + spe)

    print('Fold: {} --> BACC: {:.2f}, SEN: {:.2f}, SPE: {:.2f}'.format(kfold, bacc_orig, sen, spe))
    print('Fold: {} --> BACC: {:.2f}, SEN: {:.2f}, SPE: {:.2f}'.format(kfold, bacc_epi_uc, sen_epi, spe_epi))

    CI_patient_all.append(CI_patient)
    CI_patient_0_all.append(CI_patient_0)
    Probs_positive_responder_all.append(probs[:, 1].cpu().data.squeeze().numpy())
    Probs_negative_responder_all.append(probs[:, 0].cpu().data.squeeze().numpy())
    y_pred_all.append(y_CRT_pred_zo)
    Y_sample_GT_all.append(Y_GT_sample)
    y_true_all.append(Y_CRT[test_index])
    predicted_preds_orig_all.append(y_CRT_pred_zo_orig)
    probs_CRT_sample_all_total.append(probs_CRT_sample_all)

predicted_probs_responders = np.concatenate(predicted_probs_responders)
Probs_positive_responder_all = np.concatenate(Probs_positive_responder_all)
Probs_negative_responder_all = np.concatenate(Probs_negative_responder_all)

y_samples_true_all = np.concatenate(Y_sample_GT_all)
y_samples_pred_all = np.concatenate(y_pred_all)
eid_all = np.concatenate(eid_all)
predicted_probs_total = np.concatenate(predicted_probs_confid)
First_prediction_baseline_model_all = np.concatenate(predicted_preds_orig_all)
First_GT_baseline_model_all = np.concatenate(y_true_all)
CI_patient_all = np.concatenate(CI_patient_all)
CI_patient_0_all = np.concatenate(CI_patient_0_all)
probs_CRT_sample_all_total = np.concatenate(probs_CRT_sample_all_total)

Volume_All = np.concatenate(Volume_All)
EF_All = np.concatenate(EF_All)

tn, fp, fn, tp = confusion_matrix(First_GT_baseline_model_all, First_prediction_baseline_model_all).ravel()
sen = tp / (tp + fn)
spe = tn / (tn + fp)
bacc = 0.5 * (sen + spe)

print('Overall --> BACC: {:.2f}, SEN: {:.2f}, SPE: {:.2f}'.format(bacc, sen, spe))

outcomes = pd.DataFrame({"Sen_Sen_Acc": [sen, spe, bacc]})
outcomes.to_csv(r'{0}/Sen_Sen_Acc_Outer_{1}_{2}_model.csv'.format(save_dir, num_hidden, kld_weight), header=True,
                index=False)

ground_truth_CRT = First_GT_baseline_model_all
predictions_CRT = First_prediction_baseline_model_all
correct = predictions_CRT == ground_truth_CRT
df_correct = pd.DataFrame({"Correct": correct})
df_probs = pd.DataFrame({"Probability in predicted outcome ": predicted_probs_total})
df_preds = pd.DataFrame({"Predicted outcome ": predictions_CRT})

df_frame_reliability_ace = pd.concat([df_probs, df_preds, df_correct], axis=1)
df_frame_reliability_ace.to_csv(
    r'{0}/Model_Outcomes_{1}.csv'.format(save_dir, model_name), header=True)

AECE, AMCE, cof_min, cof_max, confidence, accuracy, n_bins = AdaptiveBinning(df_frame_reliability_ace, True)
np.savetxt('{0}/AECE_Orig.txt'.format(root_save_dir),
           [AECE])
print('AECE Orig: ', AECE)

data = compute_overconfidence_calibration_ece(ground_truth_CRT, predictions_CRT, predicted_probs_total, bin_size)
ece = data["expected_calibration_error"]
oe = data["over_confidence_calibration_error"]
mce = data["maximum_calibration_error"]
ece_loss4 = math.log(ece)
ece_loss4 = torch.tensor(ece_loss4).float()
print('ECE Loss Orig: ', ece)

oce_loss = math.log(oe)
oce_loss = torch.tensor(oce_loss).float()
print('OE Loss Orig: ', oe)
np.savetxt('{0}/OE.txt'.format(save_dir),
           [oe])

np.savetxt('{0}/ECE.txt'.format(save_dir),
           [ece])

np.savetxt('{0}/MCE.txt'.format(save_dir),
           [mce])

print('MCE Orig: ', mce)

b_score = brier_score_loss(ground_truth_CRT, predictions_CRT)
print('BScore Orig: ', b_score)
np.savetxt('{0}/BScore_GT.txt'.format(save_dir), [b_score])

data_epi = compute_overconfidence_calibration_ece(y_samples_true_all, y_samples_pred_all, probs_CRT_sample_all_total,
                                                  bin_size)
correct2 = y_samples_true_all == y_samples_pred_all
df_correct2 = pd.DataFrame({"Correct": y_samples_true_all})
df_probs2 = pd.DataFrame({"Probability in predicted outcome ": probs_CRT_sample_all_total})
df_preds2 = pd.DataFrame({"Predicted outcome ": y_samples_pred_all})

df_frame_reliability_ace2 = pd.concat([df_probs2, df_preds2, df_correct2], axis=1)
df_frame_reliability_ace2.to_csv(
    r'{0}/Model_Outcomes_Epi_{1}.csv'.format(save_dir, model_name), header=True)

AECE_epi, AMCE, cof_min, cof_max, confidence, accuracy, n_bins = AdaptiveBinning(df_frame_reliability_ace2, True)

np.savetxt('{0}/AECE_Epi.txt'.format(root_save_dir),
           [AECE_epi])

ece_epi = data_epi["expected_calibration_error"]
oe_epi = data_epi["over_confidence_calibration_error"]
mce_epi = data_epi["maximum_calibration_error"]

ece_loss_epi = math.log(ece_epi)
ece_loss4_epi = torch.tensor(ece_loss_epi).float()
print('ECE_Epi: ', ece_epi)
oce_loss_epi = math.log(oe_epi)
oce_loss_epi = torch.tensor(oce_loss_epi).float()

print('OE Loss_Epi: ', oe_epi)

np.savetxt('{0}/ECE_Epi.txt'.format(save_dir),
           [ece_epi])

np.savetxt('{0}/OE_Epi.txt'.format(save_dir),
           [oe_epi])

np.savetxt('{0}/MCE_Epi.txt'.format(save_dir),
           [mce_epi])

print('MCE_Epi: ', mce_epi)

b_score_epi = brier_score_loss(y_samples_true_all, y_samples_pred_all)
print('BScore Epi: ', b_score_epi)
np.savetxt('{0}/BScore_Epi.txt'.format(save_dir), [b_score_epi])
