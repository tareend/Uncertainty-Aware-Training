# **********************************************************************************************************************
# Imports
# **********************************************************************************************************************

import os
import numpy as np
import nibabel as nib
from skimage import measure
import SimpleITK as sitk

# **********************************************************************************************************************
# Functions
# **********************************************************************************************************************

def segmentation_roi_coords(Y, roi_size):
    """ Returns coordinates based on centering segmented objects.

    Parameters
    ----------
    Y - input array - img_x x img_y x img_slices
    roi_size - 2-tuple size of ROI

    Notes
    -----
    Segmentation map assumed to have background of 0

    """
    coord_list = []
    for s in range(Y.shape[2]):
        img = Y[:, :, s]
        img = getLargestCC(img == 1).astype(int) + 2 * getLargestCC(img == 2).astype(int) + 3 * getLargestCC(
            img == 3).astype(int)
        nz = np.nonzero(img)
        if len(nz[0]) == 0:
            # segmentation map is all 0
            coord_list.append(None)
        else:
            x_min, x_max = np.min(nz[0]), np.max(nz[0])
            y_min, y_max = np.min(nz[1]), np.max(nz[1])
            x_mid = (x_max + x_min) // 2
            y_mid = (y_max + y_min) // 2
            x_start = x_mid - roi_size[0] // 2
            if x_start < 0:
                x_start = 0
            x_end = x_start + roi_size[0]
            y_start = y_mid - roi_size[1] // 2
            if y_start < 0:
                y_start = 0
            y_end = y_start + roi_size[1]
            coord_list.append(((x_start, x_end), (y_start, y_end)))
    return coord_list


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    if labels.max() == 0:  # assume at least 1 CC
        return segmentation
    else:
        largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largest_cc


def resample_by_spacing(im, new_spacing, interpolator=sitk.sitkLinear, keep_z_spacing=False):
    '''
    resample by image spacing
    :param im: sitk image
    :param new_spacing: new image spa
    :param interpolator: sitk.sitkLinear, sitk.NearestNeighbor
    :return:
    '''

    scaling = np.array(new_spacing) / (1.0 * (np.array(im.GetSpacing())))
    new_size = np.round((np.array(im.GetSize()) / scaling)).astype("int").tolist()
    origin_z = im.GetSize()[2]
    old_size = im.GetSize()

    if keep_z_spacing:
        new_size[2] = origin_z
    if not keep_z_spacing and new_size[2] == origin_z:
        print('shape along z axis does not change')

    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())

    image_resampled = sitk.Resample(im, new_size, transform, interpolator, im.GetOrigin(), new_spacing,
                                    im.GetDirection())
    new_size = image_resampled.GetSize()
    if new_size[2] != old_size[2]:
        print('A')

    return image_resampled


def image_roi(X, Y, roi_size, offset=None, num_slices=1):
    """ Returns imagespace region of interest

    Crops using coordinates of provided segmentation Y

    Parameters
    ----------
    X - image - img_x x img_y x img_slices x time_index
    Y - segmentation - img_x x img_y x img_slices
    roi_size - 2-tuple size of ROI, or int for square
    offset - 2-tuple offset from centre of segmentation
    num_slices - number of slices to provide in cropped image

    Returns
    -------
    roi_x x roi_y x img_slices' x time_index array of cropped images
    roi_x x roi_y x img_slices' x time_index array of cropped segmentations

    Notes
    -----
    """
    if isinstance(roi_size, int):
        roi_size = (roi_size, roi_size)
    if not offset:
        offset = (0, 0)

    pixels_seg_per_slice = np.sum(np.sum(Y[:, :, :, 0] == 1, axis=0), axis=0) + np.sum(
        np.sum(Y[:, :, :, 0] == 2, axis=0), axis=0) + np.sum(np.sum(Y[:, :, :, 0] == 3, axis=0), axis=0)
    slices_with_seg = np.where(pixels_seg_per_slice != 0)[0]
    mid_slice = int((slices_with_seg.max() + slices_with_seg.min()) / 2) + 1
    selected_slices = [mid_slice - 2, mid_slice - 1, mid_slice]
    print(selected_slices)
    coord_list = segmentation_roi_coords(Y[..., 0], roi_size)
    cropped_img_slice_list = []
    cropped_seg_slice_list = []
    for my_slice in selected_slices:
        coords = coord_list[my_slice]
        if coords:
            x_start = coords[0][0] + offset[0]
            x_end = coords[0][1] + offset[0]
            y_start = coords[1][0] + offset[1]
            y_end = coords[1][1] + offset[1]
            X_s = X[x_start:x_end, y_start:y_end, my_slice, :]
            Y_s = Y[x_start:x_end, y_start:y_end, my_slice, :]
            cropped_img_slice_list.append(X_s)
            cropped_seg_slice_list.append(Y_s)
    X_c = np.array(cropped_img_slice_list)
    X_c = np.swapaxes(X_c, 0, 1)
    X_c = np.swapaxes(X_c, 1, 2)

    Y_c = np.array(cropped_seg_slice_list)
    Y_c = np.swapaxes(Y_c, 0, 1)
    Y_c = np.swapaxes(Y_c, 1, 2)

    return X_c, Y_c


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def load_nifti_img(subject_img_filepath):
    temp_image = sitk.ReadImage(subject_img_filepath)
    temp_image = sitk.Cast(sitk.RescaleIntensity(temp_image), sitk.sitkFloat32)
    return temp_image

# **********************************************************************************************************************
# Here we show how we worked with the 20 segmentations obtained from U-net Segmentation model to create our npy files for the models#
# **********************************************************************************************************************

Q_original = np.load('File path of original npy of images')
CRT_FOLDER = 'Path to Folder containing all the Segmentations obtained using Unet'
eid_list = sorted(os.listdir(CRT_FOLDER)) ## Extracts the names of all subjects to ensure a consitent iterating/saving method.

save_dir = 'Path where user would like to store files'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

OUT_NB_FRAMES = 25
normalize_trigger_time = np.linspace(0, OUT_NB_FRAMES, OUT_NB_FRAMES) / OUT_NB_FRAMES
img_dim = 80
new_spacing = [2.5, 2.5, 10]
keep_z_spacing = True
number_droputs = 20 # Refers to number of segmentations in our work.
eids_img_CRT_dropout = np.zeros((len(eid_list) * number_droputs), dtype=object)
sax_seg_CRT_dropout = np.zeros((len(eid_list) * number_droputs, 3, img_dim, img_dim, OUT_NB_FRAMES))

k = 0

for i, eid in enumerate(eid_list):
    print('{0}: {1}'.format(i, eid))
    eid_folder = os.path.join(CRT_FOLDER, str(eid))
    image_path = '{0}/sa_seg_Unet_00.nii.gz'.format(eid_folder)
    nim = nib.load(image_path)
    hdr = dict(nim.header)
    image = nim.get_fdata()
    X, Y, Z, NB_FRAMES = image.shape
    numpy_folder = os.path.join(eid_folder, 'npy')
    for it in range(number_droputs):
        if it == 0:
            seg = Q_original[i]
        else:
            seg = np.load(os.path.join(numpy_folder, 'seg_seg_res_80_{:02d}.npy'.format(it)))
            seg = seg.transpose(2, 0, 1, 3)
        sax_seg_CRT_dropout[k] = seg
        eids_img_CRT_dropout[k] = eid

        k += 1

np.save(os.path.join(save_dir, 'sax_seg_CRT.npy'), sax_seg_CRT_dropout) ## get all segmentations in .npy file to pass to model for training/testing
np.save(os.path.join(save_dir, 'eids_img_CRT.npy'), eids_img_CRT_dropout)

## If there is raw nii.gz data call image_roi function to obtain the relevant slices and if re-sampling required call respective function included in this file ##