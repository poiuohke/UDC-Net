from SimpleITK import sitkNearestNeighbor, ResampleImageFilter, SmoothingRecursiveGaussianImageFilter, \
    GetArrayFromImage, GetImageFromArray, sitkLinear
from skimage import morphology, measure, segmentation, filters
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import numpy as np

trash_threshold = .2

def normalize(img_arr):
    max_hu = 400.
    min_hu = -1000.
    img_arr[img_arr > max_hu] = max_hu
    img_arr[img_arr < min_hu] = min_hu
    img_arr_normalized = (img_arr - min_hu) / (max_hu - min_hu)
    return img_arr_normalized


def resample_image(sitk_img, new_spacing, new_size, method='Linear'):
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    resampler = ResampleImageFilter()
    resampler.SetOutputDirection(direction)
    resampler.SetOutputOrigin(origin)
    resampler.SetSize(new_size)
    if method == 'Linear':
        resampler.SetInterpolator(sitkLinear)
    else:
        resampler.SetInterpolator(sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    return resampler.Execute(sitk_img)


def gaussian_smooth(sitk_img, sigma=1.5):
    img_filter = SmoothingRecursiveGaussianImageFilter()
    img_filter.SetSigma(float(sigma))
    return img_filter.Execute(sitk_img)

def lung_segmentation(sitk_img, lower_bound, upper_bound):
    new_spacing = np.asarray([2.5, 2.5, 5])
    orig_size = sitk_img.GetSize()
    orig_spacing = sitk_img.GetSpacing()
    new_size = [int(np.ceil(orig_size[0] / new_spacing[0] * orig_spacing[0])),
                int(np.ceil(orig_size[1] / new_spacing[1] * orig_spacing[1])),
                int(np.ceil(orig_size[2] / new_spacing[2] * orig_spacing[2]))]
    new_sitk_img = resample_image(sitk_img, new_spacing, new_size)
    new_sitk_img = gaussian_smooth(new_sitk_img)
    imgs_to_process = GetArrayFromImage(new_sitk_img)

    imgs_to_process[imgs_to_process < lower_bound] = lower_bound
    binary_threshold = filters.threshold_otsu(imgs_to_process)
    img = imgs_to_process < binary_threshold

    old_bbox = imgs_to_process.shape
    del imgs_to_process
    temp = np.zeros(old_bbox)
    for c in range(old_bbox[0]):
        labels = ~img[c, :, :]
        if np.sum(labels):
            labels = measure.label(labels, neighbors=4)
            regions = measure.regionprops(labels)
            labels = [r.area for r in regions]
            index = labels.index(max(labels))
            bbox = regions[index].bbox
            dist = 1
            temp[c, bbox[0] + dist:bbox[2] - dist, bbox[1] + dist:bbox[3] - dist] = segmentation.clear_border(
                img[c, bbox[0] + dist:bbox[2] - dist, bbox[1] + dist:bbox[3] - dist])
    img = temp > 0
    del temp
    otsu_img = img.copy()

    img = morphology.binary_closing(img, selem=np.ones((1, 2, 2)))

    labels = measure.label(img, neighbors=4)
    regions = measure.regionprops(labels)

    labels = [(r.area, r.bbox) for r in regions]
    labels.sort(reverse=True)
    max_bbox = labels[0][1]
    max_bbox_zmin = max_bbox[0]
    max_bbox_zmax = max_bbox[3]-1
    for i in range(int(max_bbox_zmax - (max_bbox_zmax - max_bbox_zmin) / 3),  max_bbox_zmax):
        _slice = img[i, :, :]
        slice_labels, num = measure.label(_slice, return_num=True)
        regions = measure.regionprops(slice_labels)
        slice_labels = [[r.area, r.label] for r in regions]
        if len(slice_labels) > 2:
            slice_labels.sort(reverse=True)
            max_area = slice_labels[0][0]
            _slice = _slice.astype(np.bool)
            thresh = int(max_area) / 4
            _slice = morphology.remove_small_objects(_slice, thresh)
            img[i, :, :] = _slice

    img = img.astype(np.bool)
    labels = measure.label(img, neighbors=4)
    regions = measure.regionprops(labels)
    labels = [(r.area, r.bbox, r.coords) for r in regions]
    labels.sort(reverse=True)
    max_area = labels[0][0]
    max_bbox = labels[0][1]
    max_bbox_zmin = max_bbox[0]
    max_bbox_zmax = max_bbox[3] - 1
    for area, bbox, coords in labels:
        region_center_z = (bbox[0]+bbox[3])/2
        if area > max_area / 2:
            continue
        if region_center_z > max_bbox_zmax or region_center_z < max_bbox_zmin:
            img[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
    _slice = np.sum(img, axis=0) > 0
    slice_labels, num = measure.label(_slice, return_num=True)
    if num > 1:
        regions = measure.regionprops(slice_labels)
        slice_labels = [r.area for r in regions]
        slice_labels.sort(reverse=True)
        max_area = slice_labels[0]
        _slice = _slice.astype(np.bool)
        thresh = int(max_area) / 4
        _slice = morphology.remove_small_objects(_slice, thresh)
        bbox = np.where(_slice)
        x_min = np.min(bbox[1])
        x_max = np.max(bbox[1])
        y_min = np.min(bbox[0])
        y_max = np.max(bbox[0])
        temp = np.zeros(img.shape)
        temp[:, y_min:y_max, x_min:x_max] = img[:, y_min:y_max, x_min:x_max]
        img = temp

    img = img > 0
    del otsu_img

    img = morphology.dilation(img, selem=np.ones((2, 3, 3)))
    img = img.astype(np.uint32)

    mask_img = GetImageFromArray(img)
    mask_img.CopyInformation(new_sitk_img)
    sitk_img = resample_image(mask_img, orig_spacing, orig_size, 'near')
    img = GetArrayFromImage(sitk_img)

    binary_img = np.where(img)
    z_min, z_max = (binary_img[0].min(), binary_img[0].max())
    y_min, y_max = (binary_img[1].min(), binary_img[1].max())
    x_min, x_max = (binary_img[2].min(), binary_img[2].max())
    bbox = [z_min, y_min, x_min, z_max, y_max, x_max]

    return img, bbox
