import os
import random

import matplotlib.pyplot as plt
import numpy as np
import openslide
import glob
from multiprocessing import Pool, Value, Lock
from skimage.filters import threshold_otsu
import cv2
import PIL.Image as Image

Image.MAX_IMAGE_PIXELS = None

####======================================    User Configuration
num_thread = 4
patch_dimension_level = 0 ## 0: 40x, 1: 20x
patch_level_list = [0]  #[1,2,2]
stride = 256
psize = 256
psize_list = [256] #[256, 192, 256]

tissue_mask_threshold = 0.1
mask_dimension_level = 0

slides_folder_dir = '/Users/jbogahawatte/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/My_PhD_Documents' \
                    '/MND_Research/MIL_code/Raw_dataset/train'
slide_paths = glob.glob(os.path.join(slides_folder_dir, '*.tif'))  # change the surfix '.tif' to other if necessary
save_folder_dir = '/Users/jbogahawatte/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/My_PhD_Documents' \
                  '/MND_Research/MIL_code/patches/train2'
####======================================




#mask_level: 5, 1/32
def get_roi_bounds(tslide, isDrawContoursOnImages=False, mask_level=5, cls_kernel=50, open_kernal=30):

    subSlide = tslide.read_region((0, 0), mask_level, tslide.level_dimensions[mask_level])
    subSlide_np = np.array(subSlide)
    # subMask = subSlide.convert('L')
    # subMask_np = np.array(subMask)
    # plt.imshow(subMask_np)
    # plt.show()

    hsv = cv2.cvtColor(subSlide_np, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    try:
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
    except:
        return np.NaN, np.NaN

    minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    maxhsv = np.array([180, 255, vthresh], np.uint8)
    thresh = [minhsv, maxhsv]

    # extraction the countor for tissue
    mask = cv2.inRange(hsv, thresh[0], thresh[1])

    close_kernel = np.ones((cls_kernel, cls_kernel), dtype=np.uint8)
    image_close_img = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
    open_kernel = np.ones((open_kernal, open_kernal), dtype=np.uint8)
    image_open_np = cv2.morphologyEx(np.array(image_close_img), cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBox = [cv2.boundingRect(c) for c in contours]
    boundingBox = [sst for sst in boundingBox if sst[2] > 150 and sst[3] > 150]

    print(boundingBox)
    print('boundingBox number ', len(boundingBox))

    if isDrawContoursOnImages:
        line_color = (0, 0, 0)  # blue color code
        contours_rgb_image_np = np.array(subSlide)
        cv2.drawContours(contours_rgb_image_np, contours, -1, line_color, 50)
        contours_rgb_image_np = cv2.resize(contours_rgb_image_np, (0, 0), fx=0.2, fy=0.2)
        countours_rgb_image_img = Image.fromarray(contours_rgb_image_np.astype(np.uint8))
        countours_rgb_image_img.show()

    return image_open_np, boundingBox



def Extract_Patch_From_Slide_STRIDE(tslide:openslide.ImageSlide, tissue_mask, patch_save_folder, patch_level, mask_level, patch_stride, patch_size, threshold, level_list=[1], patch_size_list=[256], patch_surfix='png'):

    assert patch_level == level_list[0]
    assert patch_size == patch_size_list[0]

    mask_sH, mask_sW = tissue_mask.shape
    print(f'tissue mask shape {tissue_mask.shape}')

    mask_patch_size = patch_size // pow(2, mask_level-patch_level)
    mask_patch_size_square = mask_patch_size ** 2
    mask_stride = patch_stride // pow(2, mask_level-patch_level)

    #mag_factor = pow(2, mask_level-patch_level)
    mag_factor = pow(2, mask_level)    # !!
    print(f'slide level dimensions {tslide.level_dimensions}, mask patch size {mask_patch_size}, mask stride {mask_stride}, mag_factor {mag_factor}')

    tslide_name = os.path.basename(patch_save_folder)
    num_error = 0

    for iw in range(mask_sW//mask_stride):
        for ih in range(mask_sH//mask_stride):
            ww = iw * mask_stride
            hh = ih * mask_stride
            if (ww+mask_patch_size) < mask_sW and (hh+mask_patch_size) < mask_sH:
                # tmask = tissue_mask[hh:hh+mask_patch_size, ww:ww+mask_patch_size]
                # plt.imshow(tmask)
                # plt.show()
                # mRatio = float(np.sum(tmask > 0)) / mask_patch_size_square
                # print(mRatio)
                #
                # if mRatio > threshold:
                tsave_folder = getFolder_name(patch_save_folder, 0, 256)

                sww = ww * mag_factor
                shh = hh * mag_factor

                cW_l0 = sww + (patch_size // 2) * pow(2, patch_level)
                cH_l0 = shh + (patch_size // 2) * pow(2, patch_level)

                tlW_l0 = cW_l0 - (256 // 2) * pow(2, 0)
                tlH_l0 = cH_l0 - (256 // 2) * pow(2, 0)
                tpatch = tslide.read_region((0, 0), 0, (256, 256))
                plt.imshow(np.array(tpatch))
                plt.show()
                    ## (x, y) tuple giving the top left pixel in the level 0 reference frame
                tname = f'{tslide_name}_{ww * mag_factor}_{hh * mag_factor}_{iw}_{ih}_WW_{mask_sW // mask_stride}_HH_{mask_sH // mask_stride}.{patch_surfix}'
                tpatch.save(os.path.join(tsave_folder, tname))



def getFolder_name(orig_dir, level, psize):
    tslide = os.path.basename(orig_dir)
    folderName = os.path.dirname(orig_dir)

    subfolder_name = float(psize * level) / 256
    tfolder = os.path.join(folderName, str(subfolder_name*10), tslide)
    return tfolder

def read_tumor_mask(mask_path, mask_dimension_level):
    tmask = openslide.open_slide(mask_path)
    subMask = tmask.read_region((0, 0), mask_dimension_level, tmask.level_dimensions[mask_dimension_level])
    subMask = subMask.convert('L')
    subMask_np = np.array(subMask)

    return subMask_np

def Thread_PatchFromSlides(args):

    normSlidePath, slideName, tsave_slide_dir = args

    for tlevel, tsize in zip(patch_level_list, psize_list):
        tsave_dir_level = getFolder_name(tsave_slide_dir, tlevel, tsize)
        if not os.path.exists(tsave_dir_level):
            os.makedirs(tsave_dir_level)

    tslide = openslide.open_slide(normSlidePath)
    tissue_mask, boundingBoxes = get_roi_bounds(tslide, isDrawContoursOnImages=False, mask_level=mask_dimension_level)  # mask_level: absolute level
    tissue_mask = tissue_mask // 255
    # plt.imshow(tissue_mask)
    # plt.show()

    Extract_Patch_From_Slide_STRIDE(tslide, tissue_mask, tsave_slide_dir,
                                                  patch_level=patch_dimension_level, mask_level=mask_dimension_level,
                                                  patch_stride=stride, patch_size=psize,
                                                  threshold=tissue_mask_threshold,
                                                  level_list=patch_level_list,
                                                  patch_size_list=psize_list
                                                  )


if __name__ == "__main__":
    pool = Pool(processes=num_thread)
    arg_list = []

    for tSlidePath in slide_paths:
        slideName = os.path.basename(tSlidePath).split('.')[0]
        tSave_slide_dir = os.path.join(save_folder_dir, slideName)
        arg_list.append([tSlidePath, slideName, tSave_slide_dir])
        Thread_PatchFromSlides(arg_list)