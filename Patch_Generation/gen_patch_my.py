import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff

als_image = tiff.imread('/Users/jbogahawatte/Library/CloudStorage/OneDrive-TheUniversityofMelbourne'
                        '/My_PhD_Documents/MND_Research/MIL_code/Raw_dataset/test/005.tif')

# for img in range(als_image.shape[0]):
#     large_image = als_image[img]

patches_img = patchify(als_image, (224, 224), step=224)

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, :, :]
        tiff.imwrite('/Users/jbogahawatte/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/My_PhD_Documents'
                     '/MND_Research/MIL_code/patches/test/005/' + '005_' + str(i) + str(j) + '.tif',
                     single_patch_img)
