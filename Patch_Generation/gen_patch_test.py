import openslide
from matplotlib import pyplot as plt

normSlidePath = '/Users/jbogahawatte/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/My_PhD_Documents/MND_Research' \
         '/MIL_code/Raw_dataset/train/ALS_001.tif'
tslide = openslide.open_slide(normSlidePath)
subSlide = tslide.read_region((0, 0), 0, tslide.level_dimensions[0])

plt.imshow(subSlide)
plt.show()
