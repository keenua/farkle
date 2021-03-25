import cv2
from imutils import paths
import numpy as np
from os import remove

DIR = 'train'

def dhash(image, hashSize=8):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

imagePaths = list(paths.list_images(DIR))

print(f'Was: {len(imagePaths)}')

hashes = {}
# loop over our image paths
for imagePath in imagePaths:
	# load the input image and compute the hash
	image = cv2.imread(imagePath)
	h = dhash(image)
	# grab all image paths with that hash, add the current image
	# path to it, and store the list back in the hashes dictionary
	p = hashes.get(h, [])
	p.append(imagePath)
	hashes[h] = p

for v in hashes.values():
    if len(v) > 1:
        images = [(i, cv2.imread(i)) for i in v]

        groups = []

        for path, image in images:
            found = False
            for g in groups:
                if np.array_equal(g[0][1], image):
                    g.append((path, image))
                    found = True
                    break
            if not found:
                groups.append([(path, image)])
        
        for g in groups:
            for p, _ in g[1:]:
                remove(p)

imagePaths = list(paths.list_images(DIR))

print(f'Now: {len(imagePaths)}')