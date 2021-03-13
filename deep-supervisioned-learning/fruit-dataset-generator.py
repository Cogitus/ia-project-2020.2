# https://stackoverflow.com/questions/65187875/how-do-i-get-a-random-image-from-a-folder-python/65188051
# https://stackoverflow.com/questions/53551410/how-to-randomly-select-images-and-put-them-to-multiple-folders-in-python
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import os
import glob
import random
import shutil

path = os.getcwd()

sample_carambolas = random.sample(glob.glob(path+"/deep-supervisioned-learning/carambola/*.png"), 2000)
sample_pear = random.sample(glob.glob(path+"/deep-supervisioned-learning/pear/*.png"), 2000)
sample_plum = random.sample(glob.glob(path+"/deep-supervisioned-learning/plum/*.png"), 2000)

"""
2000 = 100%
1800 = 90%      Training set
200 = 10%       Validation & Test set
"""

def create_trainset_testset_from_fullset(full_set, n_samples):
    jump_size = int(len(full_set) / n_samples)
    full_set_idx_to_pop = []
    test_set = []

    for i in range(n_samples):
        next_sample_idx = ((i+1)*jump_size)-1
        test_set.append(full_set[next_sample_idx])
        full_set_idx_to_pop.append(next_sample_idx)

    training_set = [i for j, i in enumerate(full_set) if j not in full_set_idx_to_pop]

    return training_set, test_set

carambolas_training_set, carambolas_test_set = create_trainset_testset_from_fullset(sample_carambolas, 200)
pear_training_set, pear_test_set = create_trainset_testset_from_fullset(sample_pear, 200)
plum_training_set, plum_test_set = create_trainset_testset_from_fullset(sample_plum, 200)

# Carambola
for img in enumerate(carambolas_training_set, 0):
    destination = path+"/deep-supervisioned-learning/data/train/carambola"
    shutil.copy(img[1], destination)

for img in enumerate(carambolas_test_set, 0):
    destination = path+"/deep-supervisioned-learning/data/validation/carambola"
    shutil.copy(img[1], destination)

# Pear
for img in enumerate(pear_training_set, 0):
    destination = path+"/deep-supervisioned-learning/data/train/pear"
    shutil.copy(img[1], destination)

for img in enumerate(pear_test_set, 0):
    destination = path+"/deep-supervisioned-learning/data/validation/pear"
    shutil.copy(img[1], destination)

# Plum
for img in enumerate(plum_training_set, 0):
    destination = path+"/deep-supervisioned-learning/data/train/plum"
    shutil.copy(img[1], destination)

for img in enumerate(plum_test_set, 0):
    destination = path+"/deep-supervisioned-learning/data/validation/plum"
    shutil.copy(img[1], destination)


print("\n\nDone!!!")