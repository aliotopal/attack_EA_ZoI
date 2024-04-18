import time
from typing import Optional
import os, cv2
import numpy as np
import sys
import random

# random.seed(0)


# Do NOT touch the class.
class EA:
    def __init__(self, klassifier, max_iter, confidence, targeted):
        self.klassifier = klassifier
        self.max_iter = max_iter
        self.confidence = confidence
        self.targeted = targeted
        self.pop_size = 40
        self.numberOfElites = 10
        self.roi = roi

    @staticmethod
    def _get_class_prob(preds: np.ndarray, class_no: np.array) -> np.ndarray:
        """
        :param preds: an array of predictions of individuals for all the categories: (40, 1000) shaped array
        :param class_no: for the targeted attack target category index number; for the untargeted attack ancestor
        category index number
        :return: an array of the prediction of individuals only for the target/ancestor category: (40,) shaped  array
        """
        # print("\nPREDS: ", preds[:, class_no])
        return preds[:, class_no]

    @staticmethod
    def _get_fitness(probs: np.ndarray) -> np.ndarray:
        """
         It simply returns the CNN's probability for the images but different objective functions can be used here.
        :param probs: an array of images' probabilities of selected CNN
        :return: returns images' probabilities in an array (40,)
        """
        fitness = probs
        return fitness

    def _selection_untargeted(self, images: np.ndarray, fitness: np.ndarray):
        """
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images furthest from the ancestor category will be
        closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' propabilities of selected CNN
        :return: returns a tuple of elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        """
        idx_elite = fitness.argsort()[: self.numberOfElites]
        # print("IDX ELITE:", idx_elite)
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[self.numberOfElites: int(half_pop_size)]
        elite = images[idx_elite, :]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0] / 2 - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, random_keep

    def _selection_targeted(self, images: np.ndarray, fitness: np.ndarray):
        """
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images closest to the target category will be
        closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' probabilities of selected CNN
        :return: returns elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        """
        idx_elite = fitness.argsort()[-self.numberOfElites:]
        # print("IDX ELITE: ", idx_elite[::-1])
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[int(half_pop_size): -self.numberOfElites]
        elite = images[idx_elite, :][::-1]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0] / 2 - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, random_keep

    @staticmethod
    def _get_no_of_pixels(im_size: int) -> int:
        """
        :param im_size: Original inputs' size, represented by an integer value.
        :return: returns an integer that will be used to decide how many pixels will be mutated
        in the image during the current generation.
        """
        u_factor = np.random.uniform(0.0, 1.0)
        n = 60  # normally 60, the smaller n -> more pixels to mutate
        res = (u_factor ** (1.0 / (n + 1))) * im_size
        no_of_pixels = im_size - res
        return no_of_pixels

    @staticmethod
    def _mutation(
            _x: np.ndarray,
            no_of_pixels: int,
            mutation_group: np.ndarray,
            percentage: float,
            boundary_min: int,
            boundary_max: int,
    ) -> np.ndarray:
        """
        :param _x: An array with the original input to be attacked.
        :param no_of_pixels: An integer determines the number of pixels to mutate in the original input for the current
            generation.
        :param mutation_group: An array with the individuals which will be mutated
        :param percentage: A decimal number from 0 to 1 that represents the percentage of individuals in the mutation
            group that will undergo mutation.
        :param boundary_min: keep the pixel within [0, 255]
        :param boundary_max: keep the pixel within [0, 255]
        :return: An array of mutated individuals
        """
        mutated_group = mutation_group.copy()
        # np.random.shuffle(mutated_group)
        no_of_individuals = len(mutated_group)  # 20 individuals
        for individual in range(int(no_of_individuals * percentage)):
            locations_x = np.random.randint(x_array.shape[0], size=int(no_of_pixels))
            locations_y = np.random.randint(x_array.shape[1], size=int(no_of_pixels))
            locations_z = np.random.randint(x_array.shape[2], size=int(no_of_pixels))
            new_values: [int] = random.choices(np.array([-1, 1]), k=int(no_of_pixels))
            mutated_group[individual, locations_x, locations_y, locations_z] = (
                    mutated_group[individual, locations_x, locations_y, locations_z] - new_values
            )
        mutated_group = np.clip(mutated_group, boundary_min, boundary_max)
        # mutated_group = mutated_group % 200
        return mutated_group

    @staticmethod
    def _mutation_new(
            _x: np.ndarray,
            no_of_pixels: int,
            mutation_group: np.ndarray,
            percentage: float,
            boundary_min: int,
            boundary_max: int,
            roi: np.array
    ) -> np.ndarray:
        """
        :param _x: An array with the original input to be attacked.
        :param no_of_pixels: An integer determines the number of pixels to mutate in the original input for the current
            generation.
        :param mutation_group: An array with the individuals which will be mutated
        :param percentage: A decimal number from 0 to 1 that represents the percentage of individuals in the mutation
            group that will undergo mutation.
        :param boundary_min: keep the pixel within [0, 255]
        :param boundary_max: keep the pixel within [0, 255]
        :return: An array of mutated individuals
        """
        mutated_group = mutation_group.copy()
        # np.random.shuffle(mutated_group)
        no_of_individuals = len(mutated_group)  # 20 individuals
        for individual in range(int(no_of_individuals * percentage)):
            no_of_pixels = random.randrange(0, len(roi))
            roi_indx = np.random.choice(roi.shape[0], size=no_of_pixels)
            roi_new = roi[roi_indx]
            locations_x = np.array(roi_new)[:, 1]
            locations_y = np.array(roi_new)[:, 0]
            locations_z = np.random.randint(3, size=int(no_of_pixels))
            new_values: [int] = random.choices(np.array([-1, 1]), k=int(no_of_pixels))
            mutated_group[individual, locations_x, locations_y, locations_z] = (
                    mutated_group[individual, locations_x, locations_y, locations_z] - new_values
            )

            noise = mutated_group[individual] - _x
            noise = np.clip(noise, -epsilon, epsilon)
            mutated_group[individual] = _x + noise

        mutated_group = np.clip(mutated_group, boundary_min, boundary_max)
        # mutated_group = mutated_group % 200
        return mutated_group

    @staticmethod
    def _get_crossover_parents(crossover_group: np.ndarray) -> list:
        size = crossover_group.shape[0]  # size = 30
        no_of_parents = random.randrange(0, size, 2)  # gives random even number between 0 and size.
        parents_idx = random.sample(range(0, size), no_of_parents)
        return parents_idx  # returns parents indexs who will be used for corssover.

    @staticmethod
    def _crossover_new(_x: np.ndarray, crossover_group: np.ndarray, parents_idx: list, roi: np.ndarray) -> np.ndarray:
        ''' Randomly select the regions which will be swaped between two individuals. '''
        crossedover_group = crossover_group.copy()  # shape: (30, 224, 224, 3)
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            roi_3d = np.hstack((roi, np.random.randint(0, 3, size=(len(roi), 1))))
            no_of_pixels = random.randrange(0, len(roi))
            roi_indx = np.random.choice(roi_3d.shape[0], size=no_of_pixels)
            roi_3d_selected = roi_3d[roi_indx]
            z = np.random.randint(x_array.shape[2])
            coords_img1, coords_img2 = roi_3d_selected[:, :2], roi_3d_selected[:, :2]
            channels_img1, channels_img2 = roi_3d_selected[:, 2], roi_3d_selected[:, 2]

            # Swap pixel values between images
            img1_values = crossedover_group[parent_index_1, coords_img1[:, 1], coords_img1[:, 0], channels_img1]
            img2_values = crossedover_group[parent_index_2, coords_img2[:, 1], coords_img2[:, 0], channels_img2]
            crossedover_group[parent_index_1, coords_img1[:, 1], coords_img1[:, 0], channels_img1] = img2_values
            crossedover_group[parent_index_2, coords_img2[:, 1], coords_img2[:, 0], channels_img2] = img1_values
            # Print the first few swapped pixels for verification
            # print("Swapped pixels:")
            # for coord in roi_3d_selected[:5]:
            #     y, x, c = coord
            #     print("Pixel at ({}, {}, {}) in img1: {}".format(x, y, c, crossedover_group[parent_index_1,x, y, c]))
            #     print("Pixel at ({}, {}, {}) in img2: {}".format(x, y, c, crossedover_group[parent_index_2,x,  y, c]))




            # for coord in roi_3d_selected:
            #     x, y, c = coord
            #     temp = crossedover_group[parent_index_1, y, x, c].copy()
            #     crossedover_group[parent_index_1, y, x, c] = crossedover_group[parent_index_2, y, x, c]
            #     crossedover_group[parent_index_2, y, x, c] = temp




        return crossedover_group

    @staticmethod
    def _crossover(_x: np.ndarray, crossover_group: np.ndarray, parents_idx: list) -> np.ndarray:
        crossedover_group = crossover_group.copy()
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            crossover_range = int(x.shape[0] * 0.15)  # 15% of the image will be crossovered.
            size_x = np.random.randint(0, crossover_range)
            start_x = np.random.randint(0, _x.shape[0] - size_x)
            size_y = np.random.randint(0, crossover_range)
            start_y = np.random.randint(0, _x.shape[1] - size_y)
            z = np.random.randint(_x.shape[2])
            temp = crossedover_group[
                   parent_index_1,
                   start_x: start_x + size_x,
                   start_y: start_y + size_y,
                   z,
                   ]
            crossedover_group[
            parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y, z
            ] = crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z]
            crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z] = temp
        return crossedover_group

    def _generate(self, x, y: Optional[int] = None) -> np.ndarray:
        """
        :param x: An array with the original inputs to be attacked.
        :param y: An integer with the true or target labels.
        :return: An array holding the adversarial examples.
        """
        boundary_min = 0
        boundary_max = 255

        # img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])).copy()
        # img = Image.fromarray(x.astype(np.uint8))

        img_resized = x.resize((224, 224), resample=1)
        img_resized = img_to_array(img_resized)
        img_resized =img_resized.reshape((1, 224, 224, 3))

        img_r = preprocess_input(img_resized)
        preds = self.klassifier.predict(img_r)
        label0 = decode_predictions(preds)
        label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
        anc = label1[1]  # label
        anc_indx = np.argmax(preds)
        print("Before the image is:  " + anc + " --> " + str(label1[2]) + " ____ index: " + str(anc_indx))
        if self.targeted:
            print("Target class index number is: ", y)
        # images = np.array([x_array] * self.pop_size).astype(np.uint8)  # pop_size * ancestor images are created
        images = np.array([x_array] * self.pop_size)  # pop_size * ancestor images are created
        # convert HR images to 224x224 for evaluation
        # images = x * self.pop_size

        count = 1

        timing = open(f'timings_{ancestor}{j}.txt', 'a')
        timing.write("\t\t\tTiming Report\n")
        timing.write("Generation\tSelection\t\tMutation\t\tCrossover\t\tOverall\n")

        c_ts = []   # collection of c_t values through out the generations
        c_2best = []    # without the c_t label value, the best label values collection.


        while True:
            b0 = time.time()    # per generation

            b1 = time.time()    # selection/ evaluation
            imgResized = []
            for i in range(40):
                # img = Image.fromarray(imagesResized[i].astype(np.uint8))
                res = cv2.resize(images[i], (224, 224))#, interpolation=cv2.INTER_LANCZOS4)
                # res = img_to_array(res)
                imgResized.append(res)
            imgResized = np.array(imgResized)

            # images_r = images.copy()
            # imgResized = []
            # for i in range(40):
            #     res = cv2.resize(images_r[i], (224, 224))#, interpolation=cv2.INTER_LANCZOS4)
            #     # res = img_to_array(res)
            #     imgResized.append(res)
            # imgResized = np.array(imgResized)

            img_r = preprocess_input(imgResized)
            preds = self.klassifier.predict(img_r)  # predictions of 40 images (40, 1000)
            probs = self._get_class_prob(preds, y)  # target category values of images (40,)
            best_adv_indx = np.argmax(probs)        # best adversarial image according to the target label value

            dom_indx = np.argmax(preds[best_adv_indx])  # dominant label value of the best adversarial image

            adv_img = images[best_adv_indx]             # best adversarial image.

            print("\nc_t label value_new: ", preds[best_adv_indx][y])
            c_ts.append(preds[best_adv_indx][y])
            # Dominant category report ##################
            label0 = decode_predictions(preds, 1000)  # Reports predictions with label and label values of 40 images
            print(np.array(label0).shape)
            print("Best image: ", best_adv_indx+1)
            label1 = label0[best_adv_indx][0]  # Gets the Top1 label and label value of the best adversarial image.
            # print("label1: ", label0[best_adv_indx])
            dom_cat = label1[1]  # label
            dom_cat_prop = label1[2]  # label probability

            if dom_cat != target:
                c_2best.append(dom_cat_prop)
            else:
                label2 = label0[best_adv_indx][1]
                # dom_cat = label2[1]  # label
                dom_cat_prop_2 = label2[2]  # label probability
                c_2best.append(dom_cat_prop_2)
            ###########################################



            print(
                "\rgeneration: "
                + str(count)
                + "/"
                + str(self.max_iter)
                + " ______ "
                + dom_cat
                + ": "
                + str(dom_cat_prop)
                + " ____ index: "
                + str(dom_indx)
                + "___ct: "
                + str(preds[best_adv_indx][y])
            )


            # Stopping the algorithm criteria:
            if count == self.max_iter:
                # if algorithm can not create the adversarial image within "generation" stop the algorithm
                print("\nFailed to generate adversarial image within", self.max_iter, " generations")
                break
            if not self.targeted and dom_indx != anc_indx:
                print("\nAdversarial image is generated successfully in", count, "generations")
                break
            if self.targeted and dom_indx == y and dom_cat_prop > self.confidence:
                print("\nAdversarial image is generated successfully in", count, " generations")
                break

            percentage_middle_class = 1
            percentage_keep = 1

            # Select population classes based on fitness and create 'keep' group
            if self.targeted:
                probs = self._get_class_prob(preds, y)
                fitness = self._get_fitness(probs)
                (
                    elite,
                    middle_class,
                    random_keep,
                ) = self._selection_targeted(images, fitness)
            else:
                probs = self._get_class_prob(preds, anc_indx)
                fitness = self._get_fitness(probs)
                (
                    elite,
                    middle_class,
                    random_keep,
                ) = self._selection_untargeted(images, fitness)
            elite2 = elite.copy()
            keep = np.concatenate((elite2, random_keep))
            # Reproduce individuals by mutating Elits and Middle class---------
            # mutate and crossover individuals
            e1 = time.time()    # selection/ evaluation


            b2 = time.time()  # mutation time
            im_size = x_array.shape[0] * x_array.shape[1] * x_array.shape[2]
            no_of_pixels = self._get_no_of_pixels(im_size)
            mutated_middle_class = self._mutation_new(
                x,
                no_of_pixels,
                middle_class,
                percentage_middle_class,
                boundary_min,
                boundary_max,
                roi
            )
            mutated_keep_group1 = self._mutation_new(x, no_of_pixels, keep, percentage_keep, boundary_min, boundary_max, roi)
            mutated_keep_group2 = self._mutation_new(
                x,
                no_of_pixels,
                mutated_keep_group1,
                percentage_keep,
                boundary_min,
                boundary_max,
                roi
            )
            all_ = np.concatenate((mutated_middle_class, mutated_keep_group2))  # shape: (30, 224, 224, 3)
            e2 = time.time()  # mutation time


            b3 = time.time()  # crossover time
            parents_idx = self._get_crossover_parents(all_)
            crossover_group = self._crossover_new(x, all_, parents_idx, roi)  # shape: (30, 224, 224, 3)
            # Create new population
            images = np.concatenate((elite, crossover_group))
            e3 = time.time()  # crossover time


            e0 = time.time()  # per generation

            selection = "{:.5f}".format(e1 - b1)
            mutation = "{:.5f}".format(e2 - b2)
            crossover = "{:.5f}".format(e3 - b3)
            overall = "{:.5f}".format(e0 - b0)
            timing.write(f"{count}\t\t\t{selection}\t\t\t{mutation}\t\t\t{crossover}\t\t\t{overall}\n")

            count += 1

        return adv_img, count, dom_cat_prop, dom_cat, c_ts, c_2best


# SET UP your attack

# Import the target CNN and required libraries
from keras.applications.vgg16 import (
    decode_predictions,
    preprocess_input,
    VGG16,
)
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Step 1: Load a clean image and convert it to numpy array:
x = load_img("acorn1.JPEG")#, target_size=(224, 224), interpolation="lanczos")
# x = img_to_array(image)
x_array = img_to_array(x)
y = 306  # Optional! Target category index number. It is only for the targeted attack.
epsilon = 8

kclassifier = VGG16(weights="imagenet")

#TESTING::
ancestor = "acorn"
j = 1
target = 'rhinoceros_beetle'


# Step 3: Built the attack and generate adversarial image:
roi = np.load("unique_pixels.npy")

print('##############################################################')
print(f"#########  {ancestor}_{j}  #########\n")
print("The size of the image is: ", (x_array.shape[0], x_array.shape[1]))
print("Number of pixels to modify: ",roi.shape[0])
ratio = 100 * roi.shape[0] / (x_array.shape[0]*x_array.shape[1])
print(f"Size ZoI/Image: {roi.shape[0]}/{x_array.shape[0]*x_array.shape[1]}")
print("The percentage of the image will be modified is: %.1f%%" % (ratio))
print('##############################################################')
modelX = "VGG16"
attackEA = EA(
    kclassifier, max_iter=20000, confidence=0.30, targeted=True
)  # if targeted is True, then confidence will be taken into account.
advers, count, dom_cat_prop, dom_cat, c_ts, c_2best = attackEA._generate(x, y)  # returns adversarial image as .npy file. y is optional.
np.save("advers.npy", advers)
img = Image.fromarray(advers.astype(np.uint8))
dom_cat_prop =  "{:.3f}".format(dom_cat_prop)
img.save(f"advers_ct_{count}_{dom_cat_prop}_{dom_cat}_eps{epsilon}.png", "png")
f = open(f'report_{modelX}.txt', 'a')
f.write(f"acorn1,{count},{dom_cat},{dom_cat_prop},eps:{epsilon}\n")
f.close()

c_ts = np.array(c_ts)
c_2best = np.array(c_2best)
np.save(f'{ancestor}{j}_c_ts.npy', c_ts)
np.save(f'{ancestor}{j}_c_2best.npy',c_2best)




#TODO:
# 1. work on mutation
# 2. show the target label value in console real time
# 3. test increased region sizes

