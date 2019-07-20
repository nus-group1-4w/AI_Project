import cv2
import numpy as np
import math

class HoG():
    def __init__(self, image, cell_size=8, bin_size=8):
        self.image = image
        self.image = np.sqrt(image / float(np.max(image)))
        self.image *= 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    def feature_extract(self):
        height, width = self.image.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height//self.cell_size, width//self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i*self.cell_size:(i+1)*self.cell_size,j*self.cell_size:(j+1)*self.cell_size]
                cell_angle = gradient_angle[i*self.cell_size:(i+1)*self.cell_size,j*self.cell_size:(j+1)*self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_vector = []
        for i in range(cell_gradient_vector.shape[0]-1):
            for j in range(cell_gradient_vector.shape[1]-1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j+1])
                block_vector.extend(cell_gradient_vector[i+1][j])
                block_vector.extend(cell_gradient_vector[i+1][j+1])

                summation = 0
                for k in block_vector:
                    summation += k**2
                magnitude = math.sqrt(summation)

                if magnitude != 0:
                    for gradient in block_vector:
                        gradient /= magnitude
                hog_vector.append(block_vector)

        return hog_vector

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation[min_angle] += (gradient_strength*(1-(mod/self.angle_unit)))
                orientation[max_angle] += (gradient_strength*(mod/self.angle_unit))

        return orientation

    def get_closest_bins(self, gradient_angle):
        ind = int(gradient_angle/self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if ind == self.bin_size:
            return ind-1, ind % self.bin_size, mod
        return ind, (ind+1) % self.bin_size, mod
