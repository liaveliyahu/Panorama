"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""
        n_points = match_p_src.shape[1]
        A = np.zeros((2*n_points+1, 9))

        for i in range(n_points):
            x_s, y_s = match_p_src[:,i]
            x_d, y_d = match_p_dst[:,i]
            A[2*i] = np.array([x_s, y_s, 1, 0, 0, 0, -x_d*x_s, -x_d*y_s, -x_d])
            A[2*i+1] = np.array([0, 0, 0, x_s, y_s, 1, -y_d*x_s, -y_d*y_s, -y_d])

        A[-1,-1] = 1

        U, S, V = svd(A)
        homography = V[-1].reshape(3, 3)

        return homography

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        transformed_image = np.zeros(dst_image_shape, dtype=np.uint8)
        H, W, _ = src_image.shape
        for y in range(H):
            for x in range(W):
                transformed_vector = homography @ np.array([x, y, 1])
                u, v, _ = (transformed_vector/transformed_vector[-1]).astype(np.uint16) # check this
                if v < H and u < W:
                    transformed_image[v, u, :] = src_image[y, x, :]
        
        return transformed_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        H, W, _ = src_image.shape
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)

        xf = xx.flatten()
        yf = yy.flatten()

        xyf1 = np.ones((3, xf.shape[0]), dtype=np.uint16)
        xyf1[0,:] = xf
        xyf1[1,:] = yf

        xyf1_transformed = homography @  xyf1
        uf, vf, _ = np.rint(xyf1_transformed / xyf1_transformed[2]).astype(np.int16)

        uu = uf.reshape(xx.shape)
        vv = vf.reshape(yy.shape)

        valid = (vv >= 0) & (vv < H) & (uu >= 0) & (uu < W)
        vv_valid, uu_valid = vv[valid], uu[valid]
        yy_valid, xx_valid = yy[valid], xx[valid]

        transformed_image = np.zeros(dst_image_shape, dtype=np.uint8)
        transformed_image[vv_valid, uu_valid] = src_image[yy_valid, xx_valid]

        return transformed_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        fit_percent = 0.0
        mse = 0
        
        n_points = match_p_src.shape[1]
        fit_points = 0

        for i in range(n_points):
            x_s, y_s = match_p_src[:,i]
            x_d, y_d = match_p_dst[:,i]
            transformed_vector = homography @ np.array([x_s, y_s, 1])
            u, v, _ = (transformed_vector/transformed_vector[-1]).astype(np.uint16)

            if np.linalg.norm(np.array([x_d-u, y_d-v])) <= max_err:
                fit_points += 1
                diff = np.array([x_d - u, y_d -v])
                mse += diff.T @ diff
        
        fit_percent = fit_points / n_points
        dist_mse = mse / fit_points if fit_points != 0 else 10**9
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        N =  match_p_src.shape[1]

        src_meet_points = np.zeros((2, N))
        dst_meet_points = np.zeros((2, N))

        d = 0
        for i in range(N):
            x_s, y_s = match_p_src[:,i]
            x_d, y_d = match_p_dst[:,i]
            transformed_vector = homography @ np.array([x_s, y_s, 1])
            u, v, _ = (transformed_vector/transformed_vector[-1]).astype(np.uint16)
            
            if np.linalg.norm(np.array([x_d-u, y_d-v])) <= max_err:
                src_meet_points[:,d] = x_s, y_s
                dst_meet_points[:,d] = x_d, y_d
                d += 1
    
        meet_points = (src_meet_points[:,:d], dst_meet_points[:,:d])
        return meet_points

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        # w = inliers_percent
        # # t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        # p = 0.99
        # # the minimal probability of points which meets with the model
        # d = 0.5
        # # number of points sufficient to compute the model
        # n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        # k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        w = inliers_percent
        t = max_err
        p = 0.99
        d = 0.5
        n = 4
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        N = match_p_src.shape[1]

        max_inliers = 0
        if inliers_percent < d:
            print(r"Computing Homography with RANSAC Algorithm requires at least 50% inliers among the matching points.")
            print("Computing homography with naive approach instead: ")
            h = self.compute_homography_naive(match_p_src, match_p_dst)
            return
        for _ in range(k):
            random_indices = np.random.choice(np.arange(N), size=n, replace=False)
            h = self.compute_homography_naive(match_p_src[:,random_indices], match_p_dst[:,random_indices])
            meet_points = self.meet_the_model_points(h, match_p_src, match_p_dst, t)
            M = meet_points[0].shape[1]
            if M >= max_inliers:
                max_inliers = M
                best_h = h

        return best_h

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        H, W, _ = dst_image_shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)

        uf = uu.flatten()
        vf = vv.flatten()

        uvf1 = np.vstack((uf, vf, np.ones_like(uf)))

        uvf1_backward_transformed = backward_projective_homography @  uvf1

        xf, yf = uvf1_backward_transformed[:2] / uvf1_backward_transformed[2]

        xx = xf.reshape(uu.shape)
        yy = yf.reshape(vv.shape)

        backward_warped_src_img = np.zeros(dst_image_shape, dtype=np.uint8)
        xx_s, yy_s = np.meshgrid(np.arange(src_image.shape[1]),np.arange(src_image.shape[0]))

        for c in range(3):
            pixel_vals = griddata((yy_s.flatten(), xx_s.flatten()), src_image[:, :, c].flatten(), (yy, xx), method="cubic", fill_value=0)
            print(f'finished interpolation of channel {c}')
            backward_warped_src_img[:, :, c] = pixel_vals.reshape(H,W)
        
        return backward_warped_src_img
        

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        translation = np.eye(3)
        translation[0, 2] = -pad_left
        translation[1, 2] = -pad_up
        translated_homography = backward_homography @ translation
        translated_homography /= np.linalg.norm(translated_homography)
        return translated_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)

        pan_rows, pan_cols, pad_struct = self.find_panorama_shape(src_image, dst_image, homography)
        
        backward_homography = np.linalg.inv(homography)

        backward_homography_with_translation = self.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left, pad_struct.pad_up)

        img_panorama = self.compute_backward_mapping(backward_homography_with_translation, src_image,(pan_rows, pan_cols, 3))

        img_panorama[pad_struct.pad_up:(dst_image.shape[0] + pad_struct.pad_up),
        pad_struct.pad_left:(dst_image.shape[1] + pad_struct.pad_left), :] = dst_image[:, :, :]

        return np.clip(img_panorama, 0, 255).astype(np.uint8)
    
    @staticmethod
    def show_matching_points(src_img: np.ndarray,
                             dst_img: np.ndarray,
                             match_p_src: np.ndarray,
                             match_p_dst: np.ndarray,
                             wrong_points:list = [20,21,22,23,24]) -> None:
        """
        wrong_points of the ex images = [3, 6, 12, 21, 23]
        wrong_points of the test images = [20, 21, 22, 23, 24]

        Args:
            src_img: Source image expected to undergo projective
            transformation.
            dst_img: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            wrong_points: list of the indices for the wrong matching points.

        Returns:
            returns None, but shows the images with the matching point.s 
        """
        from matplotlib.patches import ConnectionPatch
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(src_img)
        ax2.imshow(dst_img)

        ax1.axis('off')
        ax2.axis('off')

        for i in range(match_p_src.shape[1]):
            c = 'red' if i in wrong_points else 'lime'
 
            xyS = tuple(match_p_src[:,i].astype(int))
            xyD = tuple(match_p_dst[:,i].astype(int))
            con = ConnectionPatch(xyA=xyD, xyB=xyS, coordsA="data", coordsB="data",
                                axesA=ax2, axesB=ax1, color=c)
            ax2.add_artist(con)

        ax1.scatter(match_p_src[0],match_p_src[1] ,color='blue')
        ax2.scatter(match_p_dst[0],match_p_dst[1] ,color='orange')

        ax1.set_title("Source Image")
        ax2.set_title("Destination Image") 

        plt.show()