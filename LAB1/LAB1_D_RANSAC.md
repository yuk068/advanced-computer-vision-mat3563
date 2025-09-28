# Lab1 D: RANSAC Algorithm and DoG

Note: Because the task is not much different from the reference notebooks, for LAB1 D, I will not be providing a notebook, only this report. Reference notebook(s) are in `reference_notebooks/`.

## Introduction

Image stitching is a computational photography technique that combines multiple photographic images with overlapping fields of view to produce a segmented panorama or high-resolution image. This process is fundamental in various applications, including creating panoramic views, generating high-resolution maps, and reconstructing 3D scenes. The core challenge in image stitching lies in accurately aligning the input images and seamlessly blending them to avoid visible seams or artifacts.

This report delves into the intricate details of image stitching and alignment, drawing insights from two provided Python reference files: `multistitch_customH_multiband.ipynb` and `dog_harris_alignment_customH_fixed.ipynb`. These notebooks demonstrate advanced techniques, including custom robust transform estimation and multi-band blending, which are crucial for achieving high-quality panoramic outputs. The report will cover the theoretical underpinnings, mathematical formulations, and practical implementations of key components such as keypoint detection, feature description and matching, geometric transformations (Homography, Affine, Euclidean), robust estimation using RANSAC, image warping, and multi-band blending with Laplacian pyramids. Furthermore, it will compare the approaches used in the provided files and discuss the overall pipeline and common challenges encountered in image stitching.

The journey of image stitching typically involves several critical steps: identifying distinctive features in overlapping regions of images, matching these features across different images, estimating the geometric transformation that aligns them, warping the images onto a common canvas, and finally, blending the warped images to create a seamless composite. Each of these steps involves sophisticated algorithms and mathematical models, which will be explored in detail throughout this document. The goal is to provide a thorough understanding of the entire process, from low-level pixel operations to high-level image composition, highlighting the elegance and complexity involved in creating visually compelling panoramas.

## Keypoint Detection: DoG and Harris Corner Detector

Keypoint detection is the initial and arguably one of the most critical steps in any image stitching pipeline. Its primary goal is to identify stable, distinctive, and repeatable points across multiple images, regardless of variations in viewpoint, illumination, or scale. The provided Python notebooks utilize a combination of the Difference of Gaussians (DoG) for scale-space extrema detection and the Harris Corner Detector for refining and filtering these keypoints, ensuring robustness and accuracy.

### Difference of Gaussians (DoG)

The Difference of Gaussians (DoG) is a feature enhancement algorithm that approximates the Laplacian of Gaussian (LoG) operator, which is known for its ability to detect blobs (regions of interest that are brighter or darker than their surroundings) at various scales. The LoG operator is computationally expensive due to its second-order derivative nature. DoG offers a computationally efficient alternative by subtracting two Gaussian-blurred versions of an image, each with a slightly different standard deviation.

Mathematically, a 2D Gaussian function $G(x, y, \sigma)$ is defined as:

$$ G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} $$

Where $\sigma$ is the standard deviation, controlling the extent of blurring. The Difference of Gaussians $D(x, y, \sigma)$ is then given by:

$$ D(x, y, \sigma) = G(x, y, k\sigma) - G(x, y, \sigma) $$

Here, $k$ is a constant factor that determines the scale difference between the two Gaussian kernels. In the context of scale-invariant feature detection, such as in the SIFT algorithm, a scale-space is constructed by repeatedly blurring and downsampling the image. The DoG is then computed across different scales (octaves and intervals within octaves) to identify potential keypoints that are invariant to scale changes. These keypoints are typically found at the local extrema (maxima or minima) of the DoG function in both space and scale.

The `build_gaussian_pyramid` function in the provided code constructs a Gaussian pyramid, which is a sequence of images, each being a smoothed and downsampled version of the previous one. The `build_dog_pyramid` then computes the DoG images by subtracting adjacent levels of the Gaussian pyramid. The `is_local_extrema` function checks if a pixel is a local extremum in a 3x3x3 neighborhood across scale and space, indicating a potential keypoint.

### Harris Corner Detector

While DoG is effective at detecting blob-like structures, it can also identify edges, which are generally less stable for matching purposes. The Harris Corner Detector is employed to refine the set of potential keypoints by identifying true corners, which are characterized by significant intensity variations in all directions. A corner is a point where there is a large change in image intensity in more than one direction.

The Harris Corner Detector works by considering a small window around each pixel and calculating the change in intensity for shifts of this window in various directions. This is encapsulated in the **Structure Tensor** or **Second Moment Matrix** $M$, defined as:

$$ M = \sum_{u,v} w(u,v) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix} $$

Where $I_x$ and $I_y$ are the image gradients in the x and y directions, respectively, and $w(u,v)$ is a Gaussian weighting function applied to the window. The sums $S_{xx}, S_{yy}, S_{xy}$ are computed by applying a Gaussian blur to $I_x^2$, $I_y^2$, and $I_x I_y$ respectively. The gradients are typically computed using Sobel operators.

The corner response function $R$ is then calculated from the eigenvalues of $M$. A common approximation for $R$ (used in the provided code) is:

$$ R = \text{det}(M) - k (\text{trace}(M))^2 $$

Where $\text{det}(M) = S_{xx}S_{yy} - S_{xy}^2$, $\text{trace}(M) = S_{xx} + S_{yy}$, and $k$ is an empirical constant (typically 0.04 to 0.06). A large positive value of $R$ indicates a corner, a large negative value indicates an edge, and a small absolute value indicates a flat region.

The `harris_response` function in the code calculates this response. The `pass_edge_response` function further filters out edge-like features by checking the ratio of eigenvalues of the structure tensor, ensuring that only true corners are retained. This is based on the idea that for an edge, one eigenvalue will be much larger than the other, leading to a high ratio, while for a corner, both eigenvalues will be large and similar, resulting in a low ratio.

### Non-Maximum Suppression (NMS)

After detecting potential keypoints using DoG and filtering them with the Harris response, Non-Maximum Suppression (NMS) is applied. NMS ensures that only the most prominent keypoint within a local neighborhood is selected, preventing multiple detections for the same feature. The `nonmax_suppression` function sorts keypoints by their Harris score and iteratively selects the highest-scoring keypoint, suppressing any other keypoints within a defined radius. This process yields a sparse set of robust and well-localized keypoints.

By combining DoG for multi-scale blob detection and Harris for corner refinement, the system effectively identifies a set of stable keypoints that are crucial for accurate image alignment and stitching. The `detect_keypoints_dog_harris` function orchestrates this entire process, producing a list of keypoints with their coordinates, scale, and score.



## Feature Description and Matching: ORB and Brute-Force Matcher

Once keypoints are detected, the next crucial step in image stitching is to describe these keypoints in a way that allows them to be uniquely identified and matched across different images. This involves generating feature descriptors, which are compact representations of the local image region around each keypoint. Subsequently, a matching algorithm is used to find correspondences between descriptors from different images.

### ORB (Oriented FAST and Rotated BRIEF) Descriptors

ORB (Oriented FAST and Rotated BRIEF) is a highly efficient and robust feature detector and descriptor algorithm, designed as a free alternative to SIFT and SURF. It combines the best aspects of the FAST (Features from Accelerated Segment Test) keypoint detector and the BRIEF (Binary Robust Independent Elementary Features) descriptor, with significant enhancements to improve performance and robustness, particularly regarding rotation invariance.

While the provided code uses DoG+Harris for keypoint detection, it employs ORB for descriptor computation. The process involves:

1.  **Keypoint Orientation**: For each detected keypoint, ORB computes its orientation. This is crucial for achieving rotation invariance. It typically uses the intensity centroid method, where the moments of the image patch around the keypoint are calculated to determine a dominant orientation. This orientation is then used to rotate the patch before computing the descriptor, ensuring that the descriptor is consistent regardless of the image's rotation.

2.  **BRIEF Descriptor**: The core of the ORB descriptor is based on BRIEF. BRIEF constructs a binary string descriptor by performing a series of simple intensity comparisons in a predefined pattern around the keypoint. For example, it might compare the intensity of pixel A with pixel B, and if A is brighter, the bit is 1; otherwise, it's 0. This process is repeated for many pairs, resulting in a binary feature vector. The advantage of binary descriptors like BRIEF is their computational efficiency and low memory footprint.

ORB enhances BRIEF by applying the rotation information obtained in the first step. The sampling pattern used for BRIEF comparisons is rotated according to the keypoint's orientation, making the resulting descriptor rotation-invariant. This modification is what makes BRIEF 'Rotated'. The `compute_orb_descriptors` function in the code takes the grayscale image and the detected keypoints, then uses OpenCV's `ORB_create` to compute these descriptors.

### Brute-Force (BF) Matching with Ratio Test

After computing descriptors for keypoints in two images, the next step is to find which descriptors correspond to each other. The Brute-Force (BF) Matcher is a straightforward approach that compares the descriptor of every keypoint in the first image with the descriptor of every keypoint in the second image.

For binary descriptors like ORB, the distance between two descriptors is typically measured using the Hamming distance, which counts the number of positions at which the corresponding bits are different. The BFMatcher finds the best match (the descriptor with the smallest Hamming distance) for each descriptor from the first set in the second set.

To improve the robustness of matching and filter out ambiguous or incorrect matches, a **ratio test** (also known as Lowe's ratio test) is commonly applied. This test was originally proposed for SIFT descriptors but is widely used with other descriptors as well. The principle is as follows:

For each keypoint descriptor in the first image, the BFMatcher finds the two closest matches in the second image. Let $d_1$ be the distance to the closest match and $d_2$ be the distance to the second closest match. If the ratio $d_1 / d_2$ is below a certain threshold (e.g., 0.75, as used in the provided code), then the match is considered good. If the ratio is close to 1, it means that the first and second best matches are very similar, indicating that the match might be ambiguous or unreliable. By discarding such ambiguous matches, the ratio test significantly reduces the number of false positives, leading to a more accurate set of correspondences.

The `match_descriptors` function in the provided code implements this process. It uses `cv2.BFMatcher` with `cv2.NORM_HAMMING` for ORB descriptors and applies the ratio test to filter the matches. The output is a list of `DMatch` objects, representing the reliable correspondences between keypoints in the two images. These filtered matches are then used for estimating the geometric transformation between the images.



## Geometric Transformations: Homography, Affine, and Euclidean

After establishing a set of reliable feature correspondences between images, the next step is to determine the geometric transformation that maps points from one image to another. This transformation is crucial for aligning the images onto a common plane. The provided Python notebooks implement custom estimation methods for three fundamental types of 2D geometric transformations: Homography, Affine, and Euclidean (Rotation + Translation).

### Homography Transformation (Projective Transformation)

A homography is a $3 \times 3$ matrix that maps points from one plane to another plane. It is the most general 2D planar projective transformation and has 8 degrees of freedom. Homographies are particularly useful in image stitching when the scene being photographed is planar or when the camera undergoes pure rotation around its optical center.

Given a point $(x, y)$ in the first image and its corresponding point $(u, v)$ in the second image, related by a homography $H$, the relationship in homogeneous coordinates is:

$$ \begin{bmatrix} u \cdot w \\ v \cdot w \\ w \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} $$

Where $w$ is a scaling factor. Expanding this, we get:

$$ u = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}} \\ v = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}} $$

To solve for the 8 unknown parameters of $H$ (since $h_{33}$ can be normalized to 1), at least 4 non-collinear point correspondences are required. The provided code uses the **Direct Linear Transform (DLT)** algorithm, specifically a normalized version, to estimate the homography.

#### Normalized DLT Algorithm

The DLT algorithm solves for $H$ by formulating a system of linear equations from the point correspondences. Each correspondence $(x, y) \leftrightarrow (u, v)$ provides two equations. The normalization step, as described by Hartley, is crucial for numerical stability. It involves transforming the input points such that their centroid is at the origin and their average distance from the origin is $\sqrt{2}$. This process is handled by the `normalize_points` function, which computes a normalization matrix $T$ for each set of points.

The normalized DLT process is as follows:
1.  **Normalize Points**: Apply a similarity transformation (translation and scaling) to both sets of points, $pts1$ and $pts2$, using matrices $T_1$ and $T_2$ respectively, to obtain normalized points $p1_n$ and $p2_n$.
2.  **Formulate Linear System**: For each normalized correspondence $(x_n, y_n) \leftrightarrow (u_n, v_n)$, construct two rows of a matrix $A$:

$$
\begin{bmatrix} -x_n & -y_n & -1 & 0 & 0 & 0 & u_n x_n & u_n y_n & u_n \\
0 & 0 & 0 & -x_n & -y_n & -1 & v_n x_n & v_n y_n & v_n
\end{bmatrix}
$$

This leads to a system $Ah = 0$, where $h$ is a vector containing the elements of the homography matrix $H_n$ (row-major order).

3.  **Solve using SVD**: The homogeneous system $Ah = 0$ is solved using Singular Value Decomposition (SVD). The solution $h$ is the last column of $V$ (or last row of $V^T$) corresponding to the smallest singular value of $A$. This $h$ is then reshaped into the $3 \times 3$ normalized homography matrix $H_n$.
4.  **Denormalize Homography**: The final homography $H$ is recovered by denormalizing $H_n$ using the inverse of the normalization matrices:

$$
H = T_2^{-1} H_n T_1
$$

This entire process is implemented in the `dlt_homography` function, which ensures a robust and accurate estimation of the homography matrix.

### Affine Transformation

An affine transformation is a special case of a homography where the last row of the transformation matrix is $[0, 0, 1]$. It preserves parallelism of lines but does not necessarily preserve angles or lengths. Affine transformations have 6 degrees of freedom and can represent translation, rotation, scaling, and shearing. They are suitable for aligning images when the camera motion is restricted to translation, rotation, and scaling, and there is no perspective distortion.

The affine transformation matrix $M_{affine}$ is:

$$ M_{affine} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix} $$

Given a point $(x, y)$ in the first image and its corresponding point $(u, v)$ in the second image, the relationship is:

$$ u = a_{11}x + a_{12}y + t_x \\ v = a_{21}x + a_{22}y + t_y $$

To solve for the 6 unknown parameters, at least 3 non-collinear point correspondences are required. The `estimate_affine` function in the code solves this using a linear least squares approach. It constructs a system of equations $A x = b$, where $x$ contains the 6 affine parameters, and solves it using `np.linalg.lstsq`.

### Euclidean Transformation (Rotation + Translation)

A Euclidean transformation, also known as a rigid transformation, is a special case of an affine transformation that preserves distances and angles. It consists only of rotation and translation, having 3 degrees of freedom (one for rotation angle, two for translation components). This transformation is appropriate when the images are related by a simple rigid motion without any scaling or shearing.

The Euclidean transformation matrix $M_{euclidean}$ is:

$$ M_{euclidean} = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix} $$

To estimate a Euclidean transformation, at least 2 point correspondences are needed. The provided code uses the **Kabsch algorithm** (also known as Wahba's problem or Procrustes analysis) to find the optimal rotation and translation that minimizes the Root Mean Square Deviation (RMSD) between two sets of paired points.

#### Kabsch Algorithm Steps:
1.  **Centroid Calculation**: Compute the centroids of both sets of points, $P_1$ and $P_2$.
2.  **Centroid Subtraction**: Translate both sets of points so their centroids are at the origin: $X = P_1 - \mu_1$ and $Y = P_2 - \mu_2$.
3.  **Covariance Matrix**: Compute the covariance matrix $H = X^T Y$.
4.  **SVD of Covariance**: Perform SVD on $H$: $H = U \Sigma V^T$.
5.  **Rotation Matrix**: The rotation matrix $R$ is given by $R = V U^T$. A reflection correction is applied if the determinant of $R$ is negative.
6.  **Translation Vector**: The translation vector $t$ is then calculated as $t = \mu_2 - R \mu_1$.

These steps are implemented in the `estimate_euclidean` function, providing a robust way to estimate rigid transformations. The choice of transformation model (Homography, Affine, or Euclidean) depends on the geometric relationship between the images and the desired level of accuracy and complexity. Homography is generally the most versatile for general image stitching scenarios.



## Robust Estimation: RANSAC (Random Sample Consensus)

In real-world scenarios, feature matching often produces a significant number of outliers—incorrect correspondences that do not adhere to the true geometric transformation between images. Directly using these noisy matches to estimate a transformation matrix would lead to inaccurate results. To overcome this, a robust estimation technique is required, and **Random Sample Consensus (RANSAC)** is a widely adopted algorithm for this purpose.

RANSAC is an iterative method used to estimate the parameters of a mathematical model from a set of observed data that contains a substantial number of outliers. Its core idea is to randomly select a minimal subset of data points (the *sample*) from the input, hypothesize a model based on this subset, and then evaluate how many of the remaining data points (the *consensus set*) are consistent with this model. This process is repeated multiple times, and the model that has the largest consensus set is chosen as the best estimate.

### RANSAC Algorithm Steps

The `ransac_transform` function in `multistitch_customH_multiband.ipynb` and `ransac_homography` in `dog_harris_alignment_customH_fixed.ipynb` implement the RANSAC algorithm. The general steps are as follows:

1.  **Parameter Definition**: Define the minimum number of points ($s$) required to estimate the model parameters, a distance threshold (`thresh`) to determine if a point is an inlier, the maximum number of iterations (`max_iters`), and a confidence level (`confidence`) for finding a good model.

2.  **Iterative Sampling and Model Estimation**:
    *   **Random Sample Selection**: In each iteration, $s$ data points (feature correspondences) are randomly selected from the input set. These points are assumed to be inliers.
    *   **Model Estimation**: A transformation model (Homography, Affine, or Euclidean, depending on the context) is estimated using only these $s$ selected points. The functions `dlt_homography`, `estimate_affine`, or `estimate_euclidean` are called for this step.

3.  **Inlier Identification and Scoring**:
    *   **Projection and Error Calculation**: All other data points (not in the initial sample) are transformed using the estimated model. The distance (error) between the transformed points and their actual corresponding points is calculated. The provided code uses `symmetric_transfer_errors`, which calculates the sum of forward and backward projection errors to ensure robustness.
    *   **Inlier/Outlier Classification**: Points whose error is below the predefined `thresh` are classified as inliers, forming the consensus set for the current model. Points with errors above the threshold are considered outliers.
    *   **Score Update**: The size of the consensus set (number of inliers) is counted. If this count is greater than the best count found so far, the current model is considered the `best_M` (or `best_H`), and its inliers are stored as `best_inliers`.

4.  **Adaptive Iteration Count**: A crucial optimization in RANSAC is the adaptive adjustment of `max_trials`. As more inliers are found, the probability of picking an all-inlier sample increases. The number of iterations needed to achieve a desired confidence level can be estimated using the formula:

    $$ N = \frac{\log(1 - p)}{\log(1 - w^s)} $$

    Where:
    *   $N$ is the number of iterations.
    *   $p$ is the desired probability of success (confidence).
    *   $w$ is the probability that any selected data point is an inlier (estimated as `c / N_total`, where `c` is the current best inlier count and `N_total` is the total number of correspondences).
    *   $s$ is the minimum number of points required to estimate the model.

    This adaptive update ensures that RANSAC runs only as many iterations as necessary, improving efficiency, especially when a good model is found early.

5.  **Final Model Refinement**: After `max_trials` iterations, the model with the largest consensus set is selected. Optionally, a final refinement step can be performed by re-estimating the model parameters using all identified inliers (not just the minimal sample) to achieve a more accurate fit. The provided code performs this refinement by calling the estimation function (`est`) again with `pts1[inliers]` and `pts2[inliers]`.

### Symmetric Transfer Errors

The `symmetric_transfer_errors` function is vital for accurately evaluating the quality of a transformation. Instead of just projecting points from image 1 to image 2 and measuring the error, it also projects points from image 2 back to image 1 using the inverse transformation. The total error for a correspondence is the sum of these two projection errors. This approach makes the error metric more robust to noise and ensures that the transformation is consistent in both directions. If the inverse of the transformation matrix cannot be computed (e.g., for degenerate cases), a large error is assigned to prevent such models from being selected.

By integrating RANSAC, the image stitching pipeline can effectively handle noisy feature matches, leading to highly accurate and robust estimation of the geometric transformation, which is essential for seamless image alignment.



## Image Warping and Canvas Generation

Once the geometric transformation (e.g., homography) between images has been robustly estimated, the next step is to apply this transformation to align the images onto a common canvas. This process, known as image warping, involves geometrically transforming an image so that its features align with a reference image or a predefined panoramic plane. Simultaneously, a canvas large enough to accommodate all warped images must be computed.

### Image Warping

Image warping is the process of digitally manipulating an image such that its content is geometrically transformed. In the context of image stitching, this typically means applying the estimated homography (or affine/Euclidean transformation) to one or more source images to project them into the coordinate system of a reference image or a larger panoramic canvas.

The `cv2.warpPerspective` function in OpenCV is commonly used for this purpose. It takes an input image, a $3 \times 3$ transformation matrix (like a homography), and the dimensions of the output canvas. For each pixel in the output canvas, `warpPerspective` calculates its corresponding location in the source image using the inverse of the transformation matrix. This inverse mapping is crucial to avoid holes or gaps in the warped image, as it ensures that every pixel in the destination image gets its intensity value from the source image. Interpolation techniques (e.g., bilinear or bicubic) are then used to determine the pixel values for non-integer coordinates.

### Canvas Computation

Before warping, it's essential to determine the size and offset of the final panoramic canvas. This canvas must be large enough to contain all the input images after they have been transformed and aligned. The process involves calculating the bounding box of all warped image corners.

The `compute_canvas_and_warp` function in `multistitch_customH_multiband.ipynb` (and implicitly handled in `warp_and_stitch` in `dog_harris_alignment_customH_fixed.ipynb`) performs the following steps:

1.  **Project Image Corners**: For each image, its four corner coordinates (e.g., `[0,0], [width,0], [width,height], [0,height]`) are transformed using its respective estimated transformation matrix. If an image is designated as the reference (e.g., `images[0]` in `multistitch_customH_multiband.ipynb`), its transformation matrix is an identity matrix.
    $$ \text{warped\_corner} = M \cdot \text{original\_corner} $$
    Where $M$ is the transformation matrix (Homography, Affine, or Euclidean).

2.  **Determine Bounding Box**: All the transformed corner coordinates from all images are collected. The minimum and maximum x and y coordinates among all these warped corners define the extent of the final panorama. This gives `xmin`, `ymin`, `xmax`, and `ymax`.

3.  **Calculate Translation Offset**: Since image coordinates typically start from `(0,0)` in the top-left corner, the panoramic canvas needs to be shifted so that its top-left corner is at `(0,0)`. This translation is calculated as `tx = -xmin` and `ty = -ymin`. A translation matrix $T$ is then constructed:
    $$ T = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} $$

4.  **Compute Panorama Dimensions**: The width of the panorama is `pano_w = xmax - xmin` and the height is `pano_h = ymax - ymin`.

5.  **Combine Transformations**: For each image, its original transformation matrix $M$ is combined with the global translation matrix $T$ to form a composite transformation $W = T \cdot M$. This composite matrix maps points from the original image directly to their correct positions on the new panoramic canvas.

6.  **Warp Images**: Each input image is then warped using `cv2.warpPerspective` with its corresponding composite transformation matrix $W$ and the calculated panorama dimensions `(pano_w, pano_h)`. This results in all images being correctly positioned and transformed onto the large, empty canvas.

7.  **Generate Masks**: Alongside the warped images, corresponding masks are often generated. These masks indicate the valid pixel regions of each warped image (i.e., where the original image content exists, and not the black background introduced by warping). These masks are crucial for the subsequent blending step to ensure smooth transitions.

By meticulously calculating the canvas dimensions and applying the appropriate transformations, image warping effectively brings all input images into a common coordinate system, preparing them for the final blending stage. This ensures that the stitched panorama maintains geometric consistency and accurate alignment across all its constituent parts.



## Multi-Band Blending: Laplacian Pyramids

After all images have been warped and aligned onto a common canvas, the final step in creating a seamless panorama is blending them. Simple averaging or feathering techniques can often lead to visible seams, ghosting, or blurring, especially when there are slight misalignments or differences in illumination between the source images. **Multi-band blending**, particularly using **Laplacian pyramids**, is a sophisticated technique designed to achieve seamless transitions by blending images at different frequency levels.

### The Concept of Image Pyramids

Image pyramids are hierarchical data structures that represent an image at multiple resolutions. There are two main types:

1.  **Gaussian Pyramid**: A Gaussian pyramid is created by repeatedly applying a Gaussian blur filter to an image and then downsampling it (e.g., by a factor of 2). Each level of the Gaussian pyramid is a smaller, blurrier version of the previous level. The `build_gaussian_pyr` function in the code constructs this pyramid.

2.  **Laplacian Pyramid**: A Laplacian pyramid is derived from a Gaussian pyramid and represents the difference between an image level in the Gaussian pyramid and an expanded (upsampled) version of its next higher (smaller) level. Essentially, it captures the detail or high-frequency information lost during the downsampling process of the Gaussian pyramid. The top level of the Laplacian pyramid is typically the smallest image from the Gaussian pyramid. The `build_laplacian_pyr` function generates this pyramid.

Mathematically, for a Gaussian pyramid $G_0, G_1, \dots, G_N$ (where $G_0$ is the original image), a Laplacian pyramid $L_0, L_1, \dots, L_N$ is constructed as:

$$ L_i = G_i - \text{Expand}(G_{i+1}) \quad \text{for } 0 \le i < N $$
$$ L_N = G_N $$

Where $\text{Expand}(G_{i+1})$ refers to upsampling $G_{i+1}$ and blurring it to match the size and characteristics of $G_i$. The original image can be perfectly reconstructed from its Laplacian pyramid by reversing this process, as shown in the `reconstruct_from_laplacian` function:

$$ G_i = L_i + \text{Expand}(G_{i+1}) $$

### Multi-Band Blending Process

The `multiband_blend` function implements the blending process using Laplacian pyramids. The core idea is to blend the low-frequency components (coarse details) of the images over a large region and the high-frequency components (fine details) over a smaller region. This is achieved by performing the blending operation at each level of the Laplacian pyramid.

Given two images, A and B, and a blending mask (often derived from the overlap region of the warped images), the steps are:

1.  **Generate Laplacian Pyramids**: Create Laplacian pyramids for both input images, $L_A$ and $L_B$. The images are first converted to float32 and normalized to the range [0, 1] for accurate computation.

2.  **Generate Gaussian Pyramid for Mask**: Create a Gaussian pyramid for the blending mask, $G_M$. The mask typically defines the region where image B should blend into image A. A smooth mask (e.g., generated by Gaussian blurring the binary mask) is crucial to avoid sharp transitions. The `maskB` in the code is first blurred to create a smooth transition.

3.  **Blend at Each Pyramid Level**: For each level $l$ of the pyramids, blend the corresponding Laplacian levels of images A and B using the corresponding Gaussian level of the mask:
    $$ L_{	ext{Blended},l} = L_{A,l} \cdot (1 - G_{M,l}) + L_{B,l} \cdot G_{M,l} $$
    This step effectively blends the frequency components. At lower levels (fine details), the mask is sharper, allowing for more precise blending. At higher levels (coarse details), the mask is blurrier, enabling smoother transitions over larger areas.

4.  **Reconstruct Blended Image**: Once all levels of the blended Laplacian pyramid ($L_{	ext{Blended}}$) are computed, the final blended image is reconstructed by collapsing the blended Laplacian pyramid using the `reconstruct_from_laplacian` function. This involves iteratively expanding each level and adding it to the next higher level, starting from the smallest image at the top of the pyramid.

### Advantages of Multi-Band Blending

*   **Seamless Transitions**: By blending different frequency components separately, multi-band blending can effectively hide seams and minimize artifacts that arise from misalignments, parallax, or photometric inconsistencies.
*   **Preservation of Details**: High-frequency details are blended locally, preserving sharpness, while low-frequency components are blended globally, ensuring smooth overall transitions.
*   **Robustness**: It is more robust to slight errors in alignment and illumination differences compared to simpler blending methods.

The `multiband_blend` function handles multi-channel images (like BGR) by processing each channel independently and then stacking them back together. The output is a seamlessly stitched image, ready for display or further processing. This technique is a cornerstone of high-quality image stitching, enabling the creation of visually compelling panoramas that appear as if they were captured as a single, wide-angle shot.



## Overall Image Stitching Pipeline and Common Challenges

Image stitching is a complex process that integrates various computer vision techniques to create a single, wide field-of-view image from multiple overlapping photographs. The provided Python notebooks demonstrate a robust pipeline that can be generalized to most image stitching applications. Understanding this overall pipeline and its inherent challenges is crucial for developing effective and high-quality stitching solutions.

### The Image Stitching Pipeline

The typical image stitching pipeline, as exemplified by the provided code, consists of several sequential stages:

1.  **Image Acquisition**: This initial step involves capturing a set of images with sufficient overlap. The quality of the input images (e.g., resolution, exposure, focus, and overlap percentage) significantly impacts the final panorama quality.

2.  **Keypoint Detection**: As detailed in a previous section, this stage identifies distinctive and repeatable points (keypoints) in each image. The notebooks use a combination of DoG for scale-space extrema detection and Harris Corner Detector for refinement, ensuring robust feature localization across scales and rotations.

3.  **Feature Description**: Once keypoints are found, a descriptor is computed for each keypoint. This descriptor is a compact, invariant representation of the local image region around the keypoint. The notebooks employ ORB descriptors, known for their efficiency and robustness to rotation.

4.  **Feature Matching**: This stage establishes correspondences between keypoints from different images. A Brute-Force (BF) matcher is used to compare descriptors, and a ratio test (Lowe's ratio test) is applied to filter out ambiguous matches, yielding a set of reliable point correspondences.

5.  **Geometric Transformation Estimation**: With a set of matched keypoints, a geometric transformation that maps points from one image to another is estimated. The notebooks offer custom implementations for Homography, Affine, and Euclidean transformations, robustly estimated using RANSAC. Homography is generally preferred for general panoramic stitching as it can handle perspective distortions.

6.  **Image Warping and Canvas Generation**: The estimated transformation matrices are then used to warp the input images onto a common panoramic canvas. This involves calculating the overall size and offset of the final panorama by projecting the corners of all images and determining their bounding box. Each image is then transformed and placed onto this larger canvas.

7.  **Image Blending**: The final step is to seamlessly blend the warped images to create a cohesive panorama, minimizing visible seams or artifacts. The `multistitch_customH_multiband.ipynb` notebook utilizes multi-band blending with Laplacian pyramids, a sophisticated technique that blends images at different frequency levels to achieve smooth transitions.

### Common Challenges in Image Stitching

Despite significant advancements, image stitching still faces several challenges that can degrade the quality of the final panorama:

1.  **Parallax**: This is one of the most significant challenges. Parallax occurs when the camera's viewpoint changes between shots, causing objects at different depths to shift relative to each other. If the scene is not perfectly planar or the camera is not rotated precisely around its optical center, parallax errors can lead to ghosting, double images, or misalignments, especially for nearby objects. While homographies can perfectly align planar scenes, they struggle with non-planar scenes under parallax.

2.  **Photometric Inconsistencies**: Variations in illumination, exposure, white balance, and vignetting between individual images can lead to visible seams and color differences in the stitched panorama. Even if images are perfectly aligned geometrically, photometric discrepancies can make the composite image appear unnatural. Advanced blending techniques like multi-band blending help mitigate this, but significant differences can still be problematic.

3.  **Lens Distortion**: Real-world camera lenses introduce distortions (e.g., barrel or pincushion distortion) that can cause straight lines to appear curved. If not corrected, these distortions can lead to misalignments and wavy lines in the panorama. Pre-calibration of the camera and undistortion of images are often necessary steps.

4.  **Moving Objects**: If there are moving objects in the scene (e.g., people, cars, clouds) between the capture of successive images, these objects will appear in different positions in different images. This can result in ghosting or fragmented objects in the stitched output. Some advanced techniques attempt to detect and remove moving objects or use specialized blending strategies.

5.  **Lack of Features/Repetitive Patterns**: In scenes with large uniform areas (e.g., a clear sky, a blank wall) or highly repetitive patterns (e.g., a brick wall), keypoint detection and matching can be challenging. A scarcity of unique features can lead to an insufficient number of matches, making robust transformation estimation difficult or impossible.

6.  **Computational Complexity**: High-resolution images and a large number of input images can lead to significant computational demands for feature detection, matching, transformation, and blending. Optimizing algorithms and utilizing parallel processing are often necessary for real-time or large-scale applications.

7.  **Scale and Rotation Invariance**: While algorithms like DoG and ORB are designed to be scale and rotation invariant, extreme changes in scale or rotation between images can still pose difficulties for robust feature matching and transformation estimation.

Addressing these challenges often requires a combination of careful image acquisition, robust algorithms, and sometimes user intervention or specialized hardware. The custom implementations in the provided notebooks, particularly the robust RANSAC-based transformation estimation and multi-band blending, are designed to tackle many of these common issues, contributing to higher quality stitched results.



## Comparison and Insights

The two provided Python notebooks, `multistitch_customH_multiband.ipynb` and `dog_harris_alignment_customH_fixed.ipynb`, offer valuable insights into different aspects of image stitching and alignment. While both share common foundational components like DoG+Harris keypoint detection and ORB descriptor matching, they diverge in their scope and advanced techniques, particularly in handling multiple images and blending strategies.

### Shared Foundations

Both notebooks implement the core steps of feature-based image alignment:

*   **DoG+Harris Keypoint Detection**: Both use the same custom implementation for detecting keypoints, combining the multi-scale blob detection of DoG with the corner response filtering of Harris. This ensures a robust set of distinctive keypoints that are stable across scale and rotation changes.
*   **ORB Descriptors**: Both rely on ORB for generating binary descriptors, leveraging its efficiency and rotation invariance. This choice is practical for real-time applications where computational speed is a concern.
*   **Brute-Force Matching with Ratio Test**: The descriptor matching process is identical, employing a Brute-Force matcher with Hamming distance for ORB descriptors, followed by Lowe's ratio test to filter out ambiguous matches. This step is crucial for providing a clean set of correspondences for transformation estimation.
*   **Custom RANSAC Implementation**: A significant commonality is the custom implementation of RANSAC for robust transformation estimation. Instead of relying on `cv2.findHomography`, both notebooks build their own RANSAC loop, demonstrating a deep understanding of the algorithm. This custom approach allows for flexibility in integrating different transformation models and error metrics.

### Divergence in Scope and Advanced Techniques

The primary difference between the two notebooks lies in their intended application and the advanced techniques they employ:

#### `dog_harris_alignment_customH_fixed.ipynb`: Two-Image Alignment

This notebook focuses specifically on aligning and stitching **two images**. Its main advanced feature is the custom homography estimation using Normalized DLT within RANSAC. Key aspects include:

*   **Homography-Centric**: It is designed to estimate a homography, which is suitable for aligning two images where the scene is planar or the camera motion is primarily rotational. The `dlt_homography` function, with its normalization step, is a robust implementation of the Direct Linear Transform.
*   **Simpler Stitching**: The `warp_and_stitch` function performs a basic stitching operation. It warps the second image to the first image's plane and then directly overlays the first image onto the warped second image within a newly computed canvas. This approach is simpler and effective for two images but can lead to visible seams if photometric differences are significant or if more than two images are involved.
*   **Educational Focus**: This notebook serves as an excellent educational tool for understanding the fundamental steps of two-image alignment, particularly the custom implementation of homography estimation and RANSAC.

#### `multistitch_customH_multiband.ipynb`: Multi-Image Stitching with Multi-Band Blending

This notebook extends the concept to **multiple images** and introduces a more sophisticated blending technique. Its advanced features include:

*   **Multiple Transformation Models**: Beyond homography, this notebook also provides custom implementations for Affine and Euclidean transformations within its `ransac_transform` function. This allows for greater flexibility in choosing the appropriate geometric model based on the camera motion and scene structure.
    *   **Euclidean (Rotation + Translation)**: Uses the Kabsch algorithm for rigid transformations, suitable for scenarios where only rotation and translation occur.
    *   **Affine**: Estimates a 6-DOF affine transformation, useful when scaling and shearing are also present but perspective effects are negligible.
*   **Multi-Image Alignment Strategy**: It aligns all subsequent images to a common reference image (the first image in the sequence). This is a common strategy for building panoramas incrementally.
*   **Multi-Band Blending (Laplacian Pyramids)**: This is the most significant advancement in this notebook. The `multiband_blend` function, utilizing Gaussian and Laplacian pyramids, provides a seamless blending solution. By blending images at different frequency bands, it effectively minimizes visible seams and artifacts caused by photometric inconsistencies or slight misalignments. This is crucial for high-quality panoramic outputs, especially when dealing with multiple images and varying illumination conditions.
*   **Comprehensive Canvas Computation**: The `compute_canvas_and_warp` function is designed to handle multiple warped images, calculating a global canvas that can accommodate all transformed images and generating masks for blending.

### Insights and Best Practices

1.  **Custom Implementations for Deeper Understanding**: The custom implementations of DLT, Affine, Euclidean, and RANSAC are invaluable for understanding the underlying mathematics and algorithms. While OpenCV provides optimized functions (e.g., `cv2.findHomography`), implementing them from scratch offers greater control and insight into their workings and limitations.

2.  **Importance of Normalization**: The `normalize_points` function for DLT homography estimation highlights the critical role of data normalization in numerical stability. Without it, solving linear systems for transformations can be highly susceptible to noise and lead to inaccurate results.

3.  **Robustness with RANSAC**: The RANSAC algorithm is indispensable for practical computer vision applications where data is noisy and contains outliers. Its ability to robustly estimate model parameters, even with a high percentage of incorrect matches, is fundamental to achieving reliable image alignment.

4.  **Choosing the Right Transformation Model**: The availability of Homography, Affine, and Euclidean models in `multistitch_customH_multiband.ipynb` underscores the importance of selecting the appropriate geometric transformation. A homography is powerful but requires a planar scene or pure camera rotation. For simpler motions, Affine or Euclidean transformations might be more stable and computationally less demanding.

5.  **Seamless Blending for Quality**: Multi-band blending using Laplacian pyramids is a gold standard for achieving high-quality, seamless panoramas. It addresses the photometric inconsistencies and subtle misalignments that simpler blending methods cannot, significantly enhancing the visual appeal of the final stitched image.

6.  **Modular Design**: Both notebooks exhibit a modular design, breaking down the complex image stitching process into manageable functions for keypoint detection, descriptor computation, matching, transformation estimation, warping, and blending. This modularity improves readability, maintainability, and allows for easy experimentation with different components.

In summary, `dog_harris_alignment_customH_fixed.ipynb` provides a solid foundation for understanding two-image alignment with custom homography, while `multistitch_customH_multiband.ipynb` builds upon this by extending to multi-image scenarios and incorporating advanced multi-band blending, showcasing a more complete and high-quality image stitching pipeline. The combination of these two notebooks offers a comprehensive view of the techniques involved in creating impressive panoramic images.



## Conclusion

Image stitching is a sophisticated field within computer vision that seamlessly merges multiple images into a single, expansive panorama. This report has dissected the intricate pipeline of image stitching, drawing extensively from the methodologies presented in `multistitch_customH_multiband.ipynb` and `dog_harris_alignment_customH_fixed.ipynb`. From the initial identification of salient features to the final seamless integration of images, each stage is underpinned by advanced algorithms and mathematical principles.

The journey begins with **Keypoint Detection**, where the synergy of Difference of Gaussians (DoG) and the Harris Corner Detector ensures the identification of stable and distinctive points across varying scales and orientations. These keypoints are then characterized by **ORB Descriptors**, compact binary representations that are efficient and robust to rotation. The subsequent **Feature Matching** phase employs a Brute-Force approach coupled with Lowe's ratio test to establish reliable correspondences, effectively filtering out ambiguous matches.

At the heart of image alignment are **Geometric Transformations**, including Homography, Affine, and Euclidean models. The custom implementations of these transformations, particularly the Normalized Direct Linear Transform (DLT) for homography, highlight the importance of numerical stability through normalization. To combat the pervasive issue of outliers in real-world data, the **RANSAC (Random Sample Consensus)** algorithm is critically applied, providing a robust framework for estimating transformation parameters by iteratively identifying and leveraging inliers.

Once transformations are determined, **Image Warping** projects the individual images onto a common canvas, whose dimensions are meticulously calculated to encompass all transformed content. Finally, **Multi-Band Blending** using Laplacian pyramids stands as a testament to advanced image processing, ensuring that the stitched images merge without visible seams or artifacts by blending frequency components across multiple scales.

While the `dog_harris_alignment_customH_fixed.ipynb` notebook provides a foundational understanding of two-image alignment with custom homography, `multistitch_customH_multiband.ipynb` expands this to multi-image scenarios and introduces the critical multi-band blending technique, showcasing a more complete and high-quality stitching pipeline. The comparative analysis reveals that while core feature-based methods are shared, the choice of transformation model and blending strategy significantly impacts the quality and applicability of the final panorama.

Despite the robustness of these techniques, challenges such as parallax, photometric inconsistencies, lens distortion, moving objects, and feature scarcity remain. Addressing these requires a continuous evolution of algorithms and careful consideration of the imaging environment. Nevertheless, the principles and implementations explored in this report provide a solid foundation for understanding and developing high-quality image stitching solutions, enabling the creation of immersive and expansive visual experiences.

## References

1. [Difference of Gaussians - Wikipedia](https://en.wikipedia.org/wiki/Difference_of_Gaussians)
2. [Harris corner detector - Wikipedia](https://en.wikipedia.org/wiki/Harris_corner_detector)
3. [ORB (Oriented FAST and Rotated BRIEF) - OpenCV Documentation](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
4. [OpenCV: Feature Matching - OpenCV Documentation](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
5. [Estimating the Homography Matrix with the Direct Linear Transform (DLT) - Medium](https://medium.com/@insight-in-plain-sight/estimating-the-homography-matrix-with-the-direct-linear-transform-dlt-ec6bbb82ee2b)
6. [Why normalize the data set before applying Direct Linear Transform - StackExchange](https://dsp.stackexchange.com/questions/10887/why-normalize-the-data-set-before-applying-direct-linear-transform)
7. [Affine transformation - Wikipedia](https://en.wikipedia.org/wiki/Affine_transformation)
8. [Euclidean transformation - PlanetMath](https://planetmath.org/euclideantransformation)
9. [Kabsch algorithm - Wikipedia](https://en.wikipedia.org/wiki/Kabsch_algorithm)
10. [Random sample consensus - Wikipedia](https://en.wikipedia.org/wiki/Random_sample_consensus)
11. [Image warping - Wikipedia](https://en.wikipedia.org/wiki/Image_warping)
12. [Image Blending Using Laplacian Pyramids - Becoming Human](https://becominghuman.ai/image-blending-using-laplacian-pyramids-2f8e9982077f)
13. [What Is Image Stitching? - Baeldung on Computer Science](https://www.baeldung.com/cs/image-stitching)
14. [Approaches and Challenges in Real Time Image Stitching - IJERA](https://www.ijera.com/papers/vol10no5/Series-1/C1005011924.pdf)
