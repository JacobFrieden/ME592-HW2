from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import AffineTransform, warp
import os
from sklearn.decomposition import PCA
import cv2

def crop_to_non_black_bounding_box(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to get binary image
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find bounding box of the non-black content
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        for contour in contours[1:]:
            contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
            x = min(x, contour_x)
            y = min(y, contour_y)
            w = max(w, contour_x + contour_w - x)
            h = max(h, contour_y + contour_h - y)
        
        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]
    else:
        cropped_image = image
    
    return cropped_image

def random_transformation(img):
    rows, cols, ch = img.shape
    
    # Random rotation
    angle = np.random.randint(-180, 180)
    M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    
    # Random scaling
    scale = np.random.uniform(0.5, 1.5)
    M_scale = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    
    # Random shifting
    tx = np.random.randint(-50, 50)
    ty = np.random.randint(-50, 50)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply transformations
    img = cv2.warpAffine(img, M_rotate, (cols, rows))
    img = cv2.warpAffine(img, M_scale, (cols, rows))
    img = cv2.warpAffine(img, M_shift, (cols, rows))
    
    # Random warping
    random_quad = np.float32([
        [np.random.randint(0, cols // 4), np.random.randint(0, rows // 4)],
        [np.random.randint(3 * cols // 4, cols), np.random.randint(0, rows // 4)],
        [np.random.randint(0, cols // 4), np.random.randint(3 * rows // 4, rows)],
        [np.random.randint(3 * cols // 4, cols), np.random.randint(3 * rows // 4, rows)]
    ])
    original_quad = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M_warp = cv2.getPerspectiveTransform(original_quad, random_quad)
    img = cv2.warpPerspective(img, M_warp, (cols, rows))
    
    img = crop_to_non_black_bounding_box(img)

    return img

# Create directory for transformed images
transformed_images_dir = 'AgandBio/leaves/transformed_images'
os.makedirs(transformed_images_dir, exist_ok=True)

# Part 1-----------------------------------------------------------------------
# Generate 100 transformed images
transformed_images = []
for i in range(100):
    original_img_idx = np.random.randint(1,9)
    original_image = cv2.imread(f'AgandBio/leaves/I{original_img_idx}.png')
    new_img = random_transformation(original_image)
    # cv2.imwrite(os.path.join(transformed_images_dir, f'I{original_img_idx}_transformed_l{i+1}.png'), new_img)

# Part 2 ----------------------------------------------------------------------
# Function to extract random patches from the image

def extract_patches(img, num_patches=10, patch_size=(50, 50)):
    patches = []
    img_height, img_width = img.shape[:2]  # Extract the height and width from the image shape
    for _ in range(num_patches):
        # Ensure the random top-left corner is within the bounds that allow for a full patch
        x = random.randint(0, img_width - patch_size[1])
        y = random.randint(0, img_height - patch_size[0])
        patch = img[y:y+patch_size[0], x:x+patch_size[1]]
        patches.append(patch)
    return patches

# Assume the transformed_images_dir is correctly set to where your images are stored
transformed_images_dir = 'AgandBio/leaves/transformed_images'

# Create directory for transformed patched images
transformed_patches_dir = 'AgandBio/leaves/transformed_patches'
os.makedirs(transformed_patches_dir, exist_ok=True) 

for file in os.listdir(transformed_images_dir):
    #print(file)
    img = cv2.imread(os.path.join(transformed_images_dir, file))
    patches = extract_patches(img, num_patches=5)
    for i, patch in enumerate(patches):
        cv2.imwrite(os.path.join(transformed_patches_dir, f'{file[:-4]}_patch{i+1}.png'), patch)

# Part 3 ----------------------------------------------------------------------

def zca_whitening(patches):
       
    patches_flatten = np.array([patch.flatten() for patch in patches])
    
    patches_mean = np.mean(patches_flatten, axis=0)
    patches_centered = patches_flatten - patches_mean
    
    sigma = np.cov(patches_centered, rowvar=False)
    
    U, S, V = np.linalg.svd(sigma)
    
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    
    patches_zca = np.dot(patches_centered, zca_matrix)
    
    # Adjusted normalization step
    delta = patches_zca.max() - patches_zca.min()
    if delta > 0:
        patches_zca_rescaled = 255 * (patches_zca - patches_zca.min()) / delta
    else:
        # Avoid division by zero by using the original centered patches
        patches_zca_rescaled = 255 * (patches_centered - patches_centered.min()) / max(1, patches_centered.max() - patches_centered.min())
    
    patches_zca_rescaled = np.clip(patches_zca_rescaled, 0, 255)  # Ensure values are within [0, 255]
    
    original_shape = patches[0].shape
    patches_zca_reshaped = np.array([patch.reshape(original_shape) for patch in patches_zca_rescaled]).astype(np.uint8)
    
    return patches_zca_reshaped

images_directory = "AgandBio/leaves/transformed_patches"
whitened_images_directory = "AgandBio/leaves/whitened_patches"
os.makedirs(whitened_images_directory, exist_ok=True) 
for file in os.listdir(images_directory):
    img = cv2.imread(os.path.join(images_directory, file))
    whitened = zca_whitening(img)
    cv2.imwrite(os.path.join(whitened_images_directory, f'whitened_{file[:-4]}.png'), whitened)

# Part 4 & 5 ----------------------------------------------------------------
def plot_channel_distributions(images):
    # Assuming a list of image patches and each patch is in the format (height, width, channels)
    # Initialize lists to hold pixel values of each channel
    channels_data = [[] for _ in range(images[0].shape[2])]
    
    # Separate the pixel values of each channel
    for patch in images:
        for channel in range(patch.shape[2]):
            channels_data[channel].extend(patch[:, :, channel].flatten())

    # Plotting
    fig, axes = plt.subplots(1, images[0].shape[2], figsize=(15, 5))
    if images[0].shape[2] == 1:  # If grayscale, adjust the axes array
        axes = [axes]
    
    for i, data in enumerate(channels_data):
        axes[i].hist(data, bins=50, alpha=0.7, label=f'Channel {i+1}')
        axes[i].set_title(f'Channel {i+1} Distribution')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

whitened = []
for file in os.listdir(whitened_images_directory):
    whitened.append(cv2.imread(os.path.join(whitened_images_directory,file)))

original = []
for file in os.listdir(transformed_patches_dir):
    original.append(cv2.imread(os.path.join(transformed_patches_dir,file)))

plot_channel_distributions(whitened)
plot_channel_distributions(original)
