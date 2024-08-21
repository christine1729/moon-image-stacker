import cv2
import numpy as np
from glob import glob

def calculate_image_quality(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def align_with_opencv(image, reference):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(1000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
    
    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    
    # Check if matches is empty
    if not matches:
        raise ValueError("No matches found between images")
    
    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # Use homography to warp image
    height, width, channels = reference.shape
    aligned = cv2.warpPerspective(image, h, (width, height))
    
    return aligned

def align_images(images, reference_index):
    reference = images[reference_index]
    aligned_images = [reference]
    
    for i, image in enumerate(images):
        if i == reference_index:
            continue
        try:
            print(f"Attempting to align image {i} with OpenCV...")
            aligned = align_with_opencv(image, reference)
            aligned_images.append(aligned)
            print(f"Successfully aligned image {i}.")
        except Exception as e:
            print(f"Could not align image {i}, skipping... Error: {str(e)}")
    
    return aligned_images

def lucky_imaging(images, percentile=70):
    qualities = [calculate_image_quality(img) for img in images]
    print("Quality for the images: ", qualities)
    threshold = np.percentile(qualities, percentile)
    return [img for img, quality in zip(images, qualities) if quality >= threshold]

def stack_images(images):
    float_images = [np.float32(img) for img in images]
    stacked = np.median(float_images, axis=0)
    return np.uint8(cv2.normalize(stacked, None, 0, 255, cv2.NORM_MINMAX))

def sharpen_image(image, amount=1.5, radius=1, threshold=0):
    """
    Sharpen the image using an unsharp mask.
    
    :param image: Input image
    :param amount: Sharpening strength (1.0 means no sharpening)
    :param radius: Radius of Gaussian blur
    :param threshold: Minimum brightness change to apply sharpening
    :return: Sharpened image
    """
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def main():
    # Load images
    image_files = glob('path/to/your/moon/images/*.jpg')  # Adjust the path
    images = [cv2.imread(file) for file in image_files]
    
    # Find the best quality image
    print("Finding the best quality image...")
    best_image_index = np.argmax([calculate_image_quality(img) for img in images])
    
    # Align images
    print("Aligning images...")
    aligned_images = align_images(images, best_image_index)
    
    # Select best frames (lucky imaging)
    print("Selecting best frames...")
    lucky_frames = lucky_imaging(aligned_images)
    
    # Stack images
    print("Stacking images...")
    stacked_image = stack_images(lucky_frames)
    
    # Sharpen the stacked image
    print("Sharpening the stacked image...")
    sharpened_image = sharpen_image(stacked_image, amount=1.5, radius=1, threshold=10)
    
    # Save results
    cv2.imwrite('stacked_moon.jpg', stacked_image)
    cv2.imwrite('sharpened_stacked_moon.jpg', sharpened_image)
    print("Stacked image saved as 'stacked_moon.jpg'")
    print("Sharpened stacked image saved as 'sharpened_stacked_moon.jpg'")

if __name__ == "__main__":
    main()
