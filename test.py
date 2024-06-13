import cv2
import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pyramid(image, scale=1.5, min_size=(224, 224)):
    """
    Generate image pyramid.
    
    Parameters:
    - image: The input image.
    - scale: The scaling factor for the pyramid.
    - min_size: The minimum size of the pyramid image.
    
    Returns:
    - A list of images in the pyramid.
    """
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

def sliding_window(image, step_size, window_size):
    """
    Slide a window across the image.
    
    Parameters:
    - image: The input image.
    - step_size: The step size for sliding.
    - window_size: The size of the window.
    
    Returns:
    - A list of tuples, each containing the coordinates of the top-left corner and the window image.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Paths to your input images and where you want to save the small images and labels
input_image_dir = 'path/to/large_images'
output_image_dir = 'path/to/small_images'
output_label_dir = 'path/to/labels'

# Create directories if they don't exist
create_dir(output_image_dir)
create_dir(output_label_dir)

# Define window size and step size
window_size = (224, 224)
step_size = 112  # 50% overlap

# Dummy function to get YOLO format labels for the small window
# Replace this with your actual label extraction logic
def get_yolo_labels(window, x_offset, y_offset, original_width, original_height):
    """
    Generate YOLO format labels for the given window.
    
    Parameters:
    - window: The image window.
    - x_offset: The x offset of the window in the original image.
    - y_offset: The y offset of the window in the original image.
    - original_width: The width of the original image.
    - original_height: The height of the original image.
    
    Returns:
    - A list of YOLO format labels.
    """
    labels = []
    # Example: (class_id, x_center, y_center, width, height)
    # Normalize coordinates by the size of the window
    # Example values for demonstration, replace with actual detection logic
    class_id = 0
    x_center = (x_offset + window.shape[1] // 2) / original_width
    y_center = (y_offset + window.shape[0] // 2) / original_height
    width = window.shape[1] / original_width
    height = window.shape[0] / original_height
    labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return labels

# Process each image in the input directory
for image_name in os.listdir(input_image_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_image_dir, image_name)
        image = cv2.imread(image_path)
        original_height, original_width = image.shape[:2]

        # Generate image pyramid and process each layer
        for layer_num, layer in enumerate(pyramid(image, scale=1.5)):
            for (x, y, window) in sliding_window(layer, step_size, window_size):
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    continue
                # Save the small window image
                small_image_name = f"{os.path.splitext(image_name)[0]}_layer{layer_num}_x{x}_y{y}.jpg"
                small_image_path = os.path.join(output_image_dir, small_image_name)
                cv2.imwrite(small_image_path, window)

                # Generate YOLO format labels for the window
                yolo_labels = get_yolo_labels(window, x, y, original_width, original_height)

                # Save the labels to a file
                label_name = f"{os.path.splitext(small_image_name)[0]}.txt"
                label_path = os.path.join(output_label_dir, label_name)
                with open(label_path, 'w') as label_file:
                    for label in yolo_labels:
                        label_file.write(label + '\n')
