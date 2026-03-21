import copy
import cv2 as cv,cv2
import numpy as np
from torch import nn
import ctypes
import pytorch_ssim

l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()
import os
import glob
import sys

def save_mask(mask, folder_path, filename=None, overwrite=False, dsize=None):
    """
    Save a binarized mask image (bool type or 0/1 values) after converting it to a 0/255 single-channel image.

    @param mask: NumPy array in bool, 0/1, or 0/255 format, either single-channel or three-channel.
    @param folder_path: Output path.
    @param filename: Custom file name. If no extension is provided, `.png` is appended automatically.
    @param overwrite: Whether to overwrite an existing file.
    @param dsize: Output size in the format (width, height). No resizing by default.
    @return: Full path of the saved file.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Convert the mask to a 0/255 uint8 single-channel image
    if mask.dtype == np.bool_:
        mask = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)

    if mask.ndim == 3 and mask.shape[2] > 1:
        mask = mask[:, :, 0]  # Ensure single-channel output

    # Resize if requested
    if dsize is not None:
        mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)

    # Generate the output file name
    if filename:
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = '.png'
        new_filename = name + ext
    else:
        new_filename = "mask_0001.png"
        if not overwrite:
            i = 1
            while os.path.exists(os.path.join(folder_path, new_filename)):
                new_filename = f"mask_{i:04d}.png"
                i += 1

    full_path = os.path.join(folder_path, new_filename)

    if not overwrite and os.path.exists(full_path):
        base, ext = os.path.splitext(new_filename)
        j = 1
        while os.path.exists(os.path.join(folder_path, f"{base}_{j:03d}{ext}")):
            j += 1
        new_filename = f"{base}_{j:03d}{ext}"
        full_path = os.path.join(folder_path, new_filename)

    # Save the file
    cv2.imwrite(full_path, mask)
    return full_path

def thresh(im_in):
    # Convert to grayscale if needed
    if im_in.ndim == 3:
        im_in = cv.cvtColor(im_in, cv.COLOR_BGR2GRAY)
    if im_in.dtype == 'float32':
        im_in = np.uint8(im_in * 255)
    # Otsu thresholding with Gaussian blur
    _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Morphological closing to fill small holes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    im_mask = cv.morphologyEx(im_mask, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(im_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(im_mask, dtype=bool), []

    # Use the largest contour directly (no convex hull)
    largest_contour = max(contours, key=cv.contourArea)

    # Fill the contour itself (not convex hull)
    im_mask_filled = np.zeros_like(im_mask, dtype=np.uint8)
    cv.drawContours(im_mask_filled, [largest_contour], contourIdx=-1, color=1, thickness=-1)
    im_mask_bool = im_mask_filled > 0

    # Bounding box of the contour
    bbox = cv.boundingRect(largest_contour)
    corners = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]]
    ]
    # Normalize corners to (-1, 1)
    h, w = im_in.shape[:2]
    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1
    return im_mask_bool, corners


def visualize_mask_with_opencv(mask: np.ndarray, window_name: str = "Mask"):
    """
    Display a mask image using OpenCV.

    Args:
        mask (np.ndarray): Binarized mask image in bool, uint8, or float format.
        window_name (str): Display window name.
    """
    # Make sure the mask is uint8 and in the range [0, 255]
    if mask.dtype == np.bool_:
        mask_vis = (mask.astype(np.uint8)) * 255
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_vis = np.clip(mask * 255, 0, 255).astype(np.uint8)
    elif mask.dtype == np.uint8:
        mask_vis = mask.copy()
    else:
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")

    # Display directly if it is single-channel
    if mask_vis.ndim == 2:
        pass
    elif mask_vis.ndim == 3 and mask_vis.shape[2] == 1:
        mask_vis = mask_vis[:, :, 0]
    else:
        raise ValueError(f"Expected single-channel mask, got shape {mask.shape}")

    # Visualization
    cv.imshow(window_name, mask_vis)
    cv.waitKey(0)
    cv.destroyAllWindows()


def compute_mask_from_paths(path1, path2):
    """
    Read images from `path1` and `path2`, compute their difference, and use `thresh` to extract the mask and corner points.

    Args:
        path1 (str): Path to the source image (for example, the background image)
        path2 (str): Path to the target image (for example, the image containing the foreground)

    Returns:
        im_mask (np.ndarray): Mask in bool format; True indicates the foreground region.
        corners (list of list): Four normalized corner coordinates in the order [top-left, top-right, bottom-right, bottom-left]
    """

    # Read the images
    img1 = cv.imread(path1, cv.IMREAD_COLOR)
    img2 = cv.imread(path2, cv.IMREAD_COLOR)

    # Safety checks
    if img1 is None:
        raise FileNotFoundError(f"Cannot read image at {path1}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read image at {path2}")

    # Convert images to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Compute the difference image (absolute value)
    diff = cv.absdiff(gray2, gray1)

    # Call the thresh function
    im_mask, corners = thresh(diff)

    return im_mask, corners

def delete_images_in_folder(folder_path):
    """
    Delete all image files in the specified folder without removing the folder itself.

    Args:
        folder_path (str): Path to the folder whose images should be cleared.
    """
    # Supported image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    files = os.listdir(folder_path)
    deleted_count = 0

    for filename in files:
        if filename.lower().endswith(valid_exts):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete: {filename}, error: {e}")

    print(f"🧹 Deleted {deleted_count} image files from: {folder_path}")
def compute_direct_indirect_light(im_cb, backlight_strength=0.9):
    """
    Compute the direct-light and indirect-light components (based on Nayar TOG'06, referenced by Moreno 3DV'12).

    Args:
        im_cb: np.ndarray
            Camera-captured checkerboard image sequence. The shape should be (H, W, C, N),
            or torch.Tensor with shape (N, C, H, W).
        backlight_strength: float
            Projector backlight strength. Default is 0.9 (a higher value is recommended for mask extraction).

    Returns:
        im_direct: np.ndarray
            Direct-light image with shape (H, W, C)
        im_indirect: np.ndarray
            Indirect-light image with shape (H, W, C)
    """
    # If the input is a torch tensor, convert it to NumPy and adjust the dimensions first
    if hasattr(im_cb, 'numpy'):
        im_cb = im_cb.numpy().transpose((2, 3, 1, 0))  # (H, W, C, N)

    l1 = np.max(im_cb, axis=3)  # Maximum image L+
    l2 = np.min(im_cb, axis=3)  # Minimum image L-

    b = backlight_strength
    im_direct = (l1 - l2) / (1 - b)  # Direct light
    im_indirect = 2 * (l2 - b * l1) / (1 - b ** 2)  # Indirect light
    im_indirect = np.clip(im_indirect, a_min=0.0, a_max=None)

    # If the indirect light is negative, use L+ directly for the direct-light component
    im_direct[im_indirect < 0] = l1[im_indirect < 0]

    return im_direct, im_indirect

def check_image_folder_exists(folder_path, verbose=True, exit_on_fail=True):
    """
    Check whether image files exist in the specified folder.

    Args:
        folder_path (str): Folder path to check.
        verbose (bool): Whether to print log messages.
        exit_on_fail (bool): Whether to exit the program if no image files are found.

    Returns:
        List[str]: A list of all discovered image file paths.
    """
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        msg = f"Error: No image files found in {folder_path}."
        if verbose:
            print(msg)
        if exit_on_fail:
            sys.exit(1)
        else:
            raise FileNotFoundError(msg)
    else:
        if verbose:
            print(f"✅ Found {len(image_files)} image(s) in {folder_path}.")

    return image_files


def checkerboard(*args):
    # Python implementation of MATLAB's checkerboard function
    # Parse inputs
    n = 10
    p = 4
    q = p

    if len(args) == 1:
        n = args[0]
    elif len(args) == 2:
        n, p = args
        q = p
    elif len(args) == 3:
        n, p, q = args

    # Generate tile
    tile = np.tile(np.kron([[0, 1], [1, 0]], np.ones((n, n))), (1, 1))

    # Create checkerboard
    if q % 2 == 0:
        # Build the left and right sections separately
        num_col_reps = int(np.ceil(q / 2))
        ileft = np.tile(tile, (p, num_col_reps))

        tile_right = np.tile(np.kron([[0, 0.7], [0.7, 0]], np.ones((n, n))), (1, 1))
        iright = np.tile(tile_right, (p, num_col_reps))

        # Tile the left and right halves together
        checkerboard = np.concatenate((ileft, iright), axis=1)
    else:
        # Build the entire image in one shot
        checkerboard = np.tile(tile, (p, q))

        # Make the right half use light-gray tiles
        mid_col = int(checkerboard.shape[1] / 2) + 1
        checkerboard[:, mid_col:] = checkerboard[:, mid_col:] - .3
        checkerboard[np.where(checkerboard < 0)] = 0

    return checkerboard.astype('float64')

def load_images_from_folder(folder_path):
    """
    Read all images from the specified folder and pack them into a 4D array with shape N x H x W x C.
    @param folder_path: Path to the folder containing images
    @return: A 4D NumPy array with shape N x H x W x C
    """
    image_list = []
    common_height = None
    common_width = None
    common_channels = None

    # Get all file names in the folder
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # # Convert from BGR to RGB
            # SCNet_surf1_img = cv2.cvtColor(SCNet_surf1_img, cv2.COLOR_BGR2RGB)

            if common_height is None:
                # First image: determine the base size and number of channels
                common_height, common_width, common_channels = img.shape
            else:
                # Other images: check size and number of channels
                h, w, c = img.shape
                if h != common_height or w != common_width or c != common_channels:
                    print(f"Resizing image {filename} to common size ({common_height}, {common_width}, {common_channels}).")
                    img = cv2.resize(img, (common_width, common_height))

            image_list.append(img)
        else:
            print(f"Warning: Unable to read image {img_path}")

    if len(image_list) == 0:
        print("no image in the folder")

    # Stack all images into a 4D array
    image_array = np.stack(image_list, axis=0)

    return image_array


def save_correspondences_to_yml(cam_pts, prj_pts, filename='sl.yml'):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write("cam_pts", cam_pts)
    fs.write("prj_pts", prj_pts)
    print("cam_pts min:", np.min(cam_pts, axis=0))
    print("cam_pts max:", np.max(cam_pts, axis=0))
    print("prj_pts min:", np.min(prj_pts, axis=0))
    print("prj_pts max:", np.max(prj_pts, axis=0))
    fs.release()


class Camera:
    def __init__(self, camera_index=0, window_name='Camera Preview, Press q to exit', delay_frames=5, delay_time=0.2):
        """
        Initialize a Camera instance.

        Args:
        - camera_index: Integer index of the camera to use (default: 0).
        - window_name: Name of the preview window (default: 'Camera Preview, Press q to exit').
        - delay_frames: Number of frames to discard before capturing the latest image (default: 5).
        - delay_time: Time interval between projection and capture in seconds (default: 0.1 seconds).
        """
        self.camera_index = camera_index  # Camera index
        self.window_name = window_name  # Window name
        self.delay_frames = delay_frames  # Number of frames to discard
        self.delay_time = delay_time  # Time interval between projection and capture
        self.cap = cv2.VideoCapture(camera_index)  # Open the camera with the specified index
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open the camera")  # Raise an exception if the camera cannot be opened

    def clear_buffer(self):
        """
        Clear the camera buffer by reading and discarding multiple frames.
        """
        for _ in range(20):
            self.cap.read()

    def capturing(self):
        """
        Capture the latest frame from the camera after discarding delayed frames.

        Returns:
        - frame: The captured image frame.

        Raises:
        - RuntimeError: If camera data cannot be read.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read camera data")  # Raise an exception if reading fails
        return frame  # Return the captured image frame

    def preview(self):
        """
        Display the live camera feed until the 'q' key is pressed.
        """
        while True:
            ret, frame = self.cap.read()  # Capture one frame
            if not ret:
                raise RuntimeError("Unable to read camera data")

            # Display the captured image in the window
            cv2.imshow(self.window_name, frame)

            # Check whether the 'q' key is pressed; if so, exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    def release(self):
        """
        Release the camera and close all OpenCV windows.
        """
        # Read and discard the specified number of delayed frames
        for _ in range(self.delay_frames):
            self.cap.read()
        self.cap.release()  # Release the camera
        cv2.destroyAllWindows()


class Projecting:
    def __init__(self, sw, sh):
        """
        Initialize an instance of the Projecting class.

        @param sw: Screen width
        @param sh: Screen height
        """
        super().__init__()
        self.prj_win = 'full_screen'
        self.sw, self.sh = sw, sh
        cv2.namedWindow(self.prj_win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.prj_win, 2561, 0)
        cv2.setWindowProperty(self.prj_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.img = np.zeros((self.sh, self.sw, 3), dtype=np.uint8)

    def projecting(self, img):
        """
        Project an image in a full-screen window.

        @param img: Image to be displayed
        """
        img_copy = img.copy()

        if img_copy.max() <= 1.0 and img_copy.min() >= 0.0:
            img_copy = (img_copy * 255).astype(np.uint8)

        img_h, img_w = img_copy.shape[:2]
        screen_ratio = self.sw / self.sh
        img_ratio = img_w / img_h

        if abs(img_ratio - 1.0) < 0.01:  # The image has a 1:1 aspect ratio
            # Resize based on height
            new_h = self.sh
            new_w = self.sh
        else:  # The image does not have a 1:1 aspect ratio
            new_h = self.sh
            new_w = int(self.sh * img_ratio)

        # Ensure that the image width and height do not exceed the screen size
        if new_w > self.sw:
            new_w = self.sw
            new_h = int(self.sw / img_ratio)

        if new_h > self.sh:
            new_h = self.sh
            new_w = int(self.sh * img_ratio)

        img_copy = cv2.resize(img_copy, (new_w, new_h))
        h, w = img_copy.shape[:2]

        start_x = (self.sw - w) // 2
        start_y = (self.sh - h) // 2

        # self.img.fill(0)  # Clear the previously displayed content
        self.img[start_y:start_y + h, start_x:start_x + w, :3] = img_copy[:, :, :]

        cv2.imshow(self.prj_win, self.img)
        cv2.waitKey(200)


def normalize_image(img):
    """
    Check whether an image is already normalized. If not, normalize it.

    @param img: Input image (numpy array)
    @return: Normalized image
    """
    # If the maximum value is greater than 1, treat the image as unnormalized and normalize it
    if img.max() > 1.0:
        img = img.astype(float) / 255.0

    return img

def extract_filename(file_path):
    """
    Extract the file name from a file path.

    Args:
    file_path : Full file path (string)

    Returns:
    File name (string)
    """
    return os.path.basename(file_path)
def read_and_normalize_image(image_path, normalize=True):
    """
    Read an image file and optionally normalize it.

    @param image_path: Path to the image file
    @param normalize: If True, normalize the image to the range [0, 1]. Default is True

    @return: The loaded and processed image
    """
    # Read the image file
    img = cv2.imread(image_path)

    # Check whether the image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Unable to load image from file: {image_path}")

    # If normalization is requested, normalize the image to [0, 1]
    if normalize:
        img = img.astype(float) / 255.0

    return img


def clamp_image(image, min_value=0.0, max_value=1.0):
    """

    @param image: Input image in OpenCV format
    @param min_value: 0
    @param max_value: 1
    @return: The clamped output, because the final computed image may exceed 1
    """
    return np.clip(image, min_value, max_value)

def create_directory_structure(base_path,current_time):
    """
    Create a set of folders under the specified root directory and return a dictionary of the created paths.

    @param base_path: Root directory path
    @return: Dictionary containing the created folder paths
    """
    # Define the folder structure that needs to be created
    directories = {
        "mask": f"{current_time}/cam/mask",
        # "cam_raw_cb": f"{current_time}/cam/raw/cb",
        "cam_raw_train": f"{current_time}/cam/raw/train",
        "cam_raw_test": f"{current_time}/cam/raw/test",
        "cam_raw_ref": f"{current_time}/cam/raw/ref",
        'cam_raw_sl': f"{current_time}/cam/raw/sl",
        "cam_warp_train": f"{current_time}/cam/warpSL/train",
        "cam_warp_test": f"{current_time}/cam/warpSL/test",
        "cam_warp_ref": f"{current_time}/cam/warpSL/ref",
    }
    created_directories = {}

    for key, directory in directories.items():
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            created_directories[key] = dir_path
            print(f"Created directory: {dir_path}")
        else:
            created_directories[key] = dir_path
            print(f"Directory already exists: {dir_path}")

    return created_directories

def save_images(image_data, folder_path, filename=None, overwrite=False, dsize=None):
    """
    This function saves images. The input should be image data read by OpenCV, typically in the 0-255 range.
    If overwrite = False and a file with the same name already exists, the new file will be saved as a new `img_000x_yyy.png`.
    If overwrite = True, the file will be overwritten using the specified file name or the default `img_0001.png`.
    You can also choose the output image size.

    @param image_data: Image data in OpenCV format, either a 3D or 4D array
    @param folder_path: Output folder path
    @param filename: Custom output file name (without extension). Default is None.
    @param overwrite: Whether to overwrite existing files. Default is False
    @param dsize: Target image size in the format (width, height). Default is None.
    @return: List of saved file names
    """
    # If the folder does not exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Determine whether the image data is 4D; if not, wrap it for unified processing
    is_four_dim = len(image_data.shape) == 4

    if not is_four_dim:
        image_data = image_data[None, ...]  # Wrap 3D data into 4D for easier processing

    # Check the data range; if it is within 0-1, multiply by 255
    if image_data.min() >= 0 and image_data.max() <= 1:
        image_data = (image_data * 255).astype('uint8')
    else:
        image_data = image_data.astype('uint8')

    saved_filenames = []

    for i in range(image_data.shape[0]):  # Iterate over each frame
        frame = image_data[i]

        # Resize the image if requested
        if dsize is not None:
            frame = cv2.resize(frame, dsize)

        # If a custom file name is provided
        if filename:
            name, ext = os.path.splitext(filename)
            if not ext:  # If the user did not provide an extension, default to png
                ext = '.png'
            new_filename = name + ext
        else:
            # If overwrite is True, or this is the first call, use the fixed starting name
            if overwrite or i == 0:
                new_filename = f'img_{i + 1:04d}.png'
            else:
                # Get the list of existing image files in the folder
                existing_files = [f for f in os.listdir(folder_path) if
                                  os.path.isfile(os.path.join(folder_path, f)) and f.startswith('img_') and f.endswith('.png')]

                if existing_files:
                    # Get the maximum existing file number
                    max_existing_num = max(int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f)
                    file_number = max_existing_num + 1
                    new_filename = f'img_{file_number:04d}_{i + 1:03d}.png'
                else:
                    new_filename = f'img_{i:04d}_{i + 1:03d}.png'

        # Build the full output path
        full_path = os.path.join(folder_path, new_filename)

        # Check whether an existing file should be overwritten
        if not overwrite and os.path.exists(full_path):
            base, ext = os.path.splitext(new_filename)
            j = 1
            while os.path.exists(os.path.join(folder_path, f"{base}_{j:03d}{ext}")):
                j += 1
            new_filename = f"{base}_{j:03d}{ext}"
            full_path = os.path.join(folder_path, new_filename)

        # Save the image
        cv2.imwrite(full_path, frame)
        saved_filenames.append(new_filename)
        # print(f"Saved image as: {new_filename}")
    return saved_filenames

def get_screen_size(screen_num):
    screens = []

    def callback(monitor, dc, rect, data):
        info = {}
        info['left'] = rect[0]
        info['top'] = rect[1]
        info['right'] = rect[2]
        info['bottom'] = rect[3]
        info['width'] = info['right'] - info['left']
        info['height'] = info['bottom'] - info['top']
        screens.append(info)
        return True

    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.c_long), ctypes.c_double)
    user32 = ctypes.windll.user32
    if not user32.EnumDisplayMonitors(0, 0, callback_type(callback), 0):
        raise RuntimeError("EnumDisplayMonitors failed")

    if screen_num < len(screens):
        screen = screens[screen_num]
        print(f"Resolution of screen {screen_num + 1}: {screen['width']} x {screen['height']}")
        return screen['width'], screen['height']

    else:
        raise ValueError(f"Screen number {screen_num} is out of range.")


def get_image_files(folder_path, extensions=['.png', '.jpg', '.jpeg']):
    """
    Get all image file paths in the specified folder that match the given extensions.

    Args:
    folder_path : Folder path
    extensions : List of file extensions, default is ['.png', '.jpg', '.jpeg']

    Returns:
    List of image file paths
    """
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    return image_files


def release_cv_window():
    cv2.destroyAllWindows()


def create_gray_pattern(w, h):
    # Python implementation of MATLAB's createGrayPattern; the final Gray code has four dimensions (n, h, w, 3)
    nbits = np.ceil(np.log2([w, h])).astype(int)  # Number of bits for vertical/horizontal patterns
    offset = (2 ** nbits - [w, h]) // 2  # Offset the binary pattern to make it symmetric

    # Coordinates to binary code
    c, r = np.meshgrid(np.arange(w), np.arange(h))
    bin_pattern = [np.unpackbits((c + offset[0])[..., None].view(np.uint8), axis=-1, bitorder='little', count=nbits[0])[..., ::-1],
                   np.unpackbits((r + offset[1])[..., None].view(np.uint8), axis=-1, bitorder='little', count=nbits[1])[..., ::-1]]

    # binary pattern to gray pattern
    gray_pattern = copy.deepcopy(bin_pattern)
    for n in range(len(bin_pattern)):
        for i in range(1, bin_pattern[n].shape[-1]):
            gray_pattern[n][:, :, i] = np.bitwise_xor(bin_pattern[n][:, :, i - 1], bin_pattern[n][:, :, i])

    # allPatterns also contains complementary patterns and all-0/all-1 patterns
    prj_patterns = np.zeros((h, w, 2 * sum(nbits) + 2), dtype=np.uint8)
    prj_patterns[:, :, 0] = 1  # All ones pattern

    # Vertical
    for i in range(gray_pattern[0].shape[-1]):
        prj_patterns[:, :, 2 * i + 2] = gray_pattern[0][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 3] = np.logical_not(gray_pattern[0][:, :, i]).astype(np.uint8)

    # Horizontal
    for i in range(gray_pattern[1].shape[-1]):
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 2] = gray_pattern[1][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 3] = np.logical_not(gray_pattern[1][:, :, i]).astype(np.uint8)

    prj_patterns *= 255

    # to RGB image
    # prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (0, 1, 3, 2))  # to (h, w, c, n)
    prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (2, 0, 1, 3))  # to (n, h, w, c)

    return prj_patterns

