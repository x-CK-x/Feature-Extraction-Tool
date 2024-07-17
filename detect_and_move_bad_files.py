import gradio as gr
import os
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Increase the decompression bomb error threshold
Image.MAX_IMAGE_PIXELS = None

# Define the compression level of the supported file types
compression_levels = {
    '.bmp': 1,
    '.png': 2,
    '.tif': 3,
    '.jpeg': 4,
    '.jpg': 4,
    '.jfif': 5,
    '.webp': 6
}

def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact, an image
            # Try to get EXIF data (this can fail for some corrupted images)
            img._getexif()
        return True
    except (UnidentifiedImageError, IOError, OSError, Image.DecompressionBombError, AttributeError) as e:
        if isinstance(e, AttributeError) and "_getexif" in str(e):
            print(f"EXIF error in file: {file_path}")
        return False

def hash_image(image_path):
    """Compute the SHA-256 hash of the image file."""
    with Image.open(image_path) as img:
        return hashlib.sha256(img.tobytes()).hexdigest()

def find_and_move_bad_files(input_folders, output_folder, num_cpus):
    bad_files = []
    total_files = sum(len(files) for input_folder in input_folders for _, _, files in os.walk(input_folder))

    def process_file(root, image_file, pbar):
        image_path = os.path.join(root, image_file)
        if not is_image_file(image_path):
            bad_files.append(image_path)
            print(f"Moving bad file: {image_path}")
            # Create the relative output path
            relative_path = os.path.relpath(root, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            os.makedirs(output_path, exist_ok=True)
            # Move the bad file
            shutil.move(image_path, os.path.join(output_path, image_file))
        pbar.update(1)

    with tqdm(total=total_files, desc="Processing images") as pbar:
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            for input_folder in input_folders:
                for root, _, files in os.walk(input_folder):
                    for image_file in files:
                        if image_file.lower().endswith(('.jfif', '.jpeg', '.jpg', '.bmp', '.png', '.tif', '.webp')):
                            executor.submit(process_file, root, image_file, pbar)

    return bad_files

def find_and_move_duplicates(input_folders, output_folder, num_cpus):
    image_hashes = {}
    total_files = sum(len(files) for input_folder in input_folders for _, _, files in os.walk(input_folder))

    def process_file(root, image_file, pbar):
        image_path = os.path.join(root, image_file)
        try:
            image_hash = hash_image(image_path)
            extension = os.path.splitext(image_file)[1].lower()
            if image_hash in image_hashes:
                existing_path, existing_extension = image_hashes[image_hash]
                if compression_levels[extension] < compression_levels[existing_extension]:
                    # Move the current file to duplicates and keep the existing file
                    move_to_duplicates(image_path, root, output_folder, image_file)
                else:
                    # Move the existing file to duplicates and keep the current file
                    move_to_duplicates(existing_path, os.path.dirname(existing_path), output_folder, os.path.basename(existing_path))
                    image_hashes[image_hash] = (image_path, extension)
            else:
                image_hashes[image_hash] = (image_path, extension)
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
        pbar.update(1)

    def move_to_duplicates(image_path, root, output_folder, image_file):
        relative_path = os.path.relpath(root, input_folder)
        output_path = os.path.join(output_folder, relative_path, 'duplicates')
        os.makedirs(output_path, exist_ok=True)
        shutil.move(image_path, os.path.join(output_path, image_file))
        print(f"Moved duplicate file: {image_path}")

    with tqdm(total=total_files, desc="Detecting duplicates") as pbar:
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            for input_folder in input_folders:
                for root, _, files in os.walk(input_folder):
                    for image_file in files:
                        if image_file.lower().endswith(('.jfif', '.jpeg', '.jpg', '.bmp', '.png', '.tif', '.webp')):
                            executor.submit(process_file, root, image_file, pbar)

    return [path for path, _ in image_hashes.values()]

def detect_move_and_deduplicate(input_folders, output_folder, num_cpus, mode):
    if mode == "Detect and Move Bad Files":
        bad_files = find_and_move_bad_files(input_folders, output_folder, num_cpus)
        if bad_files:
            return f"Bad files moved to {output_folder}:\n" + "\n".join(bad_files)
        else:
            return "No bad files found."
    elif mode == "Detect and Move Duplicates":
        duplicate_files = find_and_move_duplicates(input_folders, output_folder, num_cpus)
        if duplicate_files:
            return f"Duplicate files moved to {output_folder}:\n" + "\n".join(duplicate_files)
        else:
            return "No duplicate files found."

with gr.Blocks() as demo:
    with gr.Tab("Detect and Move Bad Files"):
        gr.Markdown("## Detect and Move Corrupted/Unsupported Image Files")
        input_folders = gr.Dropdown(label="Input Folder Paths", multiselect=True, allow_custom_value=True)
        output_folder = gr.Textbox(label="Output Folder Path")
        num_cpus = gr.Slider(label="Number of CPUs", minimum=1, maximum=os.cpu_count(), step=1, value=os.cpu_count()//2)
        mode = gr.Radio(label="Mode", choices=["Detect and Move Bad Files", "Detect and Move Duplicates"], value="Detect and Move Bad Files")
        run_button = gr.Button("Run")
        output_text = gr.Textbox(label="Output", interactive=False)
        run_button.click(detect_move_and_deduplicate, inputs=[input_folders, output_folder, num_cpus, mode], outputs=output_text)

    with gr.Tab("Documentation"):
        gr.Markdown("""
        ### Overview
        This tool allows you to manage your image dataset by detecting and moving corrupted/unsupported images, as well as identifying and handling duplicate images.

        ### Tabs
        - **Detect and Move Bad Files**: Detects corrupted or unsupported image files and moves them to a specified output folder while preserving the directory structure.
        - **Detect and Move Duplicates**: Identifies duplicate images across multiple input folders and moves them to a 'duplicates' folder within the mirrored directory structure in the output folder.

        ### How to Use
        1. **Select Input Folders**: Use the dropdown menu to add multiple input folder paths.
        2. **Specify Output Folder**: Enter the path where the results will be saved.
        3. **Choose Number of CPUs**: Use the slider to select the number of CPUs for concurrent processing.
        4. **Select Mode**: Choose between "Detect and Move Bad Files" and "Detect and Move Duplicates".
        5. **Run the Process**: Click the "Run" button to start the processing.

        ### Modes
        - **Detect and Move Bad Files**:
          - Scans the selected input folders for corrupted or unsupported images.
          - Moves identified bad files to the output folder, preserving the directory structure.

        - **Detect and Move Duplicates**:
          - Scans the selected input folders for duplicate images using SHA-256 hashing.
          - Moves duplicate images to a 'duplicates' folder within the mirrored directory structure in the output folder.
          - Keeps the less compressed image format if duplicates with different file types are found.

        ### Supported Image Formats
        - jfif
        - jpeg
        - jpg
        - bmp
        - png
        - tif
        - webp

        ### Requirements
        Ensure the following libraries are installed:
        - `gradio`
        - `numpy`
        - `opencv-python-headless`
        - `matplotlib`
        - `scikit-image`
        - `scipy`
        - `pillow`
        - `tqdm`
        """)

demo.launch(share=False)
