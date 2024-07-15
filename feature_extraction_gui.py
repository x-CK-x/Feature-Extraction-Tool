import gradio as gr
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from scipy.stats import skew, kurtosis
from PIL import Image, UnidentifiedImageError, ImageFile
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize_image(image):
    """Normalize the image to the range [0, 1] if it is not already in this range."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image - image.min()) / (image.max() - image.min())
    return image

def fft_transform(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return normalize_image(magnitude_spectrum)

def lbp_transform(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return normalize_image(lbp)

def gabor_transform(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    frequency = 0.6
    gabor_response, _ = gabor(image, frequency=frequency)
    return normalize_image(gabor_response)

def statistical_features(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(image)
    variance_intensity = np.var(image)
    skewness_intensity = skew(image.ravel())
    kurtosis_intensity = kurtosis(image.ravel())
    return mean_intensity, variance_intensity, skewness_intensity, kurtosis_intensity

def process_image(image, transform_type):
    if transform_type == "FFT":
        return fft_transform(image)
    elif transform_type == "LBP":
        return lbp_transform(image)
    elif transform_type == "Gabor":
        return gabor_transform(image)
    elif transform_type == "Statistical":
        mean_intensity, variance_intensity, skewness_intensity, kurtosis_intensity = statistical_features(image)
        stats_image = np.zeros_like(image)
        stats_image.fill(255)
        cv2.putText(stats_image, f'Mean: {mean_intensity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        cv2.putText(stats_image, f'Variance: {variance_intensity:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        cv2.putText(stats_image, f'Skewness: {skewness_intensity:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        cv2.putText(stats_image, f'Kurtosis: {kurtosis_intensity:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        return stats_image

def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        return True
    except (UnidentifiedImageError, IOError, OSError):
        return False

def find_bad_files(input_folders):
    bad_files = []
    for input_folder in input_folders:
        for root, _, files in os.walk(input_folder):
            for image_file in files:
                if image_file.lower().endswith(('.jfif', '.jpeg', '.jpg', '.bmp', '.png', '.tif', '.webp')):
                    image_path = os.path.join(root, image_file)
                    if not is_image_file(image_path):
                        bad_files.append(image_path)
    return bad_files

def batch_process_single_folder(input_folder, output_folder, fft, lbp, gabor):
    # Create the base output folder structure
    base_folder_name = os.path.basename(os.path.normpath(input_folder))
    base_output_path = os.path.join(output_folder, base_folder_name)
    os.makedirs(base_output_path, exist_ok=True)
    
    # Iterate over all subdirectories and files
    for root, _, files in os.walk(input_folder):
        for image_file in files:
            if image_file.lower().endswith(('.jfif', '.jpeg', '.jpg', '.bmp', '.png', '.tif', '.webp')):
                image_path = os.path.join(root, image_file)
                if is_image_file(image_path):
                    try:
                        image = Image.open(image_path)
                        mean_intensity, variance_intensity, skewness_intensity, kurtosis_intensity = statistical_features(image)
                        
                        # Create the relative output path
                        relative_path = os.path.relpath(root, input_folder)
                        output_path = os.path.join(base_output_path, relative_path)
                        os.makedirs(output_path, exist_ok=True)
                        
                        # Save the statistical features to a text file
                        base_name = os.path.splitext(image_file)[0]
                        with open(os.path.join(output_path, f"{base_name}.txt"), 'w') as f:
                            f.write(f"Mean: {mean_intensity}\n")
                            f.write(f"Variance: {variance_intensity}\n")
                            f.write(f"Skewness: {skewness_intensity}\n")
                            f.write(f"Kurtosis: {kurtosis_intensity}\n")
                        
                        # Save transformed images if the respective checkbox is checked
                        if fft:
                            fft_image = fft_transform(image)
                            fft_image_path = os.path.join(output_path, f"{base_name}_fft.png")
                            plt.imsave(fft_image_path, fft_image, cmap='gray')

                        if lbp:
                            lbp_image = lbp_transform(image)
                            lbp_image_path = os.path.join(output_path, f"{base_name}_lbp.png")
                            plt.imsave(lbp_image_path, lbp_image, cmap='gray')

                        if gabor:
                            gabor_image = gabor_transform(image)
                            gabor_image_path = os.path.join(output_path, f"{base_name}_gabor.png")
                            plt.imsave(gabor_image_path, gabor_image, cmap='gray')
                    except (UnidentifiedImageError, IOError, OSError) as e:
                        print(f"Error processing file {image_path}: {e}")
                else:
                    print(f"Unsupported or corrupted image file: {image_path}")
    
    return f"Processed {input_folder}"

def batch_process(input_folders, output_folder, fft, lbp, gabor):
    # Find and display bad files before starting batch processing
    bad_files = find_bad_files(input_folders)
    if bad_files:
        print("Bad files found:")
        for bad_file in bad_files:
            print(f"- {bad_file}")
    else:
        print("No bad files found.")

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(batch_process_single_folder, folder, output_folder, fft, lbp, gabor) for folder in input_folders]
        for future in tqdm(futures, desc="Batch Processing"):
            results.append(future.result())
    return "\n".join(results)

with gr.Blocks() as demo:
    with gr.Tab("Single Image Processing"):
        gr.Markdown("## Single Image Processing")
        image_input = gr.Image(type="pil")
        transform_type = gr.Radio(choices=["FFT", "LBP", "Gabor", "Statistical"], label="Transform Type")
        image_output = gr.Image(type="numpy")
        process_button = gr.Button("Process")
        process_button.click(process_image, inputs=[image_input, transform_type], outputs=image_output)

    with gr.Tab("Batch Processing"):
        gr.Markdown("## Batch Processing")
        input_folders = gr.Dropdown(label="Input Folder Paths", multiselect=True, allow_custom_value=True)
        output_folder = gr.Textbox(label="Output Folder Path")
        fft_checkbox = gr.Checkbox(label="FFT")
        lbp_checkbox = gr.Checkbox(label="LBP")
        gabor_checkbox = gr.Checkbox(label="Gabor")
        run_button = gr.Button("Run")
        batch_output = gr.Textbox(label="Batch Processing Output", interactive=False)
        run_button.click(batch_process, inputs=[input_folders, output_folder, fft_checkbox, lbp_checkbox, gabor_checkbox], outputs=batch_output)

    with gr.Tab("Documentation"):
        gr.Markdown("""
        ## Feature Extraction Tool Documentation

        ### Single Image Processing
        This tool allows you to apply various feature extraction techniques to a single image. The available transformations are:
        - **FFT (Fast Fourier Transform)**: Transforms the image into the frequency domain to reveal frequency components.
        - **LBP (Local Binary Patterns)**: Extracts texture features by thresholding the neighborhood of each pixel.
        - **Gabor Filters**: Applies Gabor filters to extract texture features sensitive to specific frequencies and orientations.
        - **Statistical Features**: Computes statistical measures (mean, variance, skewness, kurtosis) of pixel intensities.

        ### Batch Processing
        The batch processing feature allows you to process multiple images in one or more folders concurrently. You can specify multiple input folders and an output folder where results will be saved.

        **Steps:**
        1. **Input Folder Paths**: Use the multi-select dropdown to add multiple input folder paths.
        2. **Output Folder Path**: Specify the path where the results will be saved.
        3. **Select Transformations**: Choose which transformations to apply (FFT, LBP, Gabor).
        4. **Run**: Click the "Run" button to start batch processing.

        **Output:**
        - **Text Files**: For each image, a text file containing the mean, variance, skewness, and kurtosis values will be saved in the output folder.
        - **Transformed Images**: If selected, transformed images will be saved in respective sub-folders (FFT, LBP, Gabor) within the output folder.

        ### Progress Indicators
        The tool uses progress bars to show the status of image processing tasks both for individual folders and overall batch processing.

        ### Usage Example
        - **Single Image Processing**: Upload an image, select a transformation, and click "Process" to view the transformed image.
        - **Batch Processing**: Add input folder paths, specify an output folder, select transformations, and click "Run" to process all images in the specified folders.

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

        ### Contact
        For any questions or issues, please contact [Your Contact Information].
        """)

demo.launch(share=False)
