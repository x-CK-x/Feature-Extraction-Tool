# Feature Extraction Tool

This tool allows you to perform various feature extraction techniques on images, either individually or in batches. It supports transformations such as FFT (Fast Fourier Transform), LBP (Local Binary Patterns), Gabor Filters, and Statistical Features extraction.

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

### Setting Up the Conda Environment

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/x-CK-x/Feature-Extraction-Tool.git
   cd Feature-Extraction-Tool
   ```
2. **Create the Conda Environment**:
   ```sh
   conda env create -f environment.yml
   ```
3. **Activate the Environment**:
   ```sh
   conda activate feature_extraction_env
   ```
### Running the Tool

1. **Navigate to the Script Directory**:
   ```sh
   cd <directory_where_script_is_located>
   ```
2. **Run the Script**:
   ```sh
   python feature_extraction_gui.py
   ```
3. **Access the Gradio Interface**:
   ```txt
   After running the script, a local URL will be provided. Open this URL in your web browser to access the Gradio interface.
   ```
### Features and Usage

#### Single Image Processing

This feature allows you to apply various feature extraction techniques to a single image.

1. **Upload an Image**: Use the image upload button to upload an image.
2. **Select a Transformation**: Choose from FFT, LBP, Gabor, or Statistical features using the radio buttons.
3. **Process the Image**: Click the "Process" button to apply the selected transformation and view the result.

#### Batch Processing

This feature allows you to process multiple images in one or more folders concurrently.

1. **Add Input Folder Paths**: Use the multi-select dropdown to add multiple input folder paths.
2. **Specify Output Folder Path**: Enter the path where the results will be saved.
3. **Select Transformations**: Choose which transformations to apply (FFT, LBP, Gabor).
4. **Run the Batch Processing**: Click the "Run" button to start batch processing.

**Output**:

- For each image, a text file containing the mean, variance, skewness, and kurtosis values will be saved in the output folder.
- Transformed images will be saved in respective sub-folders (FFT, LBP, Gabor) within the output folder.

**Documentation**

The Documentation tab provides detailed information about the tool's features and usage.

### Handling Corrupted or Unsupported Files

The tool scans the input folders for corrupted or unsupported image files before starting the batch processing. These files are displayed in the terminal and are ignored during the processing.

#### Supported Image Formats

- jfif
- jpeg
- jpg
- bmp
- png
- tif
- webp

-----------------------------------------

## Dataset Pruning Tool

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


