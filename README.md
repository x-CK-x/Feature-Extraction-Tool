# Feature Extraction Tool

This tool allows you to perform various feature extraction techniques on images, either individually or in batches. It supports transformations such as FFT (Fast Fourier Transform), LBP (Local Binary Patterns), Gabor Filters, and Statistical Features extraction.

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

### Setting Up the Conda Environment

1. **Clone the Repository**:
   ```sh
   git clone <repository_url>
   cd <repository_directory>
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

    **Upload an Image**: Use the image upload button to upload an image.
    **Select a Transformation**: Choose from FFT, LBP, Gabor, or Statistical features using the radio buttons.
    **Process the Image**: Click the "Process" button to apply the selected transformation and view the result.

#### Batch Processing

This feature allows you to process multiple images in one or more folders concurrently.

    **Add Input Folder Paths**: Use the multi-select dropdown to add multiple input folder paths.
    **Specify Output Folder Path**: Enter the path where the results will be saved.
    **Select Transformations**: Choose which transformations to apply (FFT, LBP, Gabor).
    **Run the Batch Processing**: Click the "Run" button to start batch processing.

**Output**:

    For each image, a text file containing the mean, variance, skewness, and kurtosis values will be saved in the output folder.
    Transformed images will be saved in respective sub-folders (FFT, LBP, Gabor) within the output folder.

Documentation

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



