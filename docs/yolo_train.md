# YOLO Pose Estimation Training in SimBA

SimBA has a graphical user interface (GUI) for training YOLO-based pose estimation models. This GUI allows users to train custom YOLO pose estimation models on their own annotated datasets, which can then be used for pose estimation inference on videos.

<img width="756" height="201" alt="image" src="https://github.com/user-attachments/assets/0659b291-7742-4422-8670-a08450c6353a" />

## Table of Contents

- [Prerequisites](#prerequisites)
- [Accessing the YOLO Pose Training Popup](#accessing-the-yolo-pose-training-popup)
- [Interface Overview](#interface-overview)
- [Usage Instructions](#usage-instructions)
- [Output](#output)
- [Important Notes](#important-notes)
- [Troubleshooting](#troubleshooting)
- [Converting Other Data Formats to YOLO Format](#converting-other-data-formats-to-yolo-format)
- [Related Documentation](#related-documentation)

<a id="prerequisites"></a>
## Prerequisites

Before using the YOLO Pose Training popup, ensure that:

1. **NVIDIA GPU**: An NVIDIA GPU with CUDA support is highly recommended. While the popup may not strictly require it, training without a GPU is extremely slow and not practical for most use cases.
2. **Ultralytics Package**: The `ultralytics` package must be installed in your Python environment (I recommend version 8.3.156).
3. **YOLO Map File (YAML)**: You need a YAML configuration file that defines your dataset structure, including paths to training/validation images and labels, keypoint shape, and class names. Example YAML files can be found in the [SimBA repository](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model_keypoints.yaml).
4. **Initial Weights File**: You need a pre-trained YOLO pose estimation model weights file (typically a `.pt` file) to use as a starting point. Pre-trained weights can be downloaded from [HuggingFace](https://huggingface.co/Ultralytics) (e.g., `yolo11n-pose.pt`, `yolo11s-pose.pt`, `yolo11m-pose.pt`, `yolo11l-pose.pt`, `yolo11x-pose.pt`).
5. **Annotated Dataset**: You need a properly formatted YOLO dataset with images and corresponding label files. The dataset should be organized with separate directories for training and validation images and labels.

<a id="accessing-the-yolo-pose-training-popup"></a>
## Accessing the YOLO Pose Training Popup

The YOLO training menu can be accessed in the SimBA main interface. The popup window is titled **"TRAIN YOLO POSE ESTIMATION MODEL"** and provides an interface for configuring and running YOLO pose estimation model training.

<a id="interface-overview"></a>
## Interface Overview

<img width="885" height="397" alt="image" src="https://github.com/user-attachments/assets/bd5ce66e-0131-4e25-a0b0-18713064dd8f" />


The popup window is organized into a single **SETTINGS** section that contains all configuration options:

### SETTINGS

This section contains all the parameters needed to configure YOLO model training. Understanding the effects of each setting will help you optimize training for your specific dataset and hardware:

- **YOLO MAP FILE (YAML)**: Path to the YAML configuration file that defines your dataset structure. This file must specify:
  - `path`: Base directory path for your dataset
  - `train`: Relative path to training images directory
  - `val`: Relative path to validation images directory
  - `kpt_shape`: Keypoint shape as `[num_keypoints, 3]` where 3 represents (x, y, visibility)
  - `flip_idx`: Indices for horizontal flip augmentation (defines which keypoints are symmetric pairs)
  - `names`: Dictionary mapping class IDs to class names (e.g., `{0: 'mouse'}`)
  - **Effect**: This file tells YOLO where to find your training data and how to interpret it. The YAML must be correctly formatted and all paths must be valid. Incorrect YAML files will cause training to fail. See example YAML files in the SimBA repository for reference.

- **INITIAL WEIGHT FILE (E.G., .PT)**: Path to pre-trained YOLO pose estimation model weights file (e.g., `yolo11n-pose.pt`).
  - **Effect**: This serves as the starting point for training (transfer learning). Using pre-trained weights significantly speeds up training and improves final model performance compared to training from scratch. Larger models (e.g., `yolo11x-pose.pt`) provide better accuracy but require more GPU memory and training time. Smaller models (e.g., `yolo11n-pose.pt`) train faster and use less memory but may have lower accuracy. **Recommendation**: Start with `yolo11n-pose.pt` or `yolo11s-pose.pt` for faster iteration, use larger models for production if you have sufficient GPU memory.

- **SAVE DIRECTORY**: Directory where training outputs will be saved, including:
  - Trained model weights (`best.pt`, `last.pt`)
  - Training metrics and plots
  - Training logs
  - **Effect**: All training artifacts are saved here. Make sure you have sufficient disk space, as training can generate several GB of data, especially with plots enabled and long training runs.

- **EPOCHS**: Number of training epochs to run. Options range from 100 to 5500 in increments of 250. Default is 500.
  - **Effect**: More epochs allow the model to learn more from your data, potentially improving accuracy, but increase training time. Too many epochs can lead to overfitting (model memorizes training data but performs poorly on new data). Too few epochs may result in underfitting (model hasn't learned enough). **Trade-off**: More epochs = potentially better accuracy but longer training time and risk of overfitting. Start with 500-1000 epochs, monitor validation metrics, and use early stopping (PATIENCE) to prevent overfitting.

- **BATCH SIZE**: Number of images processed in each training batch. Options: 2, 4, 8, 16, 32, 64, 128. Default is 16.
  - **Effect**: Larger batch sizes (32-128) can lead to more stable training and faster training per epoch, but require significantly more GPU memory. Smaller batch sizes (2-8) use less memory but may result in noisier gradients and slower training. Very small batches (2-4) may cause training instability. **Trade-off**: Larger batch size = faster and more stable training but higher memory usage. Start with 16-32. If you get "out of memory" errors, reduce to 8 or 4. If you have a high-end GPU with lots of memory, try 32-64 for faster training.

- **IMAGE SIZE**: Input image size for training (must be square). Options: 256, 320, 416, 480, 512, 640, 720, 768, 960, 1280. Default is 640.
  - **Effect**: Larger image sizes (640-1280) provide more detail to the model, which can improve keypoint detection accuracy, especially for small animals or fine details, but significantly increase training time and GPU memory usage. Smaller image sizes (256-416) train faster and use less memory but may reduce accuracy for small keypoints. **Trade-off**: Higher resolution = better accuracy but slower training and higher memory usage. Use 640 (default) as a good starting point. Increase to 960-1280 if tracking small animals or fine details. Decrease to 416-512 for faster training or limited GPU memory.

- **PATIENCE**: Early stopping patience in epochs. If validation metrics don't improve for this many consecutive epochs, training stops early. Options range from 50 to 1000 in increments of 50. Default is 100.
  - **Effect**: Higher patience values (200-500) allow training to continue longer even if metrics plateau temporarily, which can help find better models but wastes time if the model has truly stopped improving. Lower patience values (50-100) stop training sooner when progress stalls, saving time but potentially stopping before the model reaches optimal performance. **Trade-off**: Higher patience = more training time but potentially better results vs. lower patience = faster training but may stop too early. Default of 100 is a good balance. Increase to 200-300 if you have time and want to ensure optimal performance. Decrease to 50 if you want faster iteration.

- **PLOTS**: Whether to generate training plots (loss curves, metrics, etc.). Options: TRUE (default) or FALSE.
  - **Effect**: When TRUE, Ultralytics generates comprehensive training visualization plots showing training/validation loss, precision, recall, and other metrics over epochs. These plots are saved in the save directory and are extremely useful for monitoring training progress, detecting overfitting, and diagnosing issues. When FALSE, no plots are generated, saving some disk space and slightly reducing training overhead. **Recommendation**: Keep TRUE (default) to monitor training progress and diagnose issues. The plots are essential for understanding if your model is training correctly.

- **VERBOSE**: Whether to output detailed progress information during training. Options: TRUE (default) or FALSE.
  - **Effect**: When TRUE, you'll see detailed progress updates, epoch-by-epoch metrics, timing information, and any warnings in the terminal/console. This is essential for monitoring training progress and debugging issues. When FALSE, only essential messages are shown. **Recommendation**: Keep TRUE (default) to monitor training progress, especially for long training runs.

- **CPU WORKERS**: Number of data loader worker processes for loading and preprocessing images. Options range from 1 to the maximum number of CPU cores available. Default is approximately half of the maximum cores.
  - **Effect**: More CPU workers can speed up data loading and preprocessing, preventing the GPU from waiting for data, which can significantly speed up training. However, too many workers can cause CPU overload and actually slow things down. Fewer workers use less CPU resources but may cause the GPU to wait for data, reducing training efficiency. **Trade-off**: More workers = faster data loading but higher CPU usage. Default (half of cores) is usually optimal. Increase to 75-100% of cores if you have many CPU cores and want maximum speed. Decrease if you need to reserve CPU for other tasks.

- **FORMAT**: Optional export format for the trained model. Options include: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite". Default is "engine".
  - **Effect**: Setting a format will export the trained model to that format after training completes. Different formats are optimized for different deployment scenarios (e.g., "onnx" for cross-platform deployment, "engine" for TensorRT optimization, "tflite" for mobile devices). Exporting adds time to the training process but provides models ready for specific deployment environments. **Use case**: Use "engine" (default) for TensorRT-optimized inference, "onnx" for general cross-platform deployment, or other formats based on your deployment needs. The original PyTorch `.pt` weights are always saved regardless of this setting.

- **DEVICE**: Select the device for training. Options include "CPU" or specific GPU devices (e.g., "0 : NVIDIA GeForce RTX 3090"). The default is typically the first available GPU.
  - **Effect**: GPU training (default) is essential for practical training times - training on CPU can take days or weeks for even small datasets. Use a specific GPU number if you have multiple GPUs and want to use a particular one. **Recommendation**: Always use GPU unless you have no other option. If you have multiple GPUs, select the one with the most memory or the one not being used by other processes.

<a id="usage-instructions"></a>
## Usage Instructions

### Step 1: Prepare Your Dataset

Before using the training popup, ensure you have:

1. **Annotated Dataset**: A YOLO-formatted dataset with:
   - Training images in one directory
   - Training labels (`.txt` files) in a corresponding directory
   - Validation images in a separate directory
   - Validation labels (`.txt` files) in a corresponding directory

2. **YOLO Map File (YAML)**: Create a YAML file that describes your dataset structure. See example files in the SimBA repository for the required format.

3. **Initial Weights**: Download a pre-trained YOLO pose estimation model from [HuggingFace](https://huggingface.co/Ultralytics).

### Step 2: Configure Training Parameters

1. Click on **YOLO MAP FILE (YAML)** and browse to select your dataset configuration YAML file.
2. Click on **INITIAL WEIGHT FILE (E.G., .PT)** and select your pre-trained weights file.
3. Click on **SAVE DIRECTORY** and select the folder where training outputs should be saved.
4. Adjust training parameters in the **SETTINGS** section:
   - Set **EPOCHS** based on your dataset size and training time budget
   - Set **BATCH SIZE** based on your GPU memory (start with 16, adjust if you get memory errors)
   - Set **IMAGE SIZE** based on your needs (640 is a good default)
   - Set **PATIENCE** for early stopping (100 is a good default)
   - Keep **PLOTS** and **VERBOSE** as TRUE to monitor training
   - Adjust **CPU WORKERS** based on your CPU cores
   - Select **FORMAT** based on your deployment needs
   - Select **DEVICE** (use GPU)

### Step 3: Start Training

Click the **RUN** button to start training. The training process will:

1. Validate all input paths and files
2. Load the initial weights
3. Load and validate the dataset
4. Begin training, showing progress in the terminal/console
5. Save model checkpoints, metrics, and plots to the save directory

### Step 4: Monitor Training

During training, monitor:

- **Terminal Output**: If **VERBOSE** is TRUE, you'll see epoch-by-epoch progress, loss values, and metrics
- **Training Plots**: If **PLOTS** is TRUE, check the plots in the save directory to monitor:
  - Training and validation loss (should decrease over time)
  - Precision and recall metrics
  - Any signs of overfitting (validation loss increasing while training loss decreases)

<a id="output"></a>
## Output

Training outputs are saved in the specified **SAVE DIRECTORY** and include:

- **`best.pt`**: Model weights from the epoch with the best validation metrics
- **`last.pt`**: Model weights from the final training epoch
- **Training plots**: Visualizations of training metrics (if PLOTS is TRUE)
- **Training logs**: Detailed logs of the training process
- **Exported model**: Model in the specified format (if FORMAT is set)

The `best.pt` file is typically the model you want to use for inference, as it represents the best performance on the validation set.

<a id="important-notes"></a>
## Important Notes

1. **GPU Requirement**: While not strictly enforced, GPU training is essential for practical training times. CPU training is extremely slow and not recommended.

2. **Dataset Quality**: The quality and quantity of your training data directly affects model performance. Ensure you have:
   - Sufficient training examples (typically hundreds to thousands of images)
   - Diverse examples covering different poses, lighting conditions, and scenarios
   - Accurate annotations
   - A proper train/validation split (typically 70-80% train, 20-30% validation)

3. **Training Time**: Training can take hours to days depending on:
   - Dataset size
   - Number of epochs
   - Image size
   - Batch size
   - GPU performance

4. **Memory Considerations**: Larger batch sizes and image sizes require more GPU memory. If you encounter "out of memory" errors:
   - Reduce **BATCH SIZE** (try 8, 4, or even 2)
   - Reduce **IMAGE SIZE** (try 512, 416, or 320)
   - Close other applications using GPU memory

5. **Overfitting**: Monitor training plots for signs of overfitting (validation loss increasing while training loss decreases). If this occurs:
   - Reduce the number of **EPOCHS**
   - Lower **PATIENCE** to stop training earlier
   - Increase your dataset size or use data augmentation
   - Use a smaller model (e.g., `yolo11n-pose.pt` instead of `yolo11x-pose.pt`)

6. **Early Stopping**: The **PATIENCE** parameter implements early stopping to prevent overfitting. Training will automatically stop if validation metrics don't improve for the specified number of epochs.

7. **Model Selection**: After training, use `best.pt` (not `last.pt`) for inference, as it represents the best validation performance.

<a id="troubleshooting"></a>
## Troubleshooting

- **"No GPU detected"**: While training may proceed on CPU, it will be extremely slow. Ensure you have an NVIDIA GPU with CUDA drivers installed. Check GPU availability using `nvidia-smi` in your terminal.

- **"Could not find ultralytics package"**: Install the ultralytics package using `pip install ultralytics` or `conda install ultralytics` (I've been using version 8.3.156 while writing these docs).

- **"Invalid YOLO map file"**: Verify that your YAML file is correctly formatted and all paths are valid. Check that:
  - All required fields are present (`path`, `train`, `val`, `kpt_shape`, `flip_idx`, `names`)
  - All file paths exist and are accessible
  - The YAML syntax is correct

- **"File not found errors"**: Verify that:
  - The initial weights file exists and is readable
  - The YOLO map file exists and is readable
  - All dataset paths specified in the YAML file exist
  - Training and validation image/label directories exist and contain files

- **Out of memory errors**: Reduce the **BATCH SIZE** or **IMAGE SIZE** settings to use less GPU memory. Start with batch size 8 or 4, and image size 512 or 416.

- **Training is very slow**: 
  - Ensure you're using a GPU (not CPU)
  - Increase **BATCH SIZE** if you have GPU memory available
  - Reduce **IMAGE SIZE** if acceptable for your use case
  - Increase **CPU WORKERS** to speed up data loading

- **Poor model performance**: 
  - Check that your dataset is large enough and diverse
  - Verify annotation quality
  - Try training for more epochs
  - Try a larger model (e.g., `yolo11m-pose.pt` instead of `yolo11n-pose.pt`)
  - Increase **IMAGE SIZE** if tracking small animals or fine details

- **Overfitting (validation loss increases while training loss decreases)**:
  - Reduce **EPOCHS** or lower **PATIENCE**
  - Increase dataset size or diversity
  - Use a smaller model
  - Check training plots to identify when overfitting begins

<a id="converting-other-data-formats-to-yolo-format"></a>
## Converting Other Data Formats to YOLO Format

Before training a YOLO pose estimation model, you need to convert your annotations to YOLO format. SimBA provides several conversion tools accessible through the main interface under the **Convert pose file format** menu. These tools convert various annotation formats into YOLO keypoint format, which can then be used for training.

### Available Conversion Tools

The following conversion tools are available in SimBA for converting pose estimation annotations to YOLO format:

#### 1. **DeepLabCut (DLC) to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `DLC annotations -> YOLO pose-estimation annotations`
- **Class**: `DLCYoloKeypointsPopUp` (`simba.ui.pop_ups.dlc_to_yolo_keypoints_popup.DLCYoloKeypointsPopUp`)
- **Description**: Converts DeepLabCut annotation files (CSV format with `CollectedData` in filename) to YOLO keypoint format.
- **Input**: Directory containing DLC annotation CSV files
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Supports multi-animal tracking, padding, greyscale conversion, CLAHE enhancement, and train/validation split

#### 2. **SimBA to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `SimBA -> YOLO pose-estimation annotations`
- **Class**: `SimBA2YoloKeypointsPopUp` (`simba.ui.pop_ups.simba_to_yolo_keypoints_popup.SimBA2YoloKeypointsPopUp`)
- **Description**: Converts existing SimBA project pose estimation data to YOLO keypoint format.
- **Input**: SimBA project configuration file (`.ini`)
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Extracts frames from SimBA project videos, supports frame sampling, confidence thresholding, padding, greyscale conversion, CLAHE enhancement, and train/validation split
- **Use case**: Ideal when you already have pose estimation data in a SimBA project and want to train a YOLO model on it

#### 3. **SLEAP Annotations to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `SLEAP SLP annotations -> YOLO pose-estimation annotations`
- **Class**: `SLEAPAnnotations2YoloPopUp` (`simba.ui.pop_ups.sleap_annotations_to_yolo_popup.SLEAPAnnotations2YoloPopUp`)
- **Description**: Converts SLEAP annotation files (`.slp` format) to YOLO keypoint format.
- **Input**: Directory containing SLEAP `.slp` annotation files
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Supports train/validation split, greyscale conversion, padding, CLAHE enhancement

#### 4. **SLEAP H5 Inference to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `SLEAP H5 inference -> YOLO pose-estimation annotations`
- **Class**: `SLEAPH5Inference2YoloPopUp` (`simba.ui.pop_ups.sleap_h5_inference_to_yolo_popup.SLEAPH5Inference2YoloPopUp`)
- **Description**: Converts SLEAP H5 inference/prediction files to YOLO keypoint format for training.
- **Input**: Directory containing SLEAP H5 files and corresponding video directory
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Extracts frames from videos, supports frame sampling, threshold filtering, train/validation split, greyscale conversion

#### 5. **SLEAP CSV Predictions to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `SLEAP CSV inference -> YOLO pose-estimation annotations`
- **Class**: `SLEAPcsvInference2Yolo` (`simba.ui.pop_ups.sleap_csv_predictions_to_yolo_popup.SLEAPcsvInference2Yolo`)
- **Description**: Converts SLEAP CSV prediction files to YOLO keypoint format.
- **Input**: SLEAP CSV prediction files and corresponding video directory
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Similar to SLEAP H5 conversion but works with CSV format predictions

#### 6. **DLC H5 Inference to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `DLC H5 inference -> YOLO pose-estimation annotations`
- **Class**: `DLCH5Inference2YoloPopUp` (`simba.ui.pop_ups.dlc_h5_inference_to_yolo_popup.DLCH5Inference2YoloPopUp`)
- **Description**: Converts DeepLabCut H5 inference/prediction files to YOLO keypoint format for training.
- **Input**: Directory containing DLC H5 files and corresponding video directory
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Extracts frames from videos, supports frame sampling, threshold filtering, train/validation split

#### 7. **COCO Keypoints to YOLO Keypoints**
- **Menu**: `Convert pose file format` → `COCO keypoints -> YOLO keypoints` (if available)
- **Class**: `COCOKeypoints2YOLOkeypointsPopUp` (`simba.ui.pop_ups.coco_keypoints_to_yolo_popup.COCOKeypoints2YOLOkeypointsPopUp`)
- **Description**: Converts COCO format keypoint annotations (JSON) to YOLO keypoint format.
- **Input**: COCO JSON annotation file and corresponding image directory
- **Output**: YOLO-formatted dataset with images, labels, and YAML map file
- **Features**: Supports train/validation split, greyscale conversion, CLAHE enhancement

### General Conversion Workflow

1. **Prepare Your Data**: Ensure your source annotations are in one of the supported formats (DLC, SimBA, SLEAP, COCO, etc.)

2. **Run Conversion Tool**: 
   - Open SimBA main interface
   - Navigate to `Convert pose file format` menu
   - Select the appropriate conversion tool for your data format
   - Configure settings (train/validation split, image processing options, etc.)
   - Run the conversion

3. **Verify Output**: After conversion, verify that:
   - Images are saved in `images/train/` and `images/val/` directories
   - Labels are saved in `labels/train/` and `labels/val/` directories
   - A YAML map file (`map.yaml`) is created in the save directory

4. **Use for Training**: The converted dataset can now be used with the YOLO training popup:
   - The YAML map file (`map.yaml`) is what you'll use as the **YOLO MAP FILE (YAML)** in the training popup
   - Ensure the paths in the YAML file are correct relative to where you'll run training

### Notes on Conversion

- **Train/Validation Split**: Most conversion tools allow you to specify the train/validation split ratio (typically 70-80% training, 20-30% validation). This split is important for proper model evaluation during training.

- **Image Processing Options**: Many conversion tools offer options for:
  - **Greyscale conversion**: Converts images to grayscale (may reduce file size and training time)
  - **CLAHE enhancement**: Applies Contrast Limited Adaptive Histogram Equalization to improve image quality
  - **Padding**: Adds padding around keypoints (useful for ensuring keypoints aren't too close to image edges)

- **Frame Sampling**: For video-based conversions (SimBA, SLEAP H5, DLC H5), you can specify how many frames to sample per video. This is useful for creating manageable dataset sizes from large video collections.

- **Multi-Animal Support**: Several conversion tools support multi-animal tracking scenarios. Ensure you specify the correct number of animals when converting.

- **YAML Map File**: The conversion process automatically generates a YAML map file that defines the dataset structure. This file is required for training and includes:
  - Dataset paths (train/validation image directories)
  - Keypoint shape information
  - Flip indices for data augmentation
  - Class names mapping

<a id="related-documentation"></a>
## Related Documentation

- For running inference with trained YOLO pose estimation models, see the [YOLO pose inference documentation](https://github.com/sgoldenlab/simba/blob/master/docs/yolo_pose_inference_popup.md).
- For importing YOLO pose estimation results into SimBA, see pose estimation import tools.
- Example YOLO map files: [Keypoints example](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model_keypoints.yaml), [General example](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml).
- For programmatic access to conversion functions, see:
  - `simba.third_party_label_appenders.transform.dlc_to_yolo.DLC2Yolo` - DLC to YOLO conversion
  - `simba.third_party_label_appenders.transform.simba_to_yolo.SimBA2Yolo` - SimBA to YOLO conversion
  - `simba.third_party_label_appenders.transform.sleap_to_yolo.SleapAnnotations2Yolo` - SLEAP to YOLO conversion
  - `simba.third_party_label_appenders.transform.coco_keypoints_to_yolo.COCKeypoints2Yolo` - COCO to YOLO conversion

##
Author [Simon N](https://github.com/sronilsson)

