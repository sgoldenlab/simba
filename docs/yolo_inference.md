# YOLO Pose Estimation Inference in SimBA

SimBA has graphical user interface (GUI) for running YOLO-based pose estimation inference on videos. This GUI allows users to apply pre-trained YOLO pose estimation models to analyze videos and extract keypoint tracking data.

## Prerequisites

Before using the YOLO Pose Inference popup, ensure that:

1. **NVIDIA GPU**: An NVIDIA GPU with CUDA support is required. The popup will raise an error if no NVIDIA GPUs are detected.
2. **Ultralytics Package**: The `ultralytics` package must be installed in your Python environment. The popup will verify this requirement on initialization (I recommend version 8.3.156).
3. **Trained YOLO Model**: You need a trained YOLO pose estimation model file (typically a `.pt` file).

## Accessing the YOLO Pose Inference Popup

The YOLO inference menu can be accessed in SimBA main interface. The popup window is titled **"PREDICT USING YOLO POSE ESTIMATION MODEL"** and provides a interface for configuring and running pose estimation inference.

<img width="753" height="219" alt="image" src="https://github.com/user-attachments/assets/0a8c66b8-736e-4e45-a0f3-329c4de3a714" />


## Interface Overview
<img width="1043" height="698" alt="image" src="https://github.com/user-attachments/assets/208be00e-f2b1-44d1-a513-1fb3250fee83" />


The popup window is organized into several sections:

### 1. MODEL & DATA PATHS

This section contains file and folder selection options:

- **MODEL PATH (E.G., .PT)**: Select the path to your trained YOLO model weights file. Supported formats include all YOLO model file types (typically `.pt` files).
- **SAVE DIRECTORY**: Choose the directory where the YOLO tracking results will be saved.
- **BODY-PART NAMES (.CSV)**: Select a CSV file containing the names of the body parts/keypoints that your model tracks. By default, this points to a 7-body-part configuration file [(`yolo_7bps.csv`)](https://github.com/sgoldenlab/simba/blob/master/simba/assets/lookups/yolo_schematics/yolo_7bps.csv) located in the YOLO schematics directory.
- **TRACKER CONFIG (OPTIONAL, .YML)**: Optionally select a YAML configuration file for advanced tracking settings. If not provided, standard YOLO pose inference will be used. For exampled, see teh [tracker_yaml](https://github.com/sgoldenlab/simba/tree/master/simba/assets/tracker_yml) directory in SimBA.

### 2. SETTINGS

This section contains various inference parameters that control how the YOLO model processes videos. Understanding the effects of each setting will help you optimize inference for your specific use case:

- **FORMAT**: Select an optional export format for the model. Options include: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite", or "None" (default). 
  - **Effect**: Setting a format will export the model to that format before inference, which can be useful for deployment or optimization. "None" (default) uses the original PyTorch model directly. Exporting to formats like "onnx" or "engine" may provide faster inference speeds but requires additional conversion time upfront. Use "None" unless you specifically need a different format for deployment.

- **BATCH SIZE**: Number of frames to process in parallel. Options range from 50 to 1000 in increments of 50. Default is 250.
  - **Effect**: Larger batch sizes (e.g., 500-1000) process more frames simultaneously, which can significantly speed up inference on shorter videos but requires more GPU memory. Smaller batch sizes (e.g., 50-200) use less memory and are safer for long videos or limited GPU memory. **Trade-off**: Higher batch size = faster processing but higher memory usage. If you encounter "out of memory" errors, reduce this value. For videos with many animals or high resolution, start with smaller batch sizes (100-200).

- **IMAGE SIZE**: Input image size for inference (must be square). Options: 256, 288, 320, 416, 480, 512, 640, 720, 768, 960, 1280. Default is 256.
  - **Effect**: Larger image sizes (e.g., 640-1280) provide higher resolution input to the model, which can improve detection accuracy for small animals or fine details, but significantly increases processing time and GPU memory usage. Smaller image sizes (e.g., 256-320) are faster and use less memory but may miss small animals or fine keypoint details. **Trade-off**: Higher resolution = better accuracy but slower processing. Use smaller sizes (256-416) for fast processing of large animals, and larger sizes (640-1280) when tracking small animals or when precision is critical.

- **THRESHOLD**: Confidence threshold for bounding box detection. All detections (bounding boxes AND keypoints) below this value are ignored. Options range from 0.1 to 1.0 in increments of 0.1. Default is 0.1.
  - **Effect**: Lower thresholds (0.1-0.3) include more detections, including low-confidence ones, which can help detect animals in challenging conditions (occlusion, poor lighting) but may introduce false positives. Higher thresholds (0.5-1.0) only keep high-confidence detections, reducing false positives but potentially missing some animals or keypoints. **Trade-off**: Lower threshold = more detections (including potential false positives) vs. higher threshold = fewer but more reliable detections. Start with 0.1-0.3 for challenging videos, increase to 0.4-0.6 if you see many false detections.

- **INTERPOLATE**: Whether to interpolate missing keypoints across frames using the 'nearest' method. Options: TRUE (default) or FALSE.
  - **Effect**: When TRUE, missing keypoints (due to occlusion, detection failures, or low confidence) are filled in by copying the position from the nearest frame where the keypoint was detected. This creates smoother tracking trajectories and reduces gaps in the data. When FALSE, missing keypoints remain as NaN/empty values. **Recommendation**: Keep TRUE (default) unless you specifically want to preserve gaps in the data for analysis purposes. Interpolation is especially important for downstream analysis in SimBA.

- **IOU**: Intersection over Union threshold for non-maximum suppression. Options range from 0.1 to 1.0 in increments of 0.1. Default is 0.8.
  - **Effect**: IOU controls how overlapping detections are handled. Lower IOU values (0.3-0.5) allow more overlapping bounding boxes to be kept, which can help when animals are close together or partially occluded, but may result in duplicate detections of the same animal. Higher IOU values (0.7-0.9) are more aggressive at removing overlapping detections, reducing duplicates but potentially missing animals when they overlap significantly. **Trade-off**: Lower IOU = more detections (including potential duplicates) vs. higher IOU = fewer but cleaner detections. Default of 0.8 works well for most scenarios. Reduce to 0.5-0.6 if animals are frequently overlapping, increase to 0.9 if you see duplicate detections.

- **STREAM**: If TRUE (default), processes frames one-by-one in a generator style, recommended for long videos. If FALSE, processes frames in batches.
  - **Effect**: STREAM mode (TRUE) processes frames sequentially, using less GPU memory and allowing processing of very long videos without memory issues. It's more memory-efficient but may be slightly slower. Batch mode (FALSE) loads multiple frames into memory at once, which can be faster for shorter videos but risks running out of memory on long videos. **Recommendation**: Keep TRUE (default) for videos longer than a few minutes or when GPU memory is limited. Use FALSE only for short videos with ample GPU memory.

- **CPU WORKERS**: Number of PyTorch threads to use for CPU processing. Options range from 1 to the maximum number of CPU cores available. Default is approximately half of the maximum cores.
  - **Effect**: More CPU workers can speed up data loading and preprocessing (reading frames from disk, resizing) but increases CPU usage. Fewer workers use less CPU resources but may slow down data loading, especially when processing many videos. **Trade-off**: More workers = faster data loading but higher CPU usage. Default (half of cores) is usually optimal. Increase if you have many CPU cores and want faster processing, decrease if you need to reserve CPU for other tasks.

- **VERBOSE**: If TRUE (default), outputs progress information and timing during inference. If FALSE, minimal output is displayed.
  - **Effect**: When TRUE, you'll see detailed progress updates, frame counts, processing times, and any warnings in the terminal/console. This is helpful for monitoring progress and debugging. When FALSE, only essential messages are shown. **Recommendation**: Keep TRUE (default) to monitor progress, especially for long-running inference tasks. Set to FALSE only if you're running automated scripts and want cleaner output.

- **DEVICE**: Select the device for inference. Options include "CPU" or specific GPU devices (e.g., "0 : NVIDIA GeForce RTX 3090"). The default is typically the first available GPU.
  - **Effect**: GPU inference (default) is much faster than CPU inference (often 10-100x faster). Use a specific GPU number if you have multiple GPUs and want to use a particular one. CPU inference is extremely slow and not recommended for video processing. **Recommendation**: Always use GPU unless you have no other option. If you have multiple GPUs, select the one with the most memory or the one not being used by other processes.

- **MAX TRACKS**: Maximum total number of pose tracks to keep across all classes. Options: None (all tracks retained), or 1-10. Default is None.
  - **Effect**: Limits the total number of animal tracks detected across all classes. When set to a number (e.g., 2), only the top 2 tracks by confidence are kept. When None, all detected tracks are retained. **Use case**: Set to a specific number (e.g., 2 for two-animal experiments) to filter out spurious detections or when you know the exact number of animals. Keep None if you want to detect all animals present, including unexpected ones.

- **MAX TRACKS PER ID**: Maximum number of pose tracks per class/ID. Useful when you expect a specific number of animals (e.g., if one 'resident' and one 'intruder' is expected, set this to 1). Options: None (all detected instances retained), or 1-10. Default is 1.
  - **Effect**: Controls how many instances of each animal class/ID are kept. If set to 1, only the highest-confidence detection for each class is kept. This is useful for multi-animal scenarios where you know exactly how many animals of each type should be present (e.g., 1 resident mouse, 1 intruder mouse). **Use case**: Set to 1 when you have exactly one animal per class. Set to None or higher numbers if multiple animals of the same class may be present. This setting works in conjunction with MAX TRACKS - MAX TRACKS limits total tracks, MAX TRACKS PER ID limits tracks per class.

- **SMOOTHING (MS)**: Time in milliseconds for Gaussian-applied body-part smoothing. Options: None, 50, 100, 200, 300, 400, 500. Default is 100.
  - **Effect**: Applies temporal smoothing to keypoint positions using a Gaussian filter over the specified time window. Higher values (300-500ms) create smoother trajectories by averaging over longer time periods, reducing jitter and small tracking errors, but may blur rapid movements. Lower values (50-100ms) preserve more detail and rapid movements but allow more jitter. None disables smoothing entirely. **Trade-off**: More smoothing = smoother but potentially less responsive tracking vs. less smoothing = more responsive but potentially jittery tracking. Default of 100ms is a good balance. Increase to 200-300ms for very jittery tracking, decrease to 50ms or None if you need to preserve rapid movements.

- **RECURSIVE VIDEO SEARCH**: If TRUE, analyzes all video files found recursively in the video directory. If FALSE (default), only looks in the top directory.
  - **Effect**: When TRUE, SimBA will search through all subdirectories within the selected video directory to find video files. When FALSE, only videos directly in the selected directory are processed. **Use case**: Set TRUE when your videos are organized in subfolders (e.g., by date, experiment, or condition). Set FALSE when all videos are in a single flat directory structure.

### 3. ANALYZE SINGLE VIDEO

This section allows you to analyze a single video:

- **VIDEO PATH**: Select the path to a single video file. Supports all standard video formats (MP4, AVI, etc.).
- **ANALYZE SINGLE VIDEO**: Button to start inference on the selected single video.

### 4. ANALYZE VIDEO DIRECTORY

This section allows you to analyze multiple videos in a directory:

- **VIDEO DIRECTORY PATH**: Select a directory containing video files.
- **ANALYZE VIDEO DIRECTORY**: Button to start inference on all videos in the selected directory.

## Usage Instructions

### Step 1: Configure Model and Data Paths

1. Click on **MODEL PATH (E.G., .PT)** and browse to select your trained YOLO model weights file.
2. Click on **SAVE DIRECTORY** and select the folder where you want inference results to be saved.
3. Click on **BODY-PART NAMES (.CSV)** and select the CSV file containing your body part names. If using the default 7-body-part configuration, the default file should be pre-selected.
4. (Optional) If you have a tracker configuration file, click on **TRACKER CONFIG (OPTIONAL, .YML)** and select your YAML configuration file.

### Step 2: Adjust Settings (Optional)

Review and adjust the settings in the **SETTINGS** section according to your needs:

- For faster processing on long videos, set **STREAM** to TRUE.
- Adjust **BATCH SIZE** based on your GPU memory (larger batches require more memory).
- Increase **THRESHOLD** if you want to filter out low-confidence detections.
- Enable **INTERPOLATE** to fill in missing keypoints across frames.
- Adjust **SMOOTHING (MS)** to reduce jitter in keypoint positions.
- Set **MAX TRACKS PER ID** to the expected number of animals per class.

### Step 3: Select Video(s) and Run Inference

**For a single video:**

1. Click on **VIDEO PATH** in the **ANALYZE SINGLE VIDEO** section and select your video file.
2. Click the **ANALYZE SINGLE VIDEO** button to start inference.

**For multiple videos:**

1. Click on **VIDEO DIRECTORY PATH** in the **ANALYZE VIDEO DIRECTORY** section and select the folder containing your videos.
2. If you want to search recursively in subdirectories, set **RECURSIVE VIDEO SEARCH** to TRUE.
3. Click the **ANALYZE VIDEO DIRECTORY** button to start inference on all videos in the directory.

### Step 4: Monitor Progress

If **VERBOSE** is set to TRUE, you will see progress information and timing details in the terminal/console window. The inference process will:

1. Validate all input paths and files.
2. Read video metadata to ensure videos are accessible.
3. Load the YOLO model and run inference on each video.
4. Save the results to the specified save directory.

## Output

The inference results are saved in the specified **SAVE DIRECTORY**. The output format depends on whether a tracker configuration was provided:

- **Without tracker config**: Results are saved using `YOLOPoseInference`, which outputs keypoint data in CSV format compatible with SimBA.
- **With tracker config**: Results are saved using `YOLOPoseTrackInference`, which includes additional tracking information based on the tracker configuration.

## Important Notes

1. **GPU Requirement**: This tool requires an NVIDIA GPU with CUDA support. CPU-only inference is not supported through this popup.

2. **Model Compatibility**: Ensure your YOLO model is compatible with the Ultralytics YOLO framework and supports pose estimation (keypoint detection).

3. **Body Part Configuration**: The body part names CSV file must match the keypoint structure of your trained model. The default 7-body-part configuration may need to be modified if your model tracks a different number or set of keypoints.

4. **Video Formats**: All standard video formats supported by SimBA can be processed (MP4, AVI, MOV, etc.).

5. **Performance**: For best performance on long videos, use **STREAM** mode (set to TRUE). For faster processing on shorter videos with sufficient GPU memory, you can increase the **BATCH SIZE**.

6. **Memory Considerations**: Larger batch sizes and image sizes require more GPU memory. If you encounter out-of-memory errors, reduce the **BATCH SIZE** or **IMAGE SIZE**.

7. **Tracking**: The **MAX TRACKS** and **MAX TRACKS PER ID** settings help manage multi-animal tracking scenarios. Set these appropriately based on the expected number of animals in your videos.

## Troubleshooting

- **"No NVIDIA GPUs detected"**: Ensure you have an NVIDIA GPU with CUDA drivers installed. Check GPU availability using `nvidia-smi` in your terminal.

- **"Could not find ultralytics package"**: Install the ultralytics package using `pip install ultralytics` or `conda install ultralytics` (I've been using version 8.3.156 while writing these docs.)

- **File not found errors**: Verify that all file paths are correct and files exist. Ensure video files are in supported formats.

- **Out of memory errors**: Reduce the **BATCH SIZE** or **IMAGE SIZE** settings to use less GPU memory.

- **Low detection rates**: Decrease the **THRESHOLD** value to include lower-confidence detections, or check that your model is appropriate for your video content.

## Related Documentation

- For training YOLO pose estimation models in SimBA, see the YOLO training documentation.
- For visualizing YOLO pose estimation results, see the YOLO visualization tools.
- For importing YOLO pose estimation results into SimBA, see pose estimation import tools.
- For more information on pose estimation in SimBA, see the [Multi-animal pose estimation tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md).

##
Author [Simon N](https://github.com/sronilsson)

