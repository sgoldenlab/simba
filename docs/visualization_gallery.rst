🎨 Visualization gallery
========================

A tour of what SimBA can render — plotting outputs and the video/image
processing tools. Each tile links to the code that produces it, where you'll
find the full API, options and a playable demo.

.. seealso::
   For step-by-step instructions, see the :doc:`visualization tutorials <Visualizations>`.


Plotting
--------

.. container:: simba-gallery

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/bloblocationcomputer.webp
         :alt: Blob plotter

      :class:`Blob plotter <simba.plotting.blob_plotter.BlobPlotter>`

      Plot the results of animal tracking based on blob.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/blobvisualizer.webp
         :alt: Blob visualizer

      :class:`Blob visualizer <simba.plotting.blob_visualizer.BlobVisualizer>`

      Visualize blob tracking data by overlaying geometric shapes and body part markers on video frames.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/circular_visualiation.webp
         :alt: Circular feature plotter

      :class:`Circular feature plotter <simba.plotting.circular_feature_overlay_plotter.CircularFeaturePlotter>`

      Create visualization of base angular features overlay on video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/cuelightvisualizer.webp
         :alt: Cue light visualizer

      :class:`Cue light visualizer <simba.plotting.cue_light_visualizer.CueLightVisualizer>`

      Visualize SimBA computed cue-light ON and OFF states and the aggregate statistics of ON and OFF states.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/data_plot.webp
         :alt: Data plotter

      :class:`Data plotter <simba.plotting.data_plotter.DataPlotter>`

      Tabular data visualization of animal movement and distances in the current frame and their aggregate statistics.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/directing_other_animals.webp
         :alt: Directing other animals visualizer

      :class:`Directing other animals visualizer <simba.plotting.directing_animals_visualizer.DirectingOtherAnimalsVisualizer>`

      Create videos visualizing when animals direct their gaze toward body parts of other animals (single-threaded).

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/directingroivisualizer.webp
         :alt: Directing ROI visualizer

      :class:`Directing ROI visualizer <simba.plotting.roi_directing_visualizer.DirectingROIVisualizer>`

      Visualize when animals are directing towards ROIs.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/distance_plot.webp
         :alt: Distance plotter

      :class:`Distance plotter <simba.plotting.distance_plotter.DistancePlotterSingleCore>`

      Visualize frame-wise body-part distances as line plots using single-core processing.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/mosaic_videos.webp
         :alt: Frame mergerer FFmpeg

      :class:`Frame mergerer FFmpeg <simba.plotting.frame_mergerer_ffmpeg.FrameMergererFFmpeg>`

      Merge separate visualizations of classifications, descriptive statistics etc., into single video mosaic.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/gantt_plot.webp
         :alt: Gantt creator

      :class:`Gantt creator <simba.plotting.gantt_creator.GanttCreatorSingleProcess>`

      Create classifier Gantt charts using single-process execution.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/gantt_plotly.webp
         :alt: Gantt plotly

      :func:`Gantt plotly <simba.plotting.gantt_plotly.gantt_plotly>`

      Generates a Gantt chart using Plotly to visualize bout events over time.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/geometryplotter.webp
         :alt: Geometry plotter

      :class:`Geometry plotter <simba.plotting.geometry_plotter.GeometryPlotter>`

      A class for creating overlay geometry visualization videos based on provided geometries and video name.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/heatmap.webp
         :alt: Heat mapper classifier

      :class:`Heat mapper classifier <simba.plotting.heat_mapper_clf.HeatMapperClfSingleCore>`

      Create heatmaps representing the locations of the classified behavior.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/heatmap_location.webp
         :alt: Heatmapper location

      :class:`Heatmapper location <simba.plotting.heat_mapper_location.HeatmapperLocationSingleCore>`

      Create heatmaps representing the location where animals spend time.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/interactive_probability_plot.webp
         :alt: Interactive probability grapher

      :class:`Interactive probability grapher <simba.plotting.interactive_probability_grapher.InteractiveProbabilityGrapher>`

      Launch interactive GUI for inspecting classifier probabilities with synchronized video playback.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/lightdarkboxplotter.webp
         :alt: Light dark box plotter

      :class:`Light dark box plotter <simba.plotting.light_dark_box_plotter.LightDarkBoxPlotter>`

      Generate annotated videos visualizing behavior episodes in a light/dark box setup.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/path_plot.webp
         :alt: Path plotter

      :class:`Path plotter <simba.plotting.path_plotter.PathPlotterSingleCore>`

      Create "path plots" videos and/or images detailing the movement paths of individual animals in SimBA.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/sklearn_visualization.webp
         :alt: Plot sklearn results

      :class:`Plot sklearn results <simba.plotting.plot_clf_results.PlotSklearnResultsSingleCore>`

      Plot classification results overlays on videos.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/pose_plotter.webp
         :alt: Pose plotter

      :class:`Pose plotter <simba.plotting.pose_plotter_mp.PosePlotterMultiProcess>`

      Create pose-estimation visualizations from data within a SimBA project folder using multiprocessing.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/get_path_img.webp
         :alt: Quick path plot

      :class:`Quick path plot <simba.plotting.ez_path_plot.EzPathPlot>`

      Create a simple path plot image for a single or several pose-estimation files.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roi_visualize.webp
         :alt: RO ifeature visualizer

      :class:`RO ifeature visualizer <simba.plotting.ROI_feature_visualizer.ROIfeatureVisualizer>`

      Visualizing features that depend on the relationships between the location of the animals and user-defined ROIs.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roi_visualize.webp
         :alt: ROI plot

      :class:`ROI plot <simba.plotting.roi_plotter_mp.ROIPlotMultiprocess>`

      Visualize the ROI data (number of entries/exits, time-spent in ROIs) using multiprocessing for improved performance.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roiplot_1.webp
         :alt: ROI plotter

      :class:`ROI plotter <simba.plotting.roi_plotter.ROIPlotter>`

      Visualize the ROI data (number of entries/exits, time-spent in ROIs etc).

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/shap.webp
         :alt: SHAP explanations

      :func:`SHAP explanations <simba.data_processors.cuda.create_shap_log.create_shap_log>`

      Calculate aggregate (binned) SHAP value statistics where individual bins represent reaulated features.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/skeletonvideocreator.webp
         :alt: Skeleton video creator

      :class:`Skeleton video creator <simba.plotting.skeleton_video_creator.SkeletonVideoCreator>`

      Create pose-estimation videos rendered on a solid RGB background from SimBA CSV data.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/spontaneousalternationsplotter.webp
         :alt: Spontaneous alternations plotter

      :class:`Spontaneous alternations plotter <simba.plotting.spontaneous_alternation_plotter.SpontaneousAlternationsPlotter>`

      Create plots representing delayed-alternation computations overlayed on video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/prob_plot.webp
         :alt: Treshold plot creator

      :class:`Treshold plot creator <simba.plotting.probability_plot_creator.TresholdPlotCreatorSingleProcess>`

      Create classifier-probability line plots using single-process execution.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/t1.webp
         :alt: Validate model one video

      :class:`Validate model one video <simba.plotting.single_run_model_validation_video_mp.ValidateModelOneVideoMultiprocess>`

      Create classifier validation video for a single input video using multiprocessing for improved performance.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/yolomodelcomparator.webp
         :alt: YOLO model comparator

      :class:`YOLO model comparator <simba.plotting.compare_bbox_mdls.YoloModelComparator>`

      Compare two or more YOLO models side-by-side on a set of test videos.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/yoloposevisualizer.webp
         :alt: YOLO pose visualizer

      :class:`YOLO pose visualizer <simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer>`

      Visualizes YOLO-based keypoint pose estimation data on video frames and creates an annotated output video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/yolovisualizer.webp
         :alt: YOLO visualizer

      :class:`YOLO visualizer <simba.plotting.yolo_visualize.YOLOVisualizer>`

      Visualize YOLO bounding-box inference results on a source video.


Video & image tools
-------------------

.. container:: simba-gallery

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/to_greyscale.webp
         :alt: Batch video to greyscale

      :func:`Batch video to greyscale <simba.video_processors.video_processing.batch_video_to_greyscale>`

      Convert a directory of video file to greyscale mp4 format.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/brightness_contrast_ui.webp
         :alt: Brightness contrast UI

      :class:`Brightness contrast UI <simba.video_processors.brightness_contrast_ui.BrightnessContrastUI>`

      Create a user interface using OpenCV to explore and change the brightness and contrast of a video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/playback_speed.webp
         :alt: Change playback speed

      :func:`Change playback speed <simba.video_processors.video_processing.change_playback_speed>`

      Change the playback speed of a video file.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/change_single_video_fps.webp
         :alt: Change single video fps

      :func:`Change single video fps <simba.video_processors.video_processing.change_single_video_fps>`

      Change the fps of a single video file.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/clahe_enhance_video.webp
         :alt: CLAHE enhance video

      :func:`CLAHE enhance video <simba.video_processors.video_processing.clahe_enhance_video>`

      Convert a single video file to clahe-enhanced greyscale.avi file.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/convert_to_avi.webp
         :alt: Convert to avi

      :func:`Convert to avi <simba.video_processors.video_processing.convert_to_avi>`

      Convert a directory containing videos, or a single video, to AVI format using passed quality and codec.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/convert_to_mp4_1.webp
         :alt: Convert to mp4

      :func:`Convert to mp4 <simba.video_processors.video_processing.convert_to_mp4>`

      Convert a directory containing videos, or a single video, to MP4 format using passed quality and codec.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/convert_to_webm.webp
         :alt: Convert to webm

      :func:`Convert to webm <simba.video_processors.video_processing.convert_to_webm>`

      Convert a directory containing videos, or a single video, to WEBM format using passed quality and codec.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/convert_to_webp.webp
         :alt: Convert to webp

      :func:`Convert to webp <simba.video_processors.video_processing.convert_to_webp>`

      Convert the file type of all image files within a directory to WEBP format of passed quality.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/create_average_frm_1.webp
         :alt: Create average frame

      :func:`Create average frame <simba.video_processors.video_processing.create_average_frm>`

      Create an image representing the average frame of a segment in a video or an entire video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/crop_single_video_circle.webp
         :alt: Crop multiple videos circles

      :func:`Crop multiple videos circles <simba.video_processors.video_processing.crop_multiple_videos_circles>`

      Crop multiple videos based on circular regions of interest (ROIs) selected by the user.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/crop_single_video.webp
         :alt: Crop single video

      :func:`Crop single video <simba.video_processors.video_processing.crop_single_video>`

      Crop a single video using ~simba.video_processors.roi_selector.ROISelector interface.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/crop_single_video_circle.webp
         :alt: Crop single video circle

      :func:`Crop single video circle <simba.video_processors.video_processing.crop_single_video_circle>`

      Crop a video based on circular regions of interest (ROIs) selected by the user.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roi_selector_polygon.webp
         :alt: Crop single video polygon

      :func:`Crop single video polygon <simba.video_processors.video_processing.crop_single_video_polygon>`

      Crop a video based on polygonal regions of interest (ROIs) selected by the user.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/crossfade_two_videos.webp
         :alt: Crossfade two videos

      :func:`Crossfade two videos <simba.video_processors.video_processing.crossfade_two_videos>`

      Cross-fade two videos.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/egocentricalaligner_2.webp
         :alt: Egocentric video rotator

      :class:`Egocentric video rotator <simba.video_processors.egocentric_video_rotator.EgocentricVideoRotator>`

      Perform egocentric rotation of a video using CPU multiprocessing.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/flip_videos.webp
         :alt: Flip videos

      :func:`Flip videos <simba.video_processors.video_processing.flip_videos>`

      Flip a video or directory of videos horizontally, vertically, or both, and save them to the specified directory.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/get_video_slic.webp
         :alt: Get video slic

      :func:`Get video slic <simba.video_processors.video_processing.get_video_slic>`

      Apply SLIC superpixel segmentation to all frames of a video and save the output as a new video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/horizontal_video_concatenator.webp
         :alt: Horizontal video concatenator

      :func:`Horizontal video concatenator <simba.video_processors.video_processing.horizontal_video_concatenator>`

      Concatenates multiple videos horizontally.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/interactive_clahe_ui.webp
         :alt: Interactive CLAHE UI

      :func:`Interactive CLAHE UI <simba.video_processors.clahe_ui.interactive_clahe_ui>`

      Create a user interface using OpenCV to explore and set appropriate CLAHE settings tile size and clip limit.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/mixed_mosaic_concatenator.webp
         :alt: Mixed mosaic concatenator

      :func:`Mixed mosaic concatenator <simba.video_processors.video_processing.mixed_mosaic_concatenator>`

      Create a mixed mosaic video by concatenating multiple input videos in a mosaic layout of various sizes.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/mosaic_concatenator.webp
         :alt: Mosaic concatenator

      :func:`Mosaic concatenator <simba.video_processors.video_processing.mosaic_concatenator>`

      Concatenates multiple videos into a mosaic layout.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/multicrop.webp
         :alt: Multi cropper

      :class:`Multi cropper <simba.video_processors.multi_cropper.MultiCropper>`

      Crop each video of a specific file format (e.g., mp4) in a directory into N smaller cropped videos.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/reverse_videos.webp
         :alt: Reverse videos

      :func:`Reverse videos <simba.video_processors.video_processing.reverse_videos>`

      Reverses one or more video files located at the specified path and saves the reversed videos in the specified directory.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roi_blurbox.webp
         :alt: ROI blurbox

      :func:`ROI blurbox <simba.video_processors.video_processing.roi_blurbox>`

      Blurs either the selected or unselected portion of a user-specified region-of-interest according to the passed blur level.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roi_selector.webp
         :alt: ROI selector

      :class:`ROI selector <simba.video_processors.roi_selector.ROISelector>`

      A class for selecting and reflecting Regions of Interest (ROI) in an image.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/circle_crop_2.webp
         :alt: ROI selector circle

      :class:`ROI selector circle <simba.video_processors.roi_selector_circle.ROISelectorCircle>`

      Class for selecting a circular region of interest (ROI) within an image or video frame.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/roi_selector_polygon.webp
         :alt: ROI selector polygon

      :class:`ROI selector polygon <simba.video_processors.roi_selector_polygon.ROISelectorPolygon>`

      Class for selecting a polygonal region of interest (ROI) within an image or video frame.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/rotate_video.webp
         :alt: Rotate video

      :func:`Rotate video <simba.video_processors.video_processing.rotate_video>`

      Rotate a video or a directory of videos by a specified number of degrees.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/superimpose_elapsed_time.webp
         :alt: Superimpose elapsed time

      :func:`Superimpose elapsed time <simba.video_processors.video_processing.superimpose_elapsed_time>`

      Superimposes elapsed time on the given video file(s) and saves the modified video(s).

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/superimpose_frame_count.webp
         :alt: Superimpose frame count

      :func:`Superimpose frame count <simba.video_processors.video_processing.superimpose_frame_count>`

      Superimpose frame count on a video file.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/superimpose_freetext.webp
         :alt: Superimpose freetext

      :func:`Superimpose freetext <simba.video_processors.video_processing.superimpose_freetext>`

      Superimposes passed text on the given video file(s) and saves the modified video(s).

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/inset_overlay_video.webp
         :alt: Superimpose overlay video

      :func:`Superimpose overlay video <simba.video_processors.video_processing.superimpose_overlay_video>`

      Inset a video overlay on a second video with specified size, opacity, and location.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/superimpose_video_names.webp
         :alt: Superimpose video names

      :func:`Superimpose video names <simba.video_processors.video_processing.superimpose_video_names>`

      Superimposes the video name on the given video file(s) and saves the modified video(s).

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/overlay_video_progressbar.webp
         :alt: Superimpose video progressbar

      :func:`Superimpose video progressbar <simba.video_processors.video_processing.superimpose_video_progressbar>`

      Overlay a progress bar on a directory of videos or a single video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/vertical_video_concatenator.webp
         :alt: Vertical video concatenator

      :func:`Vertical video concatenator <simba.video_processors.video_processing.vertical_video_concatenator>`

      Concatenates multiple videos vertically.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/video_bg_subtraction.webp
         :alt: Video background subtraction

      :func:`Video background subtraction <simba.video_processors.video_processing.video_bg_subtraction>`

      Subtract the background from a video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/horizontal_video_concatenator.webp
         :alt: Video concatenator

      :func:`Video concatenator <simba.video_processors.video_processing.video_concatenator>`

      Concatenate two videos to a single video either horizontally or vertically

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/videorotator.webp
         :alt: Video rotator

      :class:`Video rotator <simba.video_processors.video_processing.VideoRotator>`

      GUI Tool for rotating video.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/video_to_bw.webp
         :alt: Video to BW

      :func:`Video to BW <simba.video_processors.video_processing.video_to_bw>`

      Convert video to black and white using passed threshold.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/to_greyscale.webp
         :alt: Video to greyscale

      :func:`Video to greyscale <simba.video_processors.video_processing.video_to_greyscale>`

      Convert a video file to greyscale mp4 format.

   .. container:: simba-gallery-card

      .. image:: _static/img/gallery/watermark_video.webp
         :alt: Watermark video

      :func:`Watermark video <simba.video_processors.video_processing.watermark_video>`

      Watermark a video file or a directory of video files with specified watermark size, opacity, and location.

