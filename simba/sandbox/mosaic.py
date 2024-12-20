import subprocess

def create_mosaic(input_files, output_file, panel_width, panel_height):
    num_videos = len(input_files)
    num_cols = (num_videos + 1) // 2  # Ensure at least 2 columns
    num_rows = 2  # Fixed two rows

    # Generate filter_complex string with tile
    filter_complex = f"nullsrc=size={panel_width}x{panel_height} [base];"
    for i, input_file in enumerate(input_files):
        filter_complex += f"[{i}:v] setpts=PTS-STARTPTS, scale={panel_width}x{panel_height} [vid{i}];"

    filter_complex += f"[base][vid0] overlay=shortest=1 [tmp];"
    for i in range(1, num_videos):
        filter_complex += f"[tmp][vid{i}] overlay=shortest=1:x={i * panel_width} [tmp{i}];"

    filter_complex += f"[tmp{num_videos - 1}] tile={num_cols}x{num_rows}"

    # Generate FFmpeg command
    ffmpeg_command = (
            f"ffmpeg "
            + " ".join([f'-i "{input_file}"' for input_file in input_files])
            + f" -filter_complex \"{filter_complex}\" "
            + f"-c:v libx264 -crf 18 -preset veryfast {output_file} -y"
    )

    # Run FFmpeg command and capture stderr for logging
    subprocess.run(ffmpeg_command, shell=True)

# Example usage:
input_files = ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/2022-06-21_NOB_IOT_23.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4']
output_file = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test_.mp4'
panel_width = 100  # Example width for each panel
panel_height = 100  # Example height for each panel
create_mosaic(input_files, output_file, panel_width, panel_height)
