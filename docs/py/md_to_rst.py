import pypandoc
from pypandoc.pandoc_download import download_pandoc
download_pandoc()

IN_PATH = '/Users/simon/Desktop/envs/simba_dev/docs/md/cue_lights.md'
OUT_PATH = '/Users/simon/Desktop/envs/simba_dev/docs/tutorials_rst/cue_lights.rst'

pypandoc.convert_file(IN_PATH, 'rst', outputfile=OUT_PATH)





