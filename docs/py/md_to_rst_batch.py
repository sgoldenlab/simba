import pypandoc
import os, glob
from simba.utils.read_write import get_fn_ext
from pypandoc.pandoc_download import download_pandoc
download_pandoc()

MD_PATH = '/Users/simon/Desktop/envs/simba_dev/docs/md'
RST_PATH = '/Users/simon/Desktop/envs/simba_dev/docs/tutorials_rst'

input_paths = glob.glob(MD_PATH + '/*.md')

for input_path in input_paths:
    _, file_name, _= get_fn_ext(input_path)
    output_path = os.path.join(RST_PATH, file_name + '.rst')
    pypandoc.convert_file(input_path, 'rst', outputfile=output_path)



#output = pypandoc.convert_file('/Users/simon/Desktop/envs/simba_dev/docs/FSTTC.md', 'rst')


