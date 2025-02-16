import pypandoc
from pypandoc.pandoc_download import download_pandoc
download_pandoc()

IN_PATH = r"C:\Users\sroni\Downloads\installation_new.md"
OUT_PATH = r"C:\Users\sroni\Downloads\installation_new.rst"

pypandoc.convert_file(IN_PATH, 'rst', outputfile=OUT_PATH)





