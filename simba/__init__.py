import os
import sys
from simba.utils.enums import Paths
from dotenv import load_dotenv

script_dir = os.path.dirname(sys.argv[0]) or os.getcwd()
env_path = os.path.join(script_dir, Paths.ENV_PATH.value)
load_dotenv(env_path)