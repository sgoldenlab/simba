import argparse
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_module_has_import, check_instance,
                                check_str)
from simba.utils.errors import CountError, InvalidFileTypeError
from simba.utils.printing import stdout_warning
from simba.utils.read_write import get_fn_ext

ABSTRACT_CLASS_NAME = "AbstractFeatureExtraction"
PY_FILE_EXT = ".py"


class CustomFeatureExtractor(ConfigReader):
    """
    Class to execute a feature extraction process based on the user-defined Python script.

    This method performs the following steps:
    1. Parses the user-defined Python script.
    2. Identifies class and function names in the script.
    3. Checks for the presence of required imports and specific code patterns.
    4. Handle cases of multiple classes and missing configuration arguments.
    5. Invokes the feature extraction process if conditions are met.

    .. notes::

        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/extractFeatures.md>`_.

        The user defined feature extraction class needs to contain a ``config_path`` init argument

        If the feature extraction class contains multiple classes, then the first class  is used.

        If the feature extraction class relies on argparse AND inherits ``simba.mixins.abstract_classes.AbstractFeatureExtraction``
        then the feature extraction class will be executed thtough the subprocess module. This ensures that the GUI cannot interfere
        with the feature extraction process, and reliably execution of multicore processes, if present, in the feature-extraction class.

        If the feature extraction class does not rely on argparse, then the class will be loaded and executed in python through ``sys``. It
        might still be quick and reliable (it often is). However, I have noted that some function, in particular functions that rely on
        ``multiprocessing.imap``, are disrupted by the GUI with unacceptable effects on runtime when executed thriugh ``sys``.

        Thus, it is recommended that the custom feature extraction class inherits ``simba.mixins.abstract_classes.AbstractFeatureExtraction`` and is executed through argparse.

        For an example feature extraction class that inherits from ``simba.mixins.abstract_classes.AbstractFeatureExtraction`` and is executed through argparse, see
        `this file <https://github.com/sgoldenlab/simba/blob/master/misc/geometry_feature_extraction.py>`_. For an example feature extraction class that does **NOT** inherit from ``simba.mixins.abstract_classes.AbstractFeatureExtraction``
        and does not use argparse, see `this file <https://github.com/lapphe/AMBER-pipeline/blob/main/SimBA_AMBER_project/AMBER_2_0__feature_extraction/amber_feature_extraction_20230815.py>`_. More examples can be found in `this github documentation <https://github.com/sgoldenlab/simba/blob/master/docs/extractFeatures.md>`_.
        We can give prompt help with troubleshooting through Gitter or GitHb.

    :example:
    >>> test = CustomFeatureExtractor(extractor_file_path='/simba/misc/piotr.py', config_path='/Users/simon/Desktop/envs/troubleshooting/piotr/project_folder/train-20231108-sh9-frames-with-p-lt-2_plus3-&3_best-f1.ini')
    >>> test.run()
    >>> test = CustomFeatureExtractor(config_path='/Users/simon/Desktop/envs/troubleshooting/piotr/project_folder/train-20231108-sh9-frames-with-p-lt-2_plus3-&3_best-f1.ini', file_path='/simba/misc/piotr.py')
    >>> test.run()
    >>> test = CustomFeatureExtractor(extractor_file_path='/Users/simon/Desktop/envs/simba/simba/simba/feature_extractors/amber_feature_extractor.py', config_path='/Users/simon/Desktop/envs/simba/troubleshooting/Amber_test/project_folder/project_config.ini')
    >>> test.run()
    """

    def __init__(self,
                 extractor_file_path: Union[str, os.PathLike],
                 config_path: Union[str, os.PathLike]):
        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=extractor_file_path)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)

        file_dir, file_name, file_extension = get_fn_ext(filepath=extractor_file_path)
        if file_extension.lower() != PY_FILE_EXT:
            raise InvalidFileTypeError(msg=f"The user-defined feature extraction file ({extractor_file_path}) is not a .py file-extension", source=self.__class__.__name__)
        self.extractor_file_path, self.config_path = extractor_file_path, config_path

    def _find_astClass_from_name(self, parsed_py: ast.Module, class_name: str) -> ast.ClassDef:
        """
        Find and return the AST node representing a class with the specified name in the given AST module.

        :param ast.Module parsed_py: The AST module to search for the class.
        :param str class_name: The name of the class to find.
        :return ast.ClassDef : The AST node representing the class with the specified name.
        """

        classes = [n for n in parsed_py.body if isinstance(n, ast.ClassDef)]
        classes_match = [n for n in classes if n.name is class_name]
        if len(classes_match) == 0:
            raise CountError(msg=f"No classes named {class_name} in {self.extractor_file_path}", source=self.__class__.__name__)
        else:
            return classes_match[0]

    def _find_class_names(self, parsed_py: ast.Module) -> list:
        """
        Find and return the names of classes defined in the given AST module.

        :param ast.Module parsed_py: The AST module to search for function names.
        :return list: A list of class names found in the AST module.
        """

        check_instance(source=self.__class__.__name__, instance=parsed_py, accepted_types=ast.Module)
        classes = [n for n in parsed_py.body if isinstance(n, ast.ClassDef)]
        return [x.name for x in classes]

    def _find_function_names(self, parsed_py: ast.Module) -> list:
        """
        Find and return the names of functions defined in the given AST module.

        :param ast.Module parsed_py: The AST module to search for function names.
        :return list: A list of function names found in the AST module.
        """
        check_instance(source=self.__class__.__name__, instance=parsed_py, accepted_types=ast.Module)
        functions = [n for n in parsed_py.body if isinstance(n, ast.FunctionDef)]
        return [x.name for x in functions]

    def _check_inheritance(self, class_: ast.ClassDef, inheritance: Optional[str] = ABSTRACT_CLASS_NAME) -> bool:
        """
        Check if a class inherits from a specified class.

        :param ast.ClassDef class_: The AST node representing the class to check.
        :param Optional[str] inheritance: The name of the class to check for inheritance. Defaults to the abstract class name.
        :return bool: True if the specified class is found in the class's bases, False otherwise.
        """
        check_instance(source=self.__class__.__name__, instance=class_, accepted_types=ast.ClassDef)
        check_str(name=f'{self.__class__.__name__} inheritance', value=inheritance)
        bases = [base.id for base in class_.bases if isinstance(base, ast.Name)]
        if inheritance in bases:
            return True
        else:
            return False

    def has_block(self, file_path: Union[str, os.PathLike], target: str) -> bool:
        """
        Check if a specified block of text exists in a file.

        :param Union[str, os.PathLike] file_path: Path to the file to check.
        :param str target: The block of text to search for in the file.
        :return bool: True if the specified block of text is found in the file, False otherwise.
        """

        check_file_exist_and_readable(file_path=file_path)
        check_str(name='target', value=target)
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() == target:
                    return True
        return False

    def has_argparser_argument(self, file_path: Union[str, os.PathLike], target_argument: str) -> bool:
        """
        Check if a Python script, using argparse, has a specific command-line argument.

        :param Union[str, os.PathLike] file_path: Path to the Python script file.
        :param str target_argument: The name of the argument to check.
        :return bool: True if the specified argument is found in the script's argparse configuration, False otherwise.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--" + target_argument)
        file_content = Path(file_path).read_text()
        try:
            parsed_args, _ = parser.parse_known_args(file_content.split())
            return hasattr(parsed_args, target_argument)
        except argparse.ArgumentError:
            return False

    def run(self):
        parsed_py = ast.parse(Path(self.extractor_file_path).read_text())
        class_names = self._find_class_names(parsed_py=parsed_py)
        function_names = self._find_function_names(parsed_py=parsed_py)
        has_argparse = check_if_module_has_import(parsed_file=parsed_py, import_name="argparse")
        if len(class_names) < 1:
            raise CountError(msg=f"The user-defined feature extraction file ({self.extractor_file_path}) contains no python classes", source=self.__class__.__name__)
        elif len(class_names) > 1:
            stdout_warning(msg=f"The user-defined feature extraction file ({self.extractor_file_path}) contains more than 1 python class ({class_names}). SimBA will use the first python class: {class_names[0]}.")

        class_name = class_names[0]
        class_ = self._find_astClass_from_name(parsed_py=parsed_py, class_name=class_name)
        has_abstract_inheritance = self._check_inheritance(class_=class_)
        has_main_block = self.has_block(file_path=self.extractor_file_path, target='if __name__ == "__main__":')
        has_config_argparse = self.has_argparser_argument(file_path=self.extractor_file_path, target_argument="config_path")
        if (
            has_abstract_inheritance
            and has_config_argparse
            and has_argparse
            and has_main_block
        ):
            command = f'python "{self.extractor_file_path}" --config_path "{self.config_path}"'
            print("Follow feature extraction progress in the operating system terminal window...")
            subprocess.call(command, shell=True)

        else:
            print("Running user-defined feature extraction class...")
            spec = importlib.util.spec_from_file_location(class_name, self.extractor_file_path)
            user_module = importlib.util.module_from_spec(spec)
            sys.modules[class_name] = user_module
            spec.loader.exec_module(user_module)
            user_class = getattr(user_module, class_name)

            if has_abstract_inheritance:
                feature_extractor = user_class(self.config_path)
                feature_extractor.run()

            else:
                user_class(self.config_path)

# test = CustomFeatureExtractor(extractor_file_path='/Users/simon/Desktop/envs/simba/simba/simba/feature_extractors/amber_feature_extractor.py',
#                               config_path='/Users/simon/Desktop/envs/simba/troubleshooting/Amber_test/project_folder/project_config.ini')
# test.run()
