from tkinter import *

from simba.data_processors.boolean_conditional_calculator import \
    BooleanConditionalCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, Label, LabelFrame)
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import CountError, DuplicationError
from simba.utils.read_write import read_df


class BooleanConditionalSlicerPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(
            self, title="CONDITIONAL BOOLEAN AGGREGATE STATISTICS", size=(600, 400)
        )
        self.rule_cnt_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="CONDITIONAL RULES #",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.AGGREGATE_BOOL_STATS.value,
        )
        self.rule_cnt_dropdown = DropDownMenu(
            self.rule_cnt_frm,
            "# RULES:",
            list(range(2, 21)),
            "25",
            com=self.create_rules_frames,
        )
        self.rule_cnt_dropdown.setChoices(2)
        self.rule_cnt_frm.grid(row=0, column=0, sticky="NW")
        self.rule_cnt_dropdown.grid(row=0, column=0, sticky="NW")

        self.create_run_frm(run_function=self.run)
        check_if_filepath_list_is_empty(
            filepaths=self.feature_file_paths,
            error_msg=f"No data found in {self.features_dir}",
        )
        data_df = read_df(
            file_path=self.feature_file_paths[0], file_type=self.file_type
        )
        self.bool_cols = data_df.columns[data_df.apply(self._is_bool)]
        if len(self.bool_cols) < 2:
            raise CountError(
                msg=f"The data file {self.feature_file_paths[0]} contains less than 2 boolean columns",
                source=self.__class__.__name__,
            )
        self.create_rules_frames(rules_cnt=2)
        self.main_frm.mainloop()

    @staticmethod
    def _is_bool(column):
        unique_values = set(column)
        return unique_values.issubset({0, 1})

    def create_rules_frames(self, rules_cnt: int):
        if hasattr(self, "rule_definitions_frame"):
            self.rule_definitions_frame.destroy()
        self.rule_definitions_frame = LabelFrame(
            self.main_frm,
            text="CONDITIONAL RULES",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.rule_definitions_frame.grid(row=1, column=0, sticky="NW")
        Label(self.rule_definitions_frame, text="RULE #").grid(
            row=0, column=0, sticky=NW
        )
        Label(self.rule_definitions_frame, text="BEHAVIOR").grid(
            row=0, column=1, sticky=NW, padx=5
        )
        Label(self.rule_definitions_frame, text="STATUS").grid(
            row=0, column=2, sticky=NW, padx=5
        )

        self.rules = {}
        for rule_cnt in range(1, rules_cnt + 1):
            self.rules[rule_cnt] = {}
            Label(
                self.rule_definitions_frame,
                text=str(rule_cnt),
                font=Formats.LABELFRAME_HEADER_FORMAT.value,
            ).grid(row=rule_cnt, column=0, sticky=NW)
            self.rules[rule_cnt]["behavior_drpdwn"] = DropDownMenu(
                self.rule_definitions_frame,
                "",
                self.bool_cols,
                "1",
            )
            self.rules[rule_cnt]["behavior_drpdwn"].setChoices(
                self.bool_cols[rule_cnt - 1]
            )
            self.rules[rule_cnt]["behavior_drpdwn"].grid(
                row=rule_cnt, column=1, sticky=NW
            )
            self.rules[rule_cnt]["status_drpdwn"] = DropDownMenu(
                self.rule_definitions_frame, "", Options.BOOL_STR_OPTIONS.value, "1"
            )
            self.rules[rule_cnt]["status_drpdwn"].setChoices(
                Options.BOOL_STR_OPTIONS.value[0]
            )
            self.rules[rule_cnt]["status_drpdwn"].grid(
                row=rule_cnt, column=2, sticky=NW
            )

    def run(self):
        unique_rule_behaviors = []
        selections = {}
        for rule_id, rule_data in self.rules.items():
            unique_rule_behaviors.append(rule_data["behavior_drpdwn"].getChoices())
            selections[rule_data["behavior_drpdwn"].getChoices()] = rule_data[
                "status_drpdwn"
            ].getChoices()
        duplicates = list(
            set(
                [x for x in unique_rule_behaviors if unique_rule_behaviors.count(x) > 1]
            )
        )
        if len(duplicates) > 0:
            raise DuplicationError(
                msg=f"Each row should be a unique behavior. However, behaviors {duplicates} are selected in more than 1 rows."
            )
        boolean_calculator = BooleanConditionalCalculator(
            config_path=self.config_path, rules=selections
        )
        boolean_calculator.run()
        boolean_calculator.save()


# roi_featurizer = BooleanConditionalSlicerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
# roi_featurizer = BooleanConditionalSlicerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
