from simba.mixins.config_reader import ConfigReader

config_reader = ConfigReader(config_path='tests/data/test_projects/two_c57/project_folder/project_config.ini')
bp_dict = config_reader.create_body_part_dictionary(multi_animal_status=config_reader.multi_animal_status,
                                                    animal_id_lst=config_reader.multi_animal_id_list,
                                                    animal_cnt=config_reader.animal_cnt,
                                                    x_cols=config_reader.x_cols,
                                                    y_cols=config_reader.y_cols,
                                                    p_cols=config_reader.p_cols,
                                                    colors=config_reader.clr_lst)