import os

def write_inifile(msconfig,project_path,project_name,no_targets,target_list,bp):

############create directories################
    directory = project_path
    #generate main directories in project path
    project_folder = str(directory +'\\' + project_name + '\\project_folder')
    models_folder = str(directory + '\\' + project_name +'\\models')
    #generate sub-directories in main directories
    config_folder = str(project_folder + '\\configs')
    csv_folder = str(project_folder + '\\csv')
    frames_folder = str(project_folder + '\\frames')
    logs_folder = str(project_folder + '\\logs')
    videos_folder = str(project_folder + '\\videos')
    #csv folder
    features_extracted_folder = str(csv_folder + '\\features_extracted')
    input_csv_folder = str(csv_folder + '\\input_csv')
    machine_results_folder = str(csv_folder + '\\machine_results')
    outlier_corrected_movement_folder = str(csv_folder + '\\outlier_corrected_movement')
    outlier_corrected_location_folder = str(csv_folder + '\\outlier_corrected_movement_location')
    targets_inserted_folder = str(csv_folder + '\\targets_inserted')
    #frames
    input_folder = str(frames_folder + '\\input')
    output_folder = str(frames_folder + '\\output')

    folder_list = [project_folder,models_folder,config_folder,csv_folder,frames_folder,logs_folder,videos_folder,features_extracted_folder,input_csv_folder,machine_results_folder,outlier_corrected_movement_folder,outlier_corrected_location_folder,targets_inserted_folder,input_folder,output_folder]

    for i in folder_list:
        try:
            # Create target Directory
            os.makedirs(i)
           # print("Directory ", os.path.basename(i), " created ")
        except FileExistsError:
            pass
           # print("Directory ", os.path.basename(i), " already exists")


########create text file ##################
    f = open(project_folder + "\\project_config.ini","w+")

    #general settings
    f.write('[General settings]\n')
    f.write('project_path = ' + str(project_folder)+'\n')
    f.write('project_name = ' + str(project_name) +'\n')
    f.write('csv_path = ' + str(csv_folder) +'\n')
    f.write('use_master_config = ' + str(msconfig) +'\n')
    f.write('config_folder = ' + str(config_folder)+ '\n')
    f.write('\n')

    #sml setings
    f.write('[SML settings]\n')
    f.write('model_dir = ' + str(models_folder) +'\n')
    #for loop model path
    for i in range(int(no_targets)):
        f.write('model_path_' +str(i+1) + ' = ' + str(models_folder) +'\\' + str(target_list[i]) + '.sav' +'\n')

    f.write('No_targets = ' + str(no_targets) +'\n')
    #for loop for targetname
    for i in range(int(no_targets)):
        f.write('target_name_' + str(i+1) + ' = ' + str(target_list[i]) + '\n')

    f.write('\n')

    #frame settings
    f.write('[Frame settings]\n')
    f.write('frames_dir_in = ' + str(input_folder) + '\n')
    f.write('frames_dir_out = ' + str(output_folder) + '\n')
    f.write('mm_per_pixel = '+'\n')
    f.write('distance_mm = 0'+'\n')
    f.write('\n')


    #line plot settings
    f.write('[Line plot settings]\n')
    f.write('Bodyparts =' + '\n')
    f.write('\n')

    #path plot settings
    f.write('[Path plot settings]\n')
    f.write('Deque_points = ' +'\n')
    f.write('Behaviour_points = ' + '\n')
    f.write('plot_severity = ' + '\n')
    f.write('severity_brackets = 10' + '\n')
    f.write('file_format = .bmp' + '\n')
    f.write('\n')

    #frame folder
    f.write('[Frame folder]\n')
    f.write('frame_folder = ' + str(frames_folder) + '\n')
    f.write('copy_frames = ' + 'yes' + '\n')
    f.write('\n')

    #distance plot
    f.write('[Distance plot]\n')
    f.write('POI_1 = ' + '\n')
    f.write('POI_2 = ' + '\n')
    f.write('\n')

    #create movie settings
    f.write('[Create movie settings]\n')
    f.write('file_format = ' + '\n')
    f.write('bitrate = ' + '\n')
    f.write('\n')

    #ensemble settings
    f.write('[create ensemble settings]\n')
    f.write('pose_estimation_body_parts = ' + str(bp) + '\n')
    f.write('model_to_run = RF\n')
    f.write('load_model =\n')
    f.write('data_folder = ' + str(csv_folder+'\\targets_inserted') + '\n')
    f.write('classifier =\n')
    f.write('train_test_size = 0.20 \n')
    f.write('under_sample_setting = \n')
    f.write('under_sample_ratio = \n')
    f.write('over_sample_setting = \n')
    f.write('over_sample_ratio = \n')
    f.write('RF_n_estimators = 6000 \n')
    f.write('RF_min_sample_leaf = 1 \n')
    f.write('RF_max_features = sqrt \n')
    f.write('RF_n_jobs = -1 \n')
    f.write('RF_criterion = entropy \n')
    f.write('RF_meta_data =  \n')
    f.write('generate_example_decision_tree =  \n')
    f.write('generate_classification_report =  \n')
    f.write('generate_features_importance_log =  \n')
    f.write('generate_features_importance_bar_graph =  \n')
    f.write('compute_permutation_importance =  \n')
    f.write('generate_learning_curve =  \n')
    f.write('generate_precision_recall_curve =  \n')
    f.write('N_feature_importance_bars  =  \n')
    f.write('GBC_n_estimators = \n')
    f.write('GBC_max_features = \n')
    f.write('GBC_max_depth = \n')
    f.write('GBC_learning_rate = \n')
    f.write('GBC_min_sample_split = \n')
    f.write('XGB_n_estimators = \n')
    f.write('XGB_max_depth = \n')
    f.write('XGB_learning_rate = \n')
    f.write('meta_files_folder = ' + project_folder + '\\configs\\ \n' )
    f.write('LearningCurve_shuffle_k_splits = 0 \n')
    f.write('LearningCurve_shuffle_data_splits = 0 \n')
    f.write('\n')

    #validation/run model
    f.write('[validation/run model]\n')
    f.write('generate_validation_video = ' + '\n')
    f.write('sample_feature_file = ' + '\n')
    f.write('save_individual_frames = ' + '\n')
    f.write('classifier_path = ' + '\n')
    f.write('classifier_name = ' + '\n')
    f.write('frames_dir_out_validation = ' + '\n')
    f.write('save_frames = ' + '\n')
    f.write('save_gantt = ' + '\n')
    f.write('discrimination_threshold = ' + '\n')
    f.write('\n')

#outliersettings
    f.write('[Outlier settings]\n')
    f.write('movement_criterion = \n')
    f.write('location_criterion = \n')
    f.close

    configfile = str(project_folder + "\\project_config.ini")

    return configfile


