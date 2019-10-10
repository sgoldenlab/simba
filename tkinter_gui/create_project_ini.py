import os

def write_inifile(model_list,msconfig,project_path,project_name,no_targets,target_list,fpssettings,resolution_width,resolution_height):

############create directories################
    directory = project_path
    #generate main directories in project path
    project_folder = str(directory +'\\' + project_name + '\\project_folder')
    dlcmodels_folder = str(directory + '\\'+ project_name +'\\DLC_models')
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
    sml_behaviors_folder = str(csv_folder + '\\sml_behaviors')
    targets_inserted_folder = str(csv_folder + '\\targets_inserted')
    #frames
    input_folder = str(frames_folder + '\\input')
    output_folder = str(frames_folder + '\\output')

    folder_list = [project_folder,dlcmodels_folder,models_folder,config_folder,csv_folder,frames_folder,logs_folder,videos_folder,features_extracted_folder,input_csv_folder,machine_results_folder,outlier_corrected_movement_folder,outlier_corrected_location_folder,sml_behaviors_folder,targets_inserted_folder,input_folder,output_folder]

    for i in folder_list:
        try:
            # Create target Directory
            os.makedirs(i)
            print("Directory ", os.path.basename(i), " created ")
        except FileExistsError:
            print("Directory ", os.path.basename(i), " already exists")


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

    f.write('target_file = ' + str(csv_folder) + '\\sml_behaviours\\video_behaviours.xlsx' +'\n')

    #for loop for targetname
    for i in range(int(no_targets)):
        f.write('Model_path_chosen_' + str(i+1) + ' = ' + str(model_list[i]) + '\n')
    f.write('\n')


    #frame settings
    f.write('[Frame settings]\n')
    f.write('fps = ' + str(fpssettings) +'\n')
    f.write('resolution_width = ' + str(resolution_width) +'\n')
    f.write('resolution_height = ' + str(resolution_height) +'\n')
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

    #rf settings
    f.write('[RF settings]\n')
    f.write('n_estimators = 6000 \n')
    f.write('max_features = 60 \n')
    f.write('n_jobs = -1 \n')
    f.write('criterion = entropy \n')
    f.write('min_samples_leaf = 5 \n')
    f.write('test_size = 0.20 \n')
    f.write('model_evaluation_path =  \n')
    f.write('discrimination_threshold  = 0.49 \n')
    f.write('Feature_importances_to_plot = 15 \n')
    f.write('ensemble_method = RF \n')
    f.write('\n')

    #outliersettings
    f.write('[Outlier settings]\n')
    f.write('movement_criterion = \n')
    f.write('location_criterion = \n')
    f.close

    configfile = str(project_folder + "\\project_config.ini")

    return configfile


