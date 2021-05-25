import os
import csv
import re
import shutil
import pandas as pd

def write_dpkfile(projectdir,projectName):

    ######   create directories   ##########
    directory = projectdir
    dpkdir = os.path.join(directory,'logs','measures','dpk')
    project_name = projectName

    #generate main directories in project path
    annotation_folder = os.path.join(dpkdir, project_name, 'annotation_sets')
    models_folder = os.path.join(dpkdir, project_name, 'models')
    predictions_folder = os.path.join(dpkdir, project_name, 'predictions')
    video_folder = os.path.join(dpkdir, project_name, 'videos')
    video_input_folder = os.path.join(video_folder, 'input')
    video_output_folder = os.path.join(video_folder, 'output')

    folder_list =[annotation_folder,models_folder,predictions_folder,video_folder, video_input_folder,video_output_folder]

    #make main project folder
    mainproject = os.path.join(dpkdir,project_name)
    mainprojectconfigini = os.path.join(mainproject,'dpk_config.ini')
    mainplist = [dpkdir,mainproject]
    for i in mainplist:
        try:
            os.makedirs(i)
        except FileExistsError:
            pass

    ##make sub folder
    for i in folder_list:
        try:
            os.makedirs(i)
        except FileExistsError:
            pass

    # csv folder
    bpcsv = os.path.join(projectdir, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    skeletoncsv = pd.read_csv(bpcsv, names=["name", "parent", "swap"])
    skeletoncsv["parent"] = 'NaN'
    skeletoncsv["swap"] = 'NaN'
    skeletonSavePath = os.path.join(mainproject,'skeleton.csv')

    try:
        skeletoncsv.to_csv(skeletonSavePath, index=False)
    except:
        print('Skeleton.csv failed to generate')


    ####create text file ##################
    f = open(mainprojectconfigini,"w+")

    #general DPK Settings
    f.write('[general DPK settings]\n')
    f.write('project_folder = '+ str(mainproject) +'\n')

    #create annotation settings
    f.write('[create annotation settings]\n')
    f.write('frames_per_video = ' + '\n')
    f.write('annotation_output_name = ' +'\n')
    f.write('read_batch_size = ' + '\n')
    f.write('k_means_batch_size = ' + '\n')
    f.write('k_means_n_clusters = ' + '\n')
    f.write('k_means_max_iterations = ' + '\n')
    f.write('k_means_n_init = ' + '\n')
    f.write('\n')

    #annotator setings
    f.write('[annotator settings]\n')
    f.write('annotations_path = ' +'\n')
    f.write('skeleton_path = ' + str(skeletonSavePath) +'\n')

    #train model settings
    f.write('[train model settings]\n')
    f.write('annotations_path = ' +'\n')
    f.write('model_output_path = ' +'\n')
    f.write('augmenter= ' +'\n')
    f.write('downsampleFactor = ' +'\n')
    f.write('graph_scale = ' +'\n')
    f.write('validation_split = ' +'\n')
    f.write('sigma = ' +'\n')
    f.write('modelGrowthRate = ' +'\n')
    f.write('epochs = ' +'\n')
    f.write('model_batch_size = ' +'\n')
    f.write('validation_batch_size = ' +'\n')
    f.write('n_workers = ' +'\n')
    f.write('\n')

    #predict settings
    f.write('[predict settings]\n')
    f.write('model_path =' + '\n')
    f.write('videoPath = ' + str(video_folder) +'\n')
    f.write('skeleton_path = ' + str(skeletonSavePath) +'\n')
    f.write('outputFolder = ' + str(predictions_folder) +'\n')
    f.write('batch_size = 1' + '\n')
    f.write('\n')

    #visualize videos
    f.write('[visualize videos]\n')
    f.write('annotations_path = ' +'\n')
    f.write('predictions_folder = ' + '\n')
    f.write('inputVideofolder = ' + '\n')
    f.write('outputVideoFolder = 10' + '\n')
    f.write('\n')

    # model settings
    f.write('[StackedDenseNet/StackedHourglass settings]\n')
    f.write('n_stacks = ' + '\n')
    f.write('n_transitions = ' + str(video_folder) +'\n')
    f.write('growth_rate = ' + str(skeletonSavePath) +'\n')
    f.write('bottleneck_factor = ' + str(predictions_folder) +'\n')
    f.write('compression_factor =' + '\n')
    f.write('pretrained =' + '\n')
    f.write('subpixel =' + '\n')

    f.write('[DeepLabCut settings]\n')
    f.write('subpixel = ' + '\n')
    f.write('weights = ' + str(video_folder) + '\n')
    f.write('backbone = ' + str(skeletonSavePath) + '\n')
    f.write('alpha = ' + str(predictions_folder) + '\n')


    f.write('[LEAP settings]\n')
    f.write('filters = ' + '\n')
    f.write('upsampling_layers = ' + str(video_folder) + '\n')
    f.write('batchnorm = ' + str(skeletonSavePath) + '\n')
    f.write('pooling = ' + str(predictions_folder) + '\n')
    f.write('interpolation = ' + str(video_folder) + '\n')
    f.write('subpixel = ' + str(skeletonSavePath) + '\n')
    f.write('initializer = ' + str(predictions_folder) + '\n')

    f.close


    return mainprojectconfigini


