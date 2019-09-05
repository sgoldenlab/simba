import os
import shutil


def copy_multivideo_ini(inifile,source,filetype):
    print('Copying videos...')
    source = str(source)+'\\'
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest)+ '\\' + 'videos')
    files = []

    ########### FIND FILES ###########
    for i in os.listdir(source):
        if i.__contains__(str('.'+filetype)):
            files.append(i)

    for f in files:
        filetocopy=source +'\\'+f
        if os.path.exists(dest1+'\\'+f):
            print(f, 'already exist in', dest1)

        elif not os.path.exists(dest1+'\\'+f):
            shutil.copy(filetocopy, dest1)
            print(f, 'copied to', dest1)

    print('Finished copying videos!')

def copy_allcsv_ini(inifile,source):
    print('Copying csv files...')
    source = str(source)+'\\'
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest)+ '\\' + 'csv'+ '\\'+ 'input_csv')
    files = []
    print(dest1)
    print(source)
    ########### FIND FILES ###########
    for i in os.listdir(source):
        if i.__contains__(".csv"):
            files.append(i)

    for f in files:
        filetocopy=source +'\\'+f
        if os.path.exists(dest1+'\\'+f):
            print(f, 'already exist in', dest1)

        elif not os.path.exists(dest1+'\\'+f):
            shutil.copy(filetocopy, dest1)
            print(f, 'copied to', dest1)

    print('Finished copying csv files!')

def copy_singlevideo_ini(inifile,source):
    print('Copying video...')
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest) + '\\' + 'videos')

    if os.path.exists(dest1+'\\'+os.path.basename(source)):
        print(os.path.basename(source), 'already exist in', dest1)
    else:
        shutil.copy(source, dest1)
        print(os.path.basename(source),'copied to',dest1)

    print('Finished copying video!')

def copy_singlecsv_ini(inifile,source):
    print('Copying csv file...')
    dest = str(os.path.dirname(inifile))
    dest1 = str((dest) + '\\' + 'csv' + '\\' + 'input_csv')

    if os.path.exists(dest1+'\\'+os.path.basename(source)):
        print(os.path.basename(source), 'already exist in', dest1)
    else:
        shutil.copy(source, dest1)
        print(os.path.basename(source),'copied to',dest1)

    print('Finished copying csv file!')