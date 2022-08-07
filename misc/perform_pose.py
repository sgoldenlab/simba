import cv2
from tools.misc import load_config_yaml
from pylab import *
from dlclive import DLCLive, Processor
from tools.misc import read_bp_config_csv
from tools.determine_fps import determine_fps
from multiprocessing import shared_memory
from tools.image_manipulations import change_color, change_size
from multiprocessing.shared_memory import ShareableList, SharedMemory
from tools.unit_tests.check_shared_memory_processes import create_shared_memory_process

class CamPoseInitalizer():
    def __init__(self):
        self.shared_status = shared_memory.ShareableList(name='pose_status')
        self.shared_latency = shared_memory.ShareableList(name='latency_data')
        self.config = load_config_yaml(self.shared_status[0])
        if not self.config['CAMERA']['IP CAMERA']['status']:
            if not self.config['CAMERA']['CAMERA SETTINGS']['VIDEO FILE']['use video file']:
                self.id = int(self.config['CAMERA']['CAMERA SETTINGS']['input channel'])
            else:
                self.id = self.config['CAMERA']['CAMERA SETTINGS']['VIDEO FILE']['video path']
        self.bp_df, self.animal_names, self.body_part_names, _ = read_bp_config_csv(
            self.config['POSE']['MODEL']['bp config path'])
        self.no_animals = len(set(self.animal_names))
        self.no_bps = len(set(self.body_part_names))
        self.first_image = None
        self.session_name = self.config['GENERAL']['session name']
        self.custom_img_size = self.config['CAMERA']['IMAGE']['CUSTOM IMG SIZE']['status']
        self.modify_img = self.config['CAMERA']['IMAGE']['MODIFY_IMG']['status']
        self.bp_array = np.empty((self.no_animals, self.no_bps * self.no_animals, 3))
        self.shared_status[-1] += 1

    def initialize_pose(self):
        self.cap = cv2.VideoCapture(self.id)
        self.img_size = dict()
        self.dlc_live_object = DLCLive(self.config['POSE']['MODEL']['model path'], processor=Processor())
        while self.first_image is None:
            _, self.first_image = self.cap.read()
        self.dlc_live_object.init_inference(self.first_image)
        self.fps = determine_fps(self)
        self.img_size['width'] = self.first_image.shape[0]
        self.img_size['height'] = self.first_image.shape[1]
        self.img_size['color'] = self.first_image.shape[2]
        self.shared_status[-1] += 1
        self.shared_latency[0] = self.fps


    def perform_pose(self):
        self.dlc_live_object.init_inference(self.first_image)
        self.shared_status[-1] += 1
        self.bp_array = np.empty((self.no_animals, self.no_bps * self.no_animals, 3))

        self.shm_bp = create_shared_memory_process(shared_memory_name=self.session_name, shared_memory_size=self.bp_array.nbytes * 2000)
        self.shm_img = create_shared_memory_process(shared_memory_name='shared_img',shared_memory_size=self.first_image.nbytes)
        frame_counter, start_time, session_timer = 0, time.time(), 0
        self.shared_status[-1] += 1

        while True:
            captured, np_frame = self.cap.read()
            if captured:
                frame_counter += 1
                if self.custom_img_size:
                    np_frame = change_size(np_frame, self.config['CAMERA']['IMAGE']['CUSTOM IMG SIZE'])
                if self.modify_img:
                    method = self.config['CAMERA']['IMAGE']['MODIFY_IMG']['color']
                    np_frame = change_color(np_frame, method)
                frame_pose_results = self.dlc_live_object.get_pose(np_frame)
                self.bp_array = np.concatenate((self.bp_array, [frame_pose_results]))
                if self.bp_array.shape[0] >= 20:
                    self.bp_array = self.bp_array[-20:,:]
                shared_array = np.ndarray(self.bp_array.shape, dtype=self.bp_array.dtype, buffer=self.shm_bp.buf)
                shared_img = np.ndarray(np_frame.shape, dtype=np_frame.dtype, buffer=self.shm_img.buf)
                shared_img[:] = np_frame[:]
                shared_array[:] = self.bp_array[:]
                current_fps = round(frame_counter / (time.time() - start_time), 2)
                self.shared_latency[0] = current_fps
                self.shared_latency[1] = frame_counter
            else:
                print('No camera feed detected.')

if __name__ == "__main__":
    pose_session = Cam_Pose_Instance()
    pose_session.initialize_pose()
    pose_session.perform_pose()
