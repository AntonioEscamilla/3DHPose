"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
"""
import sys
import os.path as osp
from datetime import datetime

project_path = osp.abspath ( osp.join ( osp.dirname ( __file__ ), '..' ) )
if project_path not in sys.path:
    sys.path.insert ( 0, project_path )
CENTER_PATH = '/content/3DHPose/backend/CenterNet/src/lib'
sys.path.insert(0,CENTER_PATH)

#from backend.light_head_rcnn.person_detector import PersonDetector
from backend.tf_cpn.Detector2D import Detector2D
#from backend.ExtremeNet.dem_trial import PersonDetector
from backend.CenterNet.src.lib.detectors.detector_factory import detector_factory
from backend.CenterNet.src.lib.opts import opts
MODEL_PATH = '/content/3DHPose/backend/CenterNet/models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK,MODEL_PATH).split(' '))
detector  = detector_factory[opt.task](opt)


class Estimator_2d ( object ):

    def __init__(self, DEBUGGING=False):
        #detector de bounding boxes
        #self.bbox_detector = PersonDetector ( )
        #detector de pose
        self.pose_detector_2d = Detector2D ( show_image=DEBUGGING )

    def estimate_2d(self, img, img_id):
        #primeste o poza si sper eu sa returneze poza aia cu bounding box-uri
        data_dict = dict(data = img,image_id = img_id)
        #pentru light_head si extremeNet se ruleaza asta:
        #bbox_result = self.bbox_detector.detect ( img, img_id )
        #pentru center net se ruleaza asta:
        bbox_result = detector.run(img,data_dict)
        #print('am ajuns si aiki')
        #print('ce se intoarce din detectorul de obiecte 2D:' + str(bbox_result[1:-1]))
        #primeste o poza cu tot cu bounding boxuri, cropuie dupa bb si deseneaza scheletele din bb. de asemenea foloseste intai nms pe bb, pentru a pastra doar bb mai bune.
        start_time = datetime.now()
        dump_results = self.pose_detector_2d.detect ( bbox_result )
        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        #print('ce se intoarce din detectorul de poze 2D:' + str(dump_results))
        #sper ca returneaza imaginea, si pe ea detectiile 2D
        return dump_results


if __name__ == '__main__':
    import cv2

    img = cv2.imread ( 'datasets/Shelf/Camera0/img_000000.png' )
    est = Estimator_2d ()
    est.estimate_2d ( img, 0 )
