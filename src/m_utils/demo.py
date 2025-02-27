import os
import os.path as osp
import pickle
import sys
import time

project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
#it's for the progress bars
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.evaluate import numpify
from src.m_utils.mem_dataset import MemDataset

#modelu vine din MultiEstimator(model_cfg)
#camera parameter vine din pickle
def export(model, loader, is_info_dicts=False, show=False):
    pose_list = list ()
    for img_id, imgs in enumerate ( tqdm ( loader ) ):
        try:
            pass
        except Exception as e:
            pass
            # poses3d = model.estimate3d ( img_id=img_id, show=False )
        #if-ul asta e pe true daca se foloseste treaba aia cu preprocesare

        if is_info_dicts:
            #practc in info_dicts is imaginile, si are si heatmeaps si cropped images pt fiecare persoana, salvate din preprocesare
            info_dicts = numpify ( imgs )

            model.dataset = MemDataset ( info_dict=info_dicts, camera_parameter=camera_parameter,
                                         template_name='Unified' )
            #nu inteleg de ce aici foloseste estimate si jos predict-is retardat
            #foloseste aici estimate pentru ca alea 2D sunt deja salvate, asa ca trebuie doar estimate alea 3D. la predict trebuie facut tot.
            poses3d = model._estimate3d ( 0, show=show )
        else:
            this_imgs = list ()
            #imgs e o lista cu 3 tensori.
            #un img_batch e un tensor din acestia 3, si are dimensiunea (288,360,3) si mai trebuie sa aflu de unde vine
            for img_batch in imgs:
                #in this_imgs se aduna acelasi lucru ca in imgs, dar fara prima lista care are dimensiunea [1], asta face squeeze
                this_imgs.append ( img_batch.squeeze ().numpy () )
            poses3d = model.predict ( imgs=this_imgs, camera_parameter=camera_parameter, template_name='Unified',
                                          show=show, plt_id=img_id )

        pose_list.append ( poses3d )
    return pose_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', nargs='+', dest='datasets', required=True,
                          choices=['Shelf', 'Campus', 'ultimatum1', 'Hexagonos'] )
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    parser.add_argument("-range", nargs="+", dest='range', type=int)
    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )
    for dataset_idx, dataset_name in enumerate ( args.datasets ):
        model_cfg.testing_on = dataset_name
        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            # you can change the test_rang to visualize different images (0~3199)
            test_range = range ( 605, 1800, 5)
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            # you can change the test_rang to visualize different images (0~1999)
            test_range = [i for i in range ( 605, 1000, 5 )]
            gt_path = dataset_path

        elif dataset_name == 'Hexagonos':
            dataset_path = model_cfg.hexagonos_path
            # you can change the test_rang to visualize different images (0~1999)
            test_range = [i for i in range(args.range[0], args.range[1], 1)]
            gt_path = dataset_path

        else:
            logger.error ( f"Unknown datasets name: {dataset_name}" )
            exit ( -1 )

        # read the camera parameter of this dataset
        with open ( osp.join ( dataset_path, 'camera_parameter.pickle' ),
                    'rb' ) as f:
            camera_parameter = pickle.load ( f )

        # using preprocessed 2D poses or using CPN to predict 2D pose
        if args.dumped_dir:
            test_dataset = PreprocessedDataset ( args.dumped_dir[dataset_idx] )
            logger.info ( f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation" )
        else:

            test_dataset = BaseDataset ( dataset_path, test_range )

        test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=6, shuffle=False )
        pose_in_range = export(test_model, test_loader, is_info_dicts=bool(args.dumped_dir), show=False )
        test_range_str = '_' + str(args.range[0]) + '_' + str(args.range[1])
        with open (osp.join(model_cfg.root_dir, 'result', model_cfg.testing_on + test_range_str + '.pkl' ), 'wb' ) as f:
            pickle.dump ( pose_in_range, f )


