import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import glob
import os
import pickle
import os.path as osp


def get_calib_data(dataset_root, num_cams):
    print(dataset_root)
    json_file = glob.glob(os.path.join(dataset_root, '*.json'))[0]

    with open(json_file) as f:
        data = json.load(f)

    multi_cam_calibration_data = []
    for i in range(num_cams):
        calibration_data = {}
        name = f'cam{i + 1}'
        calibration_data['name'] = name
        P = np.zeros((3, 4))
        P[:3, :3] = np.asarray(data['cameras'][name]['K'])
        calibration_data['width'] = data['cameras'][name]['image_size'][0]
        calibration_data['height'] = data['cameras'][name]['image_size'][1]
        calibration_data['P'] = P
        calibration_data['K'] = np.asarray(data['cameras'][name]['K'])
        calibration_data['D'] = np.asarray(data['cameras'][name]['dist']).reshape((5,))
        calibration_data['R'] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

        keys_list = sorted(data['camera_poses'])
        calibration_data['translation'] = np.asarray(data['camera_poses'][keys_list[int(name.split('cam')[1]) - 1]]['T'])
        calibration_data['Q'] = np.asarray(data['camera_poses'][keys_list[int(name.split('cam')[1]) - 1]]['R'])
        multi_cam_calibration_data.append(calibration_data)

    return multi_cam_calibration_data


def transform_calibration_data(multi_cam_calibration_data):
    # ----------- new R,T from chArUco board in ground plane -----------#
    R1w = np.asarray([[0.906657, -0.41233829, 0.08916402],
                      [-0.12746963, -0.46923809, -0.87382327],
                      [0.40214994, 0.78089228, -0.47799861]])
    T1w = np.asarray([[-0.31621033],
                      [0.99795041],
                      [2.26775274]])
    multi_cam_calibration_data[0]['translation'] = T1w
    multi_cam_calibration_data[0]['Q'] = R1w
    # add RT and P for mvpose compatibility
    multi_cam_calibration_data[0]['RT'] = np.column_stack((multi_cam_calibration_data[0]['Q'], multi_cam_calibration_data[0]['translation']))
    multi_cam_calibration_data[0]['P'] = multi_cam_calibration_data[0]['K'] @ multi_cam_calibration_data[0]['RT']

    # ----------- new R,T based on chArUco board in ground plane -----------#
    # read calibration data: cam1 as seen from cam2
    R21 = multi_cam_calibration_data[1]['Q']
    T21 = multi_cam_calibration_data[1]['translation']
    # obtain inverse transformation: cam2 as seen from cam 1
    T12 = cam_pose(R21, T21)
    R12 = R21.T
    # transform operation: ground plane as seen from cam2
    multi_cam_calibration_data[1]['translation'] = transform_translation(T1w, R12, T12).reshape(-1, 1)
    multi_cam_calibration_data[1]['Q'] = transform_rotation(R1w, R12)
    multi_cam_calibration_data[1]['RT'] = np.column_stack((multi_cam_calibration_data[1]['Q'], multi_cam_calibration_data[1]['translation']))
    multi_cam_calibration_data[1]['P'] = multi_cam_calibration_data[1]['K'] @ multi_cam_calibration_data[1]['RT']

    # ----------- new R,T based on chArUco board in ground plane -----------#
    # read calibration data: cam1 as seen from cam3
    R31 = multi_cam_calibration_data[2]['Q']
    T31 = multi_cam_calibration_data[2]['translation']
    # obtain inverse transformation: cam2 as seen from cam 1
    T13 = cam_pose(R31, T31)
    R13 = R31.T
    # transform operation: ground plane as seen from cam3
    multi_cam_calibration_data[2]['translation'] = transform_translation(T1w, R13, T13).reshape(-1, 1)
    multi_cam_calibration_data[2]['Q'] = transform_rotation(R1w, R13)
    multi_cam_calibration_data[2]['RT'] = np.column_stack((multi_cam_calibration_data[2]['Q'], multi_cam_calibration_data[2]['translation']))
    multi_cam_calibration_data[2]['P'] = multi_cam_calibration_data[2]['K'] @ multi_cam_calibration_data[2]['RT']

    # ----------- new R,T based on chArUco board in ground plane -----------#
    R41 = multi_cam_calibration_data[3]['Q']
    T41 = multi_cam_calibration_data[3]['translation']
    # obtain inverse transformation: cam2 as seen from cam 1
    T14 = cam_pose(R41, T41)
    R14 = R41.T
    # transform operation: ground plane as seen from cam2
    multi_cam_calibration_data[3]['translation'] = transform_translation(T1w, R14, T14).reshape(-1, 1)
    multi_cam_calibration_data[3]['Q'] = transform_rotation(R1w, R14)
    multi_cam_calibration_data[3]['RT'] = np.column_stack((multi_cam_calibration_data[3]['Q'], multi_cam_calibration_data[3]['translation']))
    multi_cam_calibration_data[3]['P'] = multi_cam_calibration_data[3]['K'] @ multi_cam_calibration_data[3]['RT']

    return multi_cam_calibration_data


def transform_translation(T_cam, R_newRef, T_NewRef):
    r = R.from_dcm(R_newRef)
    inv_r = r.inv()
    T_cam = T_cam.reshape((3,))
    T_NewRef = T_NewRef.reshape((3,))
    return inv_r.apply(T_cam - T_NewRef)


def transform_rotation(R_cam, R_newRef):
    R_cam = R.from_dcm(R_cam)
    R_newRef = R.from_dcm(R_newRef)
    inv_R_newRef = R_newRef.inv()
    out_quat = multiply_quaternions(inv_R_newRef.as_quat(), R_cam.as_quat())
    out_R = R.from_quat(out_quat)
    return out_R.as_dcm()


def multiply_quaternions(a, b):            # quat in form [x, y, z, w]
    return np.asarray([a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                       a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                       a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                       a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]])


def cam_pose(R, T):
    return np.matmul(-R.T, T)


if __name__ == '__main__':
    dump_dir = '../../datasets/Hexagonos/'
    multi_cam_data = get_calib_data(dump_dir, 4)
    multi_cam_data = transform_calibration_data(multi_cam_data)

    K = np.stack([multi_cam_data[i]['K'] for i in range(4)], axis=0).astype(np.float32)
    RT = np.stack([multi_cam_data[i]['RT'] for i in range(4)], axis=0).astype(np.float32)
    P = np.stack([multi_cam_data[i]['P'] for i in range(4)], axis=0)

    parameter_dict = {'K': K, 'P': P, 'RT': RT}

    with open(osp.join(dump_dir, 'camera_parameter.pickle'), 'wb') as f:
        pickle.dump(parameter_dict, f)

