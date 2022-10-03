import numpy as np
import json
import pickle
from argparse import ArgumentParser


thumos_class = {
    7: 'BaseballPitch',
    9: 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}


def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data


def add_topk_detection(proposals, class_scores, class_names, k=2, max_proposals=50000):
    topk_indices = class_scores.argsort()[-k:][::-1]
    topk_scores = class_scores[topk_indices]

    detections = []
    for i in range(k):
        for proposal in proposals:
            detection = {'segment': proposal[:2].tolist()}
            detection['score'] = proposal[2] * topk_scores[i]
            detection['label'] = class_names[topk_indices[i]]
            detections.append(detection)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:max_proposals]
    return detections


def gen_detection(prop_file, cls_file, out_file):
    proposals = load_pkl(prop_file)
    class_names = [thumos_class[k] for k in thumos_class.keys()]
    class_ids = np.array([k - 1 for k in thumos_class.keys()])
    classifications = np.load(cls_file)
    classifications = classifications[:, class_ids]

    detections = {
        'version': 'THUMOS14',
        'external_data': 'used anet evaluation code',
        'results': {}
    }
    for video_name, results in proposals.items():
        video_id = int(video_name.split('_')[-1]) - 1
        class_scores = classifications[video_id]
        detections['results'][video_name] = add_topk_detection(results, class_scores, class_names)

    with open(out_file, 'w') as out:
        json.dump(detections, out)

    '''
    detections = {}
    for video_name, results in proposals.items():
        video_id = int(video_name.split('_')[-1]) - 1
        class_scores = classifications[video_id]
        detections[video_name] = add_topk_detection(results, class_scores, class_names)

    with open(out_file, 'w') as out:
        lines = []
        for video_name, dets in detections.items():
            for det in dets:
                line = [video_name] + det['segment'] + [det['label'], det['score']]
                lines.append(' '.join([str(x) for x in line]))
        out.write('\n'.join(lines))
    '''


def evaluate_detections(cfg, out_file='results/thumos_det.json', verbose=True, check_status=False):
    prop_file = cfg.DATA.RESULT_PATH
    cls_file = cfg.DATA.CLASSIFICATION_PATH
    gt_file = cfg.DATA.ANNOTATION_FILE if cfg.DATA.DETECTION_GT_FILE is None else cfg.DATA.DETECTION_GT_FILE
    split = cfg.VAL.SPLIT

    if out_file is None:
        out_file = prop_file
    print("Detection processing start")
    gen_detection(prop_file, cls_file, out_file)
    print("Detection processing finished")

    from evaluation_anet.eval_detection import ANETdetection
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=out_file,
        subset=split, tiou_thresholds=tious,
        verbose=verbose, check_status=check_status)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f}: {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = 'Detection: average-mAP {:.3f}.\n'.format(anet_detection.average_mAP * 100) + '\n'.join(mAP_at_tIoU)
    print(results)
    return anet_detection.average_mAP


def get_det_scores(prop_file, cls_file, gt_file, out_file=None, verbose=False, check_status=False):
    if out_file is None:
        out_file = prop_file
    print("Detection processing start")
    gen_detection(prop_file, cls_file, out_file)
    print("Detection processing finished")

    from evaluation_anet.eval_detection import ANETdetection
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=out_file,
        subset='testing', tiou_thresholds=tious,
        verbose=verbose, check_status=check_status)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f}: {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = 'Detection: average-mAP {:.3f}.\n'.format(anet_detection.average_mAP * 100) + '\n'.join(mAP_at_tIoU)
    print(results)
    return anet_detection.average_mAP


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--proposal-file', type=str, default='results/results.pkl')
    parser.add_argument('-c', '--classification-file', type=str, default='results/uNet_test.npy')
    parser.add_argument('-o', '--output-file', type=str, default='evaluation_thumos/detection_eval/detection_results.txt')
    parser.add_argument('-g', '--groundtruth-file', type=str, default='../datasets/thumos14/thumos_annotations/thumos_det_gt.json')
    args = parser.parse_args()

    get_det_scores(
        args.proposal_file,
        args.classification_file,
        args.groundtruth_file,
        args.output_file,
        verbose=True,
        check_status=True)
