import numpy as np
import json
from argparse import ArgumentParser


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def add_topk_detection(proposals, class_scores, class_names, k=1):
    topk_indices = class_scores.argsort()[-k:][::-1]
    topk_scores = class_scores[topk_indices]

    detections = []
    for i in range(k):
        for proposal in proposals:
            detection = {'segment': proposal['segment']}
            detection['score'] = proposal['score'] * topk_scores[i]
            detection['label'] = class_names[topk_indices[i]]
            detections.append(detection)
    return detections


def gen_detection(prop_file, cls_file, out_file):
    proposals = load_json(prop_file)
    classifications = load_json(cls_file)
    class_names = classifications['class']
    detections = {
        'version': proposals['version'],
        'external_data': proposals['external_data'],
        'results': {}
    }

    for video_name, results in proposals['results'].items():
        class_scores = np.array(classifications['results'][video_name])
        detections['results'][video_name] = add_topk_detection(results, class_scores, class_names)

    with open(out_file, 'w') as out:
        json.dump(detections, out)


def evaluate_detections(cfg, out_file=None, verbose=False, check_status=True):
    prop_file = cfg.DATA.RESULT_PATH
    cls_file = cfg.DATA.CLASSIFICATION_PATH
    gt_file = cfg.DATA.ANNOTATION_FILE
    if out_file is None:
        out_file = prop_file
    print("Detection processing start")
    gen_detection(prop_file, cls_file, out_file)
    print("Detection processing finished")

    from evaluation_anet.eval_detection import ANETdetection
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=out_file,
        subset='validation', verbose=verbose, check_status=check_status)
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
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=out_file,
        subset='validation', verbose=verbose, check_status=check_status)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f}: {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = 'Detection: average-mAP {:.3f}.\n'.format(anet_detection.average_mAP * 100) + '\n'.join(mAP_at_tIoU)
    print(results)
    return anet_detection.average_mAP


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--proposal-file', type=str, default='results/results.json')
    parser.add_argument('-c', '--classification-file', type=str, default='results/classification_results.json')
    parser.add_argument('-o', '--output-file', type=str, default='results/detection_results.json')
    parser.add_argument('-g', '--groundtruth-file', type=str, default='../datasets/activitynet/annotations/activity_net.v1-3.min.json')
    args = parser.parse_args()

    get_det_scores(
        args.proposal_file,
        args.classification_file,
        args.groundtruth_file,
        args.output_file,
        verbose=True,
        check_status=True)
