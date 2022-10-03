from fvcore.common.config import CfgNode


_C = CfgNode()

_C.GPU_IDS = [0]
_C.MODE = 'training'
_C.EVAL_TYPE = 'proposal'
_C.DATASET = 'anet'
_C.USE_ENV = True
_C.USE_AGENT = True
_C.USE_OBJ = True
_C.EVAL_SCORE = 'AUC'

_C.TRAIN = CfgNode()
_C.TRAIN.SPLIT = 'training'
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.STEP_PERIOD = 1
_C.TRAIN.ATTENTION_STEPS = 1
_C.TRAIN.LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.CHECKPOINT_FILE_PATH = ''
_C.TRAIN.LOG_DIR = 'runs/c3d_runs/'

_C.VAL = CfgNode()
_C.VAL.SPLIT = 'validation'
_C.VAL.BATCH_SIZE = 32

_C.TEST = CfgNode()
_C.TEST.SPLIT = 'testing'
_C.TEST.BATCH_SIZE = 32
_C.TEST.CHECKPOINT_PATH = 'checkpoints/c3d_checkpoints/checkpoint_6/best_auc.pth'

_C.DATA = CfgNode()
_C.DATA.ANNOTATION_FILE = '../datasets/activitynet/annotations/activity_net.v1-3.min.json'
_C.DATA.DETECTION_GT_FILE = None
_C.DATA.ENV_FEATURE_DIR = '../datasets/activitynet/c3d_env_features/'
_C.DATA.AGENT_FEATURE_DIR = '../datasets/activitynet/c3d_agent_features/'
_C.DATA.OBJ_FEATURE_DIR = '../c3d_obj_features/'
_C.DATA.CLASSIFICATION_PATH = 'results/classification_results.json'
_C.DATA.RESULT_PATH = 'results/results.json'
_C.DATA.FIGURE_PATH = 'results/result_figure.jpg'
_C.DATA.TEMPORAL_DIM = 100
_C.DATA.MAX_DURATION = 100

_C.MODEL = CfgNode()
_C.MODEL.BOUNDARY_MATCHING_MODULE = 'bmn'
_C.MODEL.SCORE_PATH = 'checkpoints/c3d_checkpoints/scores.json'
_C.MODEL.CHECKPOINT_DIR = 'checkpoints/c3d_checkpoints/'
_C.MODEL.ATTENTION_HEADS = 4
_C.MODEL.ATTENTION_LAYERS = 1
_C.MODEL.AGENT_DIM = 2048
_C.MODEL.OBJ_DIM = 2048
_C.MODEL.ENV_DIM = 2048
_C.MODEL.FEAT_DIM = 512
_C.MODEL.TRANSFORMER_DIM = 1024
_C.MODEL.ENV_HIDDEN_DIM = None
_C.MODEL.AGENT_HIDDEN_DIM = None
_C.MODEL.OBJ_HIDDEN_DIM = None
_C.MODEL.HIDDEN_DIM_1D = 256  # 256
_C.MODEL.HIDDEN_DIM_2D = 128  # 128
_C.MODEL.HIDDEN_DIM_3D = 512  # 512
_C.MODEL.TOPK_AGENTS = 4

_C.BMN = CfgNode()
_C.BMN.NUM_SAMPLES = 32
_C.BMN.NUM_SAMPLES_PER_BIN = 3
_C.BMN.PROP_BOUNDARY_RATIO = 0.5

_C.BMN.POST_PROCESS = CfgNode()
_C.BMN.POST_PROCESS.USE_HARD_NMS = False
_C.BMN.POST_PROCESS.SOFT_NMS_ALPHA = 0.4
_C.BMN.POST_PROCESS.SOFT_NMS_LOW_THRESHOLD = 0.5
_C.BMN.POST_PROCESS.SOFT_NMS_HIGH_THRESHOLD = 0.9
_C.BMN.POST_PROCESS.HARD_NMS_THRESHOLD = 0.65
_C.BMN.POST_PROCESS.NUM_THREADS = 12
_C.BMN.POST_PROCESS.MAX_PROPOSALS = 100


def _assert_and_infer_cfg(cfg):
    assert cfg.TRAIN.BATCH_SIZE % len(cfg.GPU_IDS) == 0
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
