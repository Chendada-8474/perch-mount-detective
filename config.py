from datetime import datetime, timedelta

TASK_DIR = "D:/coding/demo_nas/tasks"
WAIT_UPLOADED_DIR = "D:/coding/demo_nas/wait-upload-tasks"
TASK_TRASH_CAN = "D:/coding/demo_nas/task-trash-can"
ERROR_TASK_DIR = "D:/coding/demo_nas/error_tasks"
# TASK_DIR = "D:/perch-mount-system/tasks"
# WAIT_UPLOADED_DIR = "D:/perch-mount-system/wait-upload-tasks"
# TASK_TRASH_CAN = "D:/perch-mount-system/task-trash-can"
# ERROR_TASK_DIR = "D:/perch-mount-system/error_tasks"

DETECTIVE_MODEL_PATH = "model/custom_detectivekite.pt"

IMAGE_EXTS = {"bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng"}
VIDEO_EXTS = {"mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"}

MODEL_CONF = 0.7

BATCH = 8

MAXIMUN_IND_VID = 3

TODAY = datetime.strftime(datetime.today(), "%Y-%m-%d")

DETECT_RUN_HOURS = 9
WEEKEND_RUN_HOURS = 13

TAXON_ORDER_TRANS_CUS_DK = {
    0: 7575,
    1: 23994,
    2: 20535,
    3: 19938,
    4: 22770,
    5: 9547,
    6: 24167,
    7: 2060,
    8: 20940,
    9: 30954,
    10: 20550,
    11: 28999,
    12: 29021,
    13: 29021,
    14: 28999,
    15: 27295,
    16: 27278,
    17: 31070,
    18: 2065,
    19: 8410,
    20: 3697,
    21: 9073,
    22: 8327,
}
