import os
import json
import torch
import heapq
import logging
import config
import src.media as media
import torchvision.transforms as T
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from src.utils import get_run_hours
from src.model import model, detect_image_task, detect_video_task
from src.task import read_task, Task

ROOT = os.path.dirname(__file__)

FORMAT = "%(asctime)s %(filename)s %(levelname)s:%(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(ROOT, "log/logging.log"),
    encoding="utf-8",
    format=FORMAT,
)


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))

    return torch.utils.data.dataloader.default_collate(batch) if batch else [[], []]


def load_dataset(task: Task) -> (DataLoader, DataLoader):
    image_dataset = media.ImageDataset(task.images)
    video_dataset = media.VideoDataset(task.videos)

    image_loader = DataLoader(
        image_dataset,
        batch_size=config.BATCH,
        num_workers=0,
        collate_fn=collate_fn,
    )

    video_loader = DataLoader(
        video_dataset,
        batch_size=1,
        collate_fn=collate_fn,
    )
    return image_loader, video_loader


def detect_task(image_loader: DataLoader, video_loader: DataLoader) -> (list, list):
    detected_images, empty_images = [], []
    detected_videos, empty_videos = [], []

    detected_images, empty_images = detect_image_task(image_loader)
    detected_videos, empty_videos = detect_video_task(video_loader)

    detected = detected_images + detected_videos
    empty = empty_images + empty_videos

    return detected, empty


def to_results(detected: list, empty: list, task: Task) -> dict:
    results = {}
    results["section"] = task.section
    results["detected"] = detected
    results["empty"] = empty
    return results


def save_results_to_json(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f)


def main():
    print("排程辨識開始，請勿關閉此視窗。")
    logging.info("detect schedule start")

    run_hours = timedelta(hours=get_run_hours())
    start_time = datetime.now()

    task_filenames = os.listdir(config.TASK_DIR)

    for file_name in task_filenames:
        print(file_name)

        task_path = os.path.join(config.TASK_DIR, file_name)

        task = read_task(task_path)

        image_loader, video_loader = load_dataset(task)
        detected, empty = detect_task(image_loader, video_loader)
        results = to_results(detected, empty, task)

        save_results_to_json(results, config.DETECTED_TASK_DIR)
        logging.info("%s done" % task.basename)

        if datetime.now() - start_time > run_hours:
            break

    print("detect end.")
    logging.info("detect end")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
