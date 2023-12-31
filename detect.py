import os
import json
import torch
import heapq
import logging
import requests
import config
import src.media as media
import torchvision.transforms as T
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from src.task import read_task, Task
from torch.utils.data import DataLoader
from collections import defaultdict, Counter


ROOT = os.path.dirname(__file__)

FORMAT = "%(asctime)s %(filename)s %(levelname)s:%(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(ROOT, "log/logging.log"),
    encoding="utf-8",
    format=FORMAT,
)

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=config.DETECTIVE_MODEL_PATH,
)
model.conf = 0.5

headers = {"Content-type": "application/json", "Accept": "text/plain"}
TZ = timezone(timedelta(hours=8))


def get_run_hours():
    week_day = datetime.now(tz=TZ).weekday()
    return config.WEEKEND_RUN_HOURS if 4 <= week_day < 6 else config.DETECT_RUN_HOURS


def class_to_taxon_order(class_: int):
    return config.TAXON_ORDER_TRANS_CUS_DK[class_]


def predict(imgs):
    results = model(imgs).pandas().xyxy
    return [[dict(row) for _, row in result.iterrows()] for result in results]


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))

    return torch.utils.data.dataloader.default_collate(batch) if batch else [[], []]


def detect_image_task(image_loader):
    detected = []
    empty = []
    for imgs, metas in tqdm(image_loader):
        imgs = [T.ToPILImage()(img) for img in imgs]

        if not imgs:
            continue

        results = predict(imgs)

        for i, result in enumerate(results):
            h, w = metas["height"][i].item(), metas["width"][i].item()

            meta = {
                "medium_id": metas["medium_id"][i],
                "medium_datetime": metas["medium_datetime"][i],
                "path": metas["path"][i],
            }

            if not result:
                empty.append(meta)
                continue

            meta["individuals"] = []
            for row in result:
                meta["individuals"].append(
                    {
                        "taxon_order_by_ai": class_to_taxon_order(row["class"]),
                        "xmin": row["xmin"] / w,
                        "xmax": row["xmax"] / w,
                        "ymin": row["ymin"] / h,
                        "ymax": row["ymax"] / h,
                    }
                )
            detected.append(meta)

    return detected, empty


def detect_video_task(video_loader):
    detected = []
    empty = []
    for frames, meta in tqdm(video_loader):
        frames = [T.ToPILImage()(frame) for frame in frames]

        if not frames:
            continue

        results = predict(frames)

        max_indi = config.MAXIMUN_IND_VID

        meta = {
            "medium_id": meta["medium_id"][0],
            "medium_datetime": meta["medium_datetime"][0],
            "path": meta["path"][0],
        }

        results = [row for row in results if row]
        if not results:
            empty.append(meta)
            continue

        counter = defaultdict(int)
        for result in results:
            for class_, count in Counter(row["class"] for row in result).items():
                counter[class_] = max(counter[class_], count)

        species_heap = []
        for result in results:
            for row in result:
                heapq.heappush(species_heap, (-row["confidence"], row["class"]))

        meta["individuals"] = []

        while max_indi and species_heap:
            _, class_ = heapq.heappop(species_heap)
            if not counter[class_]:
                continue
            counter[class_] -= 1
            meta["individuals"].append(
                {
                    "taxon_order_by_ai": class_to_taxon_order(class_),
                    "xmin": None,
                    "xmax": None,
                    "ymin": None,
                    "ymax": None,
                }
            )
            max_indi -= 1

        detected.append(meta)

    return detected, empty


def batch_upload(task: Task, data: list, mode="detected"):
    if mode == "detected":
        url = config.HOST + "/api/section/%s/schedule_detect" % task.section
    else:
        url = config.HOST + "/api/section/%s/empty_media" % task.section

    batch_index = list(range(0, len(data), 500)) + [len(data)]

    for i in range(len(batch_index) - 1):
        s, e = batch_index[i], batch_index[i + 1]

        batch_data = data[s:e]
        res = requests.post(
            url,
            data=json.dumps({"media": batch_data}),
            headers=headers,
        )

        if res.status_code != 200:
            task.tag_as_error()
            with open(f"error_{mode}_{task.basename}", "w", encoding="utf-8") as f:
                json.dump(data[s:], f)

            logging.error(
                "%s error when post %s media [%s:%s]" % (task.basename, mode, s, e)
            )
            return


def save_results_to_json(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f)


def detect_task(task: Task) -> dict:
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

    detected_images, empty_images = [], []
    detected_videos, empty_videos = [], []
    if len(image_dataset):
        print("detecting image")
        detected_images, empty_images = detect_image_task(image_loader)

    if len(video_dataset):
        print("detecting video")
        detected_videos, empty_videos = detect_video_task(video_loader)

    detected = detected_images + detected_videos
    empty = empty_images + empty_videos

    results = {}
    results["section"] = task.section
    results["detected"] = detected
    results["empty"] = empty

    return results


def main():
    print("排程辨識開始，請勿關閉此視窗。")
    logging.info("detect schedule start")

    run_hours = timedelta(hours=get_run_hours())
    start_time = datetime.now()

    task_paths = os.listdir(config.TASK_DIR)

    for file_name in task_paths:
        task_path = os.path.join(config.TASK_DIR, file_name)

        task = read_task(task_path)

        print(file_name)
        results = detect_task(task)
        save_results_to_json(results, config.DETECTED_TASK_DIR)
        logging.info("%s done" % task.basename)

        if datetime.now() - start_time > run_hours:
            break

    print("detect end.")
    logging.info("detect end")


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("error")
