import os
import json
import torch
import heapq
import logging
import requests
import src.media as media
import configs.config as config
import torchvision.transforms as T
from tqdm import tqdm
from src.task import read_task, Task
from torch.utils.data import DataLoader
from collections import defaultdict, Counter

logging.basicConfig(level=logging.DEBUG, filename="log/logging.log", encoding="utf-8")

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=config.DETECTIVE_MODEL_PATH,
)
model.conf = 0.5

headers = {"Content-type": "application/json", "Accept": "text/plain"}


def class_to_taxon_order(class_: int):
    return config.TAXON_ORDER_TRANS_CUS_DK[class_]


def predict(imgs):
    results = model(imgs).pandas().xyxy
    return [[dict(row) for _, row in result.iterrows()] for result in results]


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def detect_image_task(image_loader):
    detected = []
    empty = []
    for imgs, metas in image_loader:
        imgs = [T.ToPILImage()(img) for img in imgs]
        results = predict(imgs)

        for i, result in enumerate(tqdm(results)):
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
        results = predict(frames)

        counter = defaultdict(int)
        species_heap = []
        max_indi = config.MAXIMUN_IND_VID

        meta = {
            "medium_id": meta["medium_id"][0],
            "medium_datetime": meta["medium_datetime"][0],
            "path": meta["path"][0],
        }

        if not results:
            empty.append(meta)
            continue

        for result in results:
            for class_, count in Counter(row["class"] for row in result).items():
                counter[class_] = max(counter[class_], count)

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


def detect_task(task: Task):
    image_dataset = media.ImageDataset(task.images)
    video_dataset = media.VideoDataset(task.videos)

    image_loader = DataLoader(
        image_dataset,
        batch_size=config.BATCH,
        num_workers=2,
        collate_fn=collate_fn,
    )

    video_loader = DataLoader(
        video_dataset,
        batch_size=1,
        collate_fn=collate_fn,
    )

    if len(image_dataset):
        print("detecting image")
        detected_images, empty_images = detect_image_task(image_loader)

    if len(video_dataset):
        print("detecting video")
        detected_videos, empty_videos = detect_video_task(video_loader)

    detected = detected_images + detected_videos
    empty = empty_images + empty_videos

    requests.post(
        config.HOST + "/api/section/%s/detected_media" % task.section,
        data=json.dumps({"media": detected}),
        headers=headers,
    )

    requests.post(
        config.HOST + "/api/section/%s/empty_media" % task.section,
        data=json.dumps({"media": empty}),
        headers=headers,
    )

    task.tag_as_detected()


def main():
    task_paths = os.listdir(config.TASK_DIR)

    for file_name in task_paths:
        task_path = os.path.join(config.TASK_DIR, file_name)

        task = read_task(task_path)

        print(file_name)
        detect_task(task)

    print("detect end.")


if __name__ == "__main__":
    main()
