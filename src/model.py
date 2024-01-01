import heapq
from tqdm import tqdm
from collections import Counter, defaultdict
import torch
import torchvision.transforms as T
import config


model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=config.DETECTIVE_MODEL_PATH,
)


def _class_to_taxon_order(class_: int):
    return config.TAXON_ORDER_TRANS_CUS_DK[class_]


def _predict(imgs):
    results = model(imgs).pandas().xyxy
    return [[dict(row) for _, row in result.iterrows()] for result in results]


def detect_image_task(image_loader):
    detected = []
    empty = []
    for imgs, metas in tqdm(image_loader):
        imgs = [T.ToPILImage()(img) for img in imgs]

        if not imgs:
            continue

        results = _predict(imgs)

        for i, result in enumerate(results):
            h, w = metas["height"][i].item(), metas["width"][i].item()

            meta = {
                "medium_id": metas["medium_id"][i],
                "medium_datetime": metas["medium_datetime"][i],
                "path": metas["path"][i],
                "nas_path": metas["nas_path"][i],
            }

            if not result:
                empty.append(meta)
                continue

            meta["individuals"] = []
            for row in result:
                meta["individuals"].append(
                    {
                        "taxon_order_by_ai": _class_to_taxon_order(row["class"]),
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

        results = _predict(frames)

        max_indi = config.MAXIMUN_IND_VID

        meta = {
            "medium_id": meta["medium_id"][0],
            "medium_datetime": meta["medium_datetime"][0],
            "path": meta["path"][0],
            "nas_path": meta["nas_path"][0],
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
                    "taxon_order_by_ai": _class_to_taxon_order(class_),
                    "xmin": None,
                    "xmax": None,
                    "ymin": None,
                    "ymax": None,
                }
            )
            max_indi -= 1

        detected.append(meta)

    return detected, empty
