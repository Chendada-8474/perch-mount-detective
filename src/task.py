import os
import json
import config as config

# from yolov5.utils.dataloaders import create_dataloader


class Task:
    batch = config.BATCH

    def __init__(self, path: str) -> None:
        self.path = path
        self._json = self._read_json(path)
        self.images = self._get_image_task(self._json["media"])
        self.videos = self._get_video_task(self._json["media"])

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            media = json.load(f)

        return media

    def tag_as_detected(self):
        des_path = os.path.join(config.DETECTED_TASK_DIR, self.basename)
        os.rename(self.path, des_path)

    def tag_as_error(self):
        des_path = os.path.join(config.ERROR_TASK_DIR, self.basename)
        os.rename(self.path, des_path)

    @property
    def basename(self):
        return os.path.basename(self.path)

    @property
    def section(self):
        return self._json["section"]

    def _get_image_task(self, media: list) -> list:
        return [
            medium
            for medium in media
            if medium["path"].split(".")[-1].lower() in config.IMAGE_EXTS
        ]

    def _get_video_task(self, media: list) -> list:
        return [
            medium
            for medium in media
            if medium["path"].split(".")[-1].lower() in config.VIDEO_EXTS
        ]


def read_task(path: str) -> Task:
    return Task(path)


if __name__ == "__main__":
    path = "D:/coding/demo_nas/task/測試棲架_2023-09-22.json"
    task = read_task(path)

    print(task.media_path)
