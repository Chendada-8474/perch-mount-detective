from torch.utils.data import Dataset
import torchvision.io as TVI
import torchvision.transforms as T


def create_dataset(task: list, media_type="img"):
    if media_type == "img":
        return ImageDataset(task)
    elif media_type == "vid":
        return VideoDataset(task)


class MediumDataset(Dataset):
    resize = T.Resize(size=[640, 640])

    def __init__(self, task: list) -> None:
        self.task = task
        super().__init__()

    def __len__(self):
        return len(self.task)


class ImageDataset(MediumDataset):
    def __getitem__(self, index):
        try:
            img = TVI.read_image(self.task[index]["path"])
            img = self.resize(img)
            self.task[index]["height"] = img.size(1)
            self.task[index]["width"] = img.size(2)

        except Exception as e:
            print(e)
            img = None
        return img, self.task[index]


class VideoDataset(MediumDataset):
    def __init__(self, task: list, interval=1) -> None:
        self.interval = interval
        super().__init__(task)

    def __getitem__(self, index):
        try:
            frames, _, info = TVI.read_video(self.task[index]["path"])
            interval_frames = int(self.interval * info["video_fps"])
            frames = [frames[i] for i in range(0, len(frames), interval_frames)]
            frames = [frame.permute(2, 0, 1) for frame in frames]
            frames = [self.resize(frame) for frame in frames]
            frames = [frame[0] for frame in frames]

        except Exception as e:
            print(e)
            frames = None
        return frames, self.task[index]
