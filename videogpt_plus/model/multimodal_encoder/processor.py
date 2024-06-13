from videogpt_plus.constants import IMAGE_RESOLUTION
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BaseProcessor:
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


class ImageTrainProcessor(BaseProcessor):
    def __init__(self, image_size=IMAGE_RESOLUTION, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def preprocess(self, item, return_tensors):
        return {'pixel_values': [self.transform(item)]}


class ImageEvalProcessor(BaseProcessor):
    def __init__(self, image_size=IMAGE_RESOLUTION, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def preprocess(self, item, return_tensors):
        return {'pixel_values': [self.transform(item)]}
