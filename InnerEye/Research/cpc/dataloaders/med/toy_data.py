import torch
import torchvision


class FakeData(torchvision.datasets.VisionDataset):
    def __init__(self, size=10, image_size=(1, 224, 224, 224), transform=None, n_classes=2):
        super(FakeData, self).__init__(None, transform=transform)
        self.size = size
        self.image_size = image_size
        self.n_classes = n_classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            Tensor: image
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        img = torch.empty(*self.image_size, dtype=torch.float32)
        labels = torch.randint(0, self.n_classes, size=[1])
        if self.transform is not None:
            img = self.transform(img)
        return {
            "images": img,
            "labels": labels,
            "features": torch.empty([self.image_size[0], 28])
        }

    def __len__(self):
        return self.size


def get_toy_med_data_loader(batch_size,
                            num_workers,
                            transforms=None,
                            size=200,
                            image_size=(1, 320, 32, 320)):
    """
    TODO
    """
    # Create fake unlabelled MED dataset
    TOY_DATASET = FakeData(size=size,
                           image_size=image_size,
                           transform=transforms)

    data_loader = torch.utils.data.DataLoader(
        TOY_DATASET,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return data_loader
