from varitex.data.dataset_specifics import FFHQ


class CustomDataset:
    data = None
    indices = None
    N = None

    def __init__(self, opt, split=None, augmentation=None):
        self.opt = opt
        self.split = split if split is not None else self.opt.dataset_split
        self.augmentation = augmentation if augmentation is not None else self.opt.augmentation

        if self.opt.dataset.lower() == 'ffhq':
            self.initial_height, self.initial_width = FFHQ.image_height, FFHQ.image_width
            if self.augmentation:
                self.transform_params = FFHQ.get_transform_params(opt.transform_mode)
        else:
            raise NotImplementedError("Not implemented dataset '{}'".format(self.opt.dataset))
