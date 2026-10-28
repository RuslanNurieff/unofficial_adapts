import cv2
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import Split
from torch.utils.data import DataLoader

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def build_train_loader(data_root, category, img_size, batch_size=8, num_workers=4):
    args = DatasetArguments(
        dataset_path=data_root, img_size=img_size, gt_mask_size=img_size
    )
    dataset = MVTecDataset(args, category, Split.TRAIN)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def build_test_loader(data_root, category, img_size, batch_size=8, num_workers=4):
    args = DatasetArguments(
        dataset_path=data_root, img_size=img_size, gt_mask_size=img_size
    )
    dataset = MVTecDataset(args, category, Split.TEST)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
