import argparse
import os
import torch


def train(args):

    import library_check
    library_check.check(raise_exception=True)

    import data_prep_utils
    import models
    import train_utils
    if args.data_dir:
        data_prep_utils.set_root(args.data_dir)
    data_prep_utils.init()

    import torch

    opt = dict(num_workers=args.num_workers, pin_memory=args.pin_memory)

    train_data_cls_1 = data_prep_utils.rsna_pneumonia_detection_challenge.torch_classification_dataset(
        data_prep_utils.transforms.get_classification_train_transforms())
    train_loader_cls_1 = torch.utils.data.DataLoader(  # type: ignore
        train_data_cls_1,
        sampler=data_prep_utils.samplers.ImbalancedDatasetSampler(train_data_cls_1),
        batch_size=args.classification_batch_size, **opt
    )

    train_data_cls_2 = data_prep_utils.covid_19_radiography_dataset.torch_classification_dataset(
        data_prep_utils.transforms.get_classification_train_transforms())
    train_loader_cls_2 = torch.utils.data.DataLoader(  # type: ignore
        train_data_cls_2,
        sampler=data_prep_utils.samplers.ImbalancedDatasetSampler(train_data_cls_2),
        batch_size=args.classification_batch_size, **opt
    )

    train_loader_det = data_prep_utils.rsna_pneumonia_detection_challenge.torch_detection_dataset(
        data_prep_utils.transforms.get_detection_train_transforms(),
        batch_size=args.detection_batch_size,
        shuffle=True, **opt
    )

    train_loaders = [
        train_loader_cls_1,
        train_loader_cls_2,
        train_loader_det,
    ]

    efficientnet_backbone = models.get_efficientnet_backbone(depth=4, in_channels=1, image_size=256, pretrained=True)
    model_c = models.get_classifier(efficientnet_backbone)
    model_d = models.get_detector(efficientnet_backbone)
    models.freeze_backbone_gradient(efficientnet_backbone)

    opt_c = torch.optim.AdamW(model_c.parameters(), lr=0.00005)
    opt_d = torch.optim.AdamW(model_d.parameters(), lr=0.0001)

    with train_utils.AdvancedFitter(
        model_c, model_d, opt_c, opt_d, args.epoch,
        train_iter=train_loaders, val_iter=train_loaders,
        snapshot_dir=args.checkpoint_dir,
        verbose=True, timer=True, log_interval=args.log_interval,
    ).to(torch.device(args.device)) as fitter:
        fitter.fit()

    if args.save_model:
        torch.save({'classifier': model_c.state_dict(), 'detector': model_d.state_dict()}, args.save_path)


parser = argparse.ArgumentParser(prog='train.py', description='Script for Pytorch Model Training.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch', type=int, default=100,
                    help='How many epochs to train')
parser.add_argument('--classification_batch_size', default=32,
                    help='How many batches used in classification training')
parser.add_argument('--detection_batch_size', default=3,
                    help='How many batches used in detection training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='The number of dataloader work processes')
parser.add_argument('--log_interval', type=int, default=1,
                    help='How many batches before logging training status')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Pytorch device')
parser.add_argument('--pin_memory', action='store_true', default=False,
                    help='Whether to initialize dataloader with memory pinned')

parser.add_argument('--data_dir', type=str, default='',
                    help='Path of dataset root directory')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                    help='Path of checkpoint directory')
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--save_path', type=str, default=os.path.join('checkpoint', 'model_weight.pt'),
                    help='Path to save model')
default_args = parser.parse_args([])

del argparse, os, torch


if __name__ == '__main__':
    train(parser.parse_args())
