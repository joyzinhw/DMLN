import torch.nn as nn
from torchvision import transforms, datasets
import json
import argparse
import os
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from collections import Counter

from Res2Net import Res2Net

from train_utils import get_params_groups, create_lr_scheduler, train_one_epoch, evaluate 
import gc 
gc.collect()
torch.cuda.empty_cache()

from sklearn.metrics import confusion_matrix


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tb_writer = SummaryWriter(log_dir="/home/joyzinhw/Documentos/DMLN/tensorboard_logs")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 360)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

   
    train_dir = os.path.join("/home/joyzinhw/Documentos/DMLN/dataset-balanceado/train")
    val_dir = os.path.join("/home/joyzinhw/Documentos/DMLN/dataset-balanceado/test")


    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Diretório de treino não encontrado: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Diretório de validação não encontrado: {val_dir}")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])

    train_num = len(train_dataset)
    batch_size = args.batch_size
    num_workers = 2
    print(f'Using {num_workers} dataloader workers every process')
    
    print("Train class distribution:", Counter([s[1] for s in train_dataset.samples]))
    print("Val class distribution:", Counter([s[1] for s in val_dataset.samples]))



    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=num_workers
    )

    model = Res2Net(layers=[3, 4, 6, 3], num_classes=args.num_classes, width=16, scales=4, groups=1)
    model.to(device)

    pg = get_params_groups(model, weight_decay=args.wd)

    
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=10)

    best_acc = 0.
    best_train_acc = 0.
    best_kappa = 0.

    for epoch in range(args.epochs):
        train_loss, train_acc, train_kappa1, train_truee, train_predd = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler
        )
        print(f"train_kappa= {train_kappa1:.4f}")

        if train_acc > best_train_acc:
            conf_matrix = confusion_matrix(train_truee, train_predd)
            data = {'True label': train_truee, 'Predict label': train_predd}
            df = pd.DataFrame(data, columns=['True label', 'Predict label'])
            confmtpd = pd.crosstab(df['True label'], df['Predict label'], dropna=False)
            print(f"Confusion matrix with pandas:\n{confmtpd}")
            plt.figure()
            sn.heatmap(confmtpd, annot=True, cmap='Greens', fmt='d')
            plt.tight_layout()
            plt.show()
            plt.close()


            best_train_acc = train_acc

        val_loss, val_acc, val_kappa1, val_truee, val_predd = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )
        print(f"val_kappa= {val_kappa1:.4f}")

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "train_kappa", "val_kappa"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], train_kappa1, epoch)
        tb_writer.add_scalar(tags[6], val_kappa1, epoch)

        if best_acc < val_acc:
            save_path = "/home/joyzinhw/Documentos/DMLN/classfication_result/cancer_res2net.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

            val_conf_matrix = confusion_matrix(val_truee, val_predd)
            val_data = {'True label': val_truee, 'Predict label': val_predd}
            val_df = pd.DataFrame(val_data, columns=['True label', 'Predict label'])
            val_confmtpd = pd.crosstab(val_df['True label'], val_df['Predict label'], dropna=False)
            print(f"Confusion matrix with pandas:\n{val_confmtpd}")

            plt.figure()
            sn.heatmap(val_confmtpd, annot=True, cmap='Greens', fmt='d')
            plt.tight_layout()  # opcional, mas melhora layout
            plt.savefig('/home/joyzinhw/Documentos/DMLN/classfication_result/cancer_res2net_confusion_matrix.png')
            plt.close()



            best_acc = val_acc

        if best_kappa < val_kappa1:
            best_kappa = val_kappa1

        print(f"best_train_acc = {best_train_acc:.4f}")
        print(f"best_val_acc = {best_acc:.4f}")
        print(f"best_val_kappa = {best_kappa:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--data-path', type=str,
                        default="BMP_classification/")
    parser.add_argument('--weights', type=str,
                        default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)