import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import  load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    print(model)
    model = model.to(device)
    model.train()
    print(next(model.parameters()).device)

    train_data = load_data("drive_data/train", shuffle=True, transform_pipeline="default", batch_size=batch_size, num_workers=8)
    val_data = load_data("drive_data/val", shuffle=False, num_workers=4)

    # create loss function and optimizer
    class_weights = torch.FloatTensor([0.1, 0.45, 0.45]).to(device)
    segmentation_loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    estimation_loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = torch.optim.Adam()

    global_step = 0
    metrics = {"train_acc": DetectionMetric(), "val_acc": DetectionMetric()}
    max_iou = 0.0

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].reset()

        model.train()

        for batch in train_data:
            #batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            image = batch["image"]
            track = batch["track"]
            depth = batch["depth"]
            image, track, depth = image.to(device), track.to(device), depth.to(device)

            # TODO: implement training step
            logits, pred_depth = model(image)
            #print(logits.shape)
            #print(track.shape)
            #print(pred_depth.shape)
            #print(logits.device)
            #print(track.device)

            seg_loss = segmentation_loss_func(logits, track)
            est_loss = estimation_loss_func(pred_depth, depth)
            loss = (0.5 * seg_loss) + (0.5 * est_loss)
            metrics["train_acc"].add(logits.argmax(dim=1), track, pred_depth, depth)
            logger.add_scalar('train_loss', loss, global_step)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                image = batch["image"]
                track = batch["track"]
                depth = batch["depth"]
                image, track, depth = image.to(device), track.to(device), depth.to(device)

                # TODO: compute validation accuracy
                test_seg, test_depth = model.predict(image)
                metrics["val_acc"].add(test_seg, track, test_depth, depth)


        # log average train and val accuracy to tensorboard
        epoch_train_accuracy = (metrics["train_acc"].compute()["accuracy"])
        epoch_train_iou = (metrics["train_acc"].compute()["iou"])
        epoch_train_abs_depth_error = (metrics["train_acc"].compute()["abs_depth_error"])
        epoch_train_tp_depth_error = (metrics["train_acc"].compute()["tp_depth_error"])

        epoch_val_accuracy = (metrics["val_acc"].compute()["accuracy"])
        epoch_val_iou = (metrics["val_acc"].compute()["iou"])
        epoch_val_abs_depth_error = (metrics["val_acc"].compute()["abs_depth_error"])
        epoch_val_tp_depth_error = (metrics["val_acc"].compute()["tp_depth_error"])

        logger.add_scalar('train_accuracy', (epoch_train_accuracy), global_step)
        logger.add_scalar('train_iou', epoch_train_iou, global_step)
        logger.add_scalar('train_abs_depth_error', epoch_train_abs_depth_error, global_step)
        logger.add_scalar('train_tp_depth_error', epoch_train_tp_depth_error, global_step)

        logger.add_scalar('val_accuracy', epoch_val_accuracy, global_step)
        logger.add_scalar('val_iou', epoch_val_iou, global_step)
        logger.add_scalar('val_abs_depth_error', epoch_val_abs_depth_error, global_step)
        logger.add_scalar('val_tp_depth_error', epoch_val_tp_depth_error, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                #f"train_acc={epoch_train_accuracy:.4f} "
                f"train_iou={epoch_train_iou:.4f} "
                #f"train_abs_depth_error={epoch_train_abs_depth_error:.4f} "
                f"train_tp_depth_error={epoch_train_tp_depth_error:.4f} \n"

                #f"val_acc={epoch_val_accuracy:.4f} "
                f"val_iou={epoch_val_iou:.4f} "
                #f"val_abs_depth_error={epoch_val_abs_depth_error:.4f} "
                f"val_tp_depth_error={epoch_val_tp_depth_error:.4f} "
            )
            if(epoch_val_iou > max_iou):
                max_iou = epoch_val_iou
                torch.save(model.state_dict(), log_dir / f"{model_name}-earlystop.th")
                print(f"Model saved to {log_dir / f'{model_name}-earlystop.th'}")

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    # save and overwrite the model in the root directory for grading
    model.load_state_dict(torch.load(log_dir / f"{model_name}-earlystop.th", weights_only=True))
    model.eval()
    save_model(model)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    # optional: additional model hyperparamters
    #parser.add_argument("--num_blocks", type=int, default=1)
    # pass all arguments to train
    train(**vars(parser.parse_args()))
