import torch
import math
import os
import time
import copy
import numpy as np

from utils import evaluate_metrics
import logging
import os


class Trainer(object):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        args,
        lr_scheduler=None,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.best_path = os.path.join(self.args.log_dir, "best_model.pth")
        self.logger = logging.getLogger("DSTGCRN")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for _, (source, label) in enumerate(self.val_loader):
                source = source[..., : self.args.dataset.input_dim]
                label = label[..., : self.args.output_dim]
                output, _ = self.model(source)

                loss = self.loss(output.cpu(), label.cpu())
                total_val_loss += loss.item()

                all_labels.append(label.cpu().numpy())
                all_predictions.append(output.cpu().numpy())

        val_loss = total_val_loss / len(self.val_loader)
        self.logger.info(
            "**********Val Epoch {}: Validation loss: {:.6f}".format(epoch, val_loss)
        )

        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        all_labels = self.scaler.inverse_transform(all_labels)
        all_predictions = self.scaler.inverse_transform(all_predictions)

        # Evaluate metrics
        avg_metrics = {"MAE": 0, "MAPE": 0, "RMSE": 0, "RMSPE": 0, "R-squared": 0}

        for t in range(all_predictions.shape[1]):
            metrics = evaluate_metrics(
                all_labels[:, t, ...].reshape(all_labels.shape[0], -1),
                all_predictions[:, t, ...].reshape(all_predictions.shape[0], -1),
            )
            for metric, value in metrics.items():
                avg_metrics[metric] += value

        for metric, value in avg_metrics.items():
            avg_metrics[metric] /= all_predictions.shape[1]

        for metric, value in avg_metrics.items():
            self.logger.info(f"Validate_Average {metric}: {value}")

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (source, label) in enumerate(self.train_loader):
            source = source[..., : self.args.dataset.input_dim]
            label = label[..., : self.args.output_dim]
            self.optimizer.zero_grad()
            output, _ = self.model(source)
            loss = self.loss(output.cpu(), label.cpu())
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

            # Log training information
            if batch_idx % self.args.log_step == 0:
                self.logger.info(
                    "Train Epoch {}: {}/{} Loss: {:.6f}".format(
                        epoch, batch_idx, len(self.train_loader), loss.item()
                    )
                )
        train_epoch_loss = total_loss / len(self.train_loader)
        train_epoch_rmse = math.sqrt(train_epoch_loss)
        self.logger.info(
            "**********Train Epoch {}: Train Loss: {:.6f}, RMSE: {:.6f}".format(
                epoch,
                train_epoch_loss,
                train_epoch_rmse,
            )
        )

        return train_epoch_loss, train_epoch_rmse

    def train(self):
        best_model = None
        best_train_rmse = float("inf")
        best_loss = float("inf")
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss, train_epoch_rmse = self.train_epoch(epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            val_epoch_loss = self.val_epoch(epoch)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_train_rmse = train_epoch_rmse
                not_improved_count = 0
                best_state = True
                best_epoch_number = epoch
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.args.early_stop_patience)
                    )
                    break
            # save the best state
            if best_state == True:
                self.logger.info(
                    "*********************************Current best model saved!"
                )
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info(
            "Total training time: {:.4f}min, best training RMSE: {:.6f}, best validation loss: {:.6f}, epoch number: {}".format(
                (training_time / 60),
                best_train_rmse,
                best_loss,
                best_epoch_number,
            )
        )

        # save the best model to file
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        torch.save(best_model, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def test(model, args, test_loader, scaler, logger):
        model.eval()
        y_pred = []
        y_true = []
        adjs = []
        with torch.no_grad():
            for _, (source, label) in enumerate(test_loader):
                source = source[..., : args.dataset.input_dim]
                label = label[..., : args.output_dim]
                output, adjmatrices = model(source)
                adjs.append(adjmatrices.cpu().numpy())
                y_true.append(label.cpu().numpy())  # added cpu().numpy()
                y_pred.append(output.cpu().numpy())  # added cpu().numpy()

        # Apply the inverse transformation after converting tensors to numpy and concatenating
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

        # Evaluate metrics
        avg_metrics = {"MAE": 0, "MAPE": 0, "RMSE": 0, "RMSPE": 0, "R-squared": 0}

        for t in range(y_pred.shape[1]):
            metrics = evaluate_metrics(
                y_true[:, t, ...].reshape(y_pred.shape[0], -1),
                y_pred[:, t, ...].reshape(y_pred.shape[0], -1),
            )
            for metric, value in metrics.items():
                avg_metrics[metric] += value

        for metric, value in avg_metrics.items():
            avg_metrics[metric] /= y_pred.shape[1]

        for metric, value in avg_metrics.items():
            logger.info(f"Test_New_Average {metric}: {value}")

        logger.info(f"Lag: {args.lag}, Horizon: {args.horizon}")
