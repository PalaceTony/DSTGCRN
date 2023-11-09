import torch
import torch.nn as nn
from DSTGCRN import DSTGCRN as Network
from trainer import Trainer
from lib.dataloader import get_dataloader
import hydra
from omegaconf import DictConfig
import logging

import random
import numpy as np

logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path="../../configuration/modules",
    config_name="DSTGCRN",
)
def main(args: DictConfig) -> None:
    # Random seed
    random.seed(args.seed)
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = "cpu"

    print(args.device)

    # Logging
    logging.info(f"Embed Dimension: {args.embed_dim}")
    logging.info(f"Number of Layers: {args.num_layers}")
    logging.info(f"RNN Units: {args.rnn_units}")
    logging.info(f"Chebyshev Polynomial Order: {args.cheb_k}")
    logging.info(f"Initial Learning Rate: {args.lr_init}")
    # logging.info(f"Number of Heads: {args.num_heads}")
    logging.info(f"Hidden Dimension of Node: {args.hidden_dim_node}")
    logging.info(f"Number of Layers of Node: {args.num_layers_node}")
    # logging.info(f"Input Dimension: {args.input_dim}")

    # Number of parameters
    model = Network(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    logging.info("Model parameters:")
    for name, param in model.named_parameters():
        logging.info(f"{name}: {param.shape}")
    logging.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # load dataset
    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
    ) = get_dataloader(
        args,
        normalizer=args.normalizer,
        single=True,
    )

    # init loss function, optimizer
    loss = torch.nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False,
    )

    # learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print("Applying learning rate decay.")
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(","))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate
        )

    # Load the checkpoint
    if args.saved_model_path is not None:
        logging.info("Loading saved model from {}".format(args.saved_model_path))
        model.load_state_dict(torch.load(args.saved_model_path))
        model.to(args.device)
        # testing the model
        logging.info("Testing the model.")

        trainer = Trainer(
            model,
            loss,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            scaler,
            args,
            lr_scheduler=lr_scheduler,
        )

        test_loss = trainer.test(
            model, trainer.args, trainer.test_loader, trainer.scaler, trainer.logger
        )
        return test_loss
    else:
        # Train from the start
        trainer = Trainer(
            model,
            loss,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            scaler,
            args,
            lr_scheduler=lr_scheduler,
        )
        trainer.train()  # test phase involved in the train


if __name__ == "__main__":
    main()
