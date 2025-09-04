import os
import sys
import json
import torch
import random
import datetime
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from models.fid_gr_modules import FiDGRModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_model(args):
    if args.fid:
        print(f"@@@ Using FiDGRModel")
        return FiDGRModel(args)

def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    model = set_model(args)
    trainer = pl.Trainer(**train_params)
    print(train_params)
    #print trainer parameters
    print(trainer.max_steps)
    if args.do_train:
        now = datetime.datetime.now()
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Training...")
        if args.resume_from_checkpoint is None:
            trainer.fit(model)
        else:
            print(f"@@@ Resume Training from {args.resume_from_checkpoint}")
            trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
        now = datetime.datetime.now()
        print(
            f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Training..."
        )
    #if args.do_test:
    #    now = datetime.datetime.now()
    #    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Testing...")
    #    #raise NotImplementedError
    #    trainer.test(model)
    #    now = datetime.datetime.now()
    #    print(
    #        f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Testing... "
    #    )
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str)
    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument("--machine", default='lg', type=str)
    parser.add_argument("--exp_mode", default='train', type=str)
    parser.add_argument("--eval_mode", default='1', type=str)
    parser.add_argument("--test_model_path", default='', type=str) # You may just write the directory and not the exact tfmr num. We'll find the correct tfmr for you.
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--from_model_path", default=None, type=str)
    parser.add_argument("--dataset", type=str, default='nq')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--peft", action='store_true')
    parser.add_argument("--fid", action='store_true')
    parser.add_argument("--sortfid", action='store_true')
    parser.add_argument("--run_3b", action='store_true')
    parser.add_argument("--load_from_fid", action='store_true')
    parser.add_argument("--learning_rate", default=6e-05, type=float)
    parser.add_argument("--train_batch_size", default=48, type=int)
    parser.add_argument("--eval_batch_size", default=48, type=int)
    parser.add_argument("--val_beam_size", default=4, type=int)
    parser.add_argument("--max_input_length", default=512, type=int)
    parser.add_argument("--max_output_length", default=20, type=int)
    ### jeongwoo
    parser.add_argument("--pooling_type", default=None, type=str)
    parser.add_argument("--n_passages", default=20, type=int)
    parser.add_argument("--dist_option", default='rank_inverse', type=str)
    parser.add_argument("--softmax_temp", default=1.0, type=float)
    parser.add_argument("--add_special_tokens", action='store_true')
    parser.add_argument("--n_special_tokens", default=0, type=int)
    parser.add_argument("--special_pooling", default='first', type=str)
    parser.add_argument("--target_seq" , default='token', type=str)
    parser.add_argument("--decoding_strategy", default='single', type=str)
    parser.add_argument('--num_train_steps', default=50000,type=int)
    parser.add_argument('--local_weight', default=1.0, type=float)
    parser.add_argument('--local_all', default=False, type=bool)    
    # file names
    parser.add_argument("--train-files", default=['./data/train.jsonl'], nargs='+')
    parser.add_argument("--eval-files", default=['./data/validation.jsonl'], nargs='+')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--sub-mode", default='', type=str)
    parser.add_argument("--prompt_type", default='2', type=str)
    parser.add_argument("--options", default='None', type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--name', default='noname', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--lr_scheduler', default='constant', type=str)
    parser.add_argument('--accelerator', default='deepspeed', type=str)
    parser.add_argument('--eval_steps', default=None, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--listwise_k', default=20, type=int)
    parser.add_argument('--encoder_output_k', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--output_dir', default='../checkpoints/', type=str)

    # Calculate Score (cos, dot)
    # loss function (listnet, Ranknet)
    parser.add_argument('--score_type', default='dot', type=str)
    parser.add_argument('--loss_type', default="listnet", type=str)
    parser.add_argument('--use_special_tokens', default=True)
    parser.add_argument('--extra_id', default=0, type=int)

    parser.add_argument('--eval_epochs', default=None, type=int)    

    args = parser.parse_args()
    args.weight_decay = 0
    args.adam_epsilon = 1e-8

    args.output_dir += args.name

    if args.eval_epochs is None and args.eval_steps is None:
        print("@@@ No eval epochs or steps specified. Using eval_step default 1000")
        args.eval_steps = 1000
    if args.eval_epochs is not None and args.eval_steps is not None:
        print("@@@ Both eval epochs and steps specified. Using eval_steps")
        args.eval_epochs = None

    if args.encoder_output_k == -1:
        args.encoder_output_k = args.max_input_length
    try:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        devices = ','.join([str(x) for x in list(range(torch.cuda.device_count()))])
    command_str = f"CUDA_VISIBLE_DEVICES={devices} python3 train.py {' '.join(sys.argv[1:])}"
    args.n_gpu = torch.cuda.device_count()
    args.command_str = command_str
    if args.wandb:
        arg2dict = vars(args)

        project_name = 'PUT YOUR PROJECT NAME HERE'
        entity_name = 'PUT YOUR ENTITY NAME HERE'

        wandb_logger = WandbLogger(
            project=project_name, name=args.name, entity=entity_name,
            config=arg2dict
        )
    else:
        wandb_logger = None
    if not args.do_train:
        args.n_gpu = 1
    callbacks = []

    # Store the best model based on validation NDCG@10
    checkpoint_callback = ModelCheckpoint(
        monitor='ndcg@10',                          
        mode='max',                                 
        dirpath=args.output_dir,                 
        filename='best-{epoch:02d}-{global_step}',  
        save_top_k=1,                                                         
        verbose=True                                
    )

    # Store model of every eval step
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='global_step',
    #     mode='max',
    #     dirpath=args.output_dir,
    #     every_n_train_steps=args.eval_steps,
    #     filename="{epoch:02d}-step{global_step}",
    #     #save_top_k=150,
    #     save_last=True
    # )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler == "constant":
        print(f"@@@ Not Using Learning Rate Scheduler")
    else:
        lr_callback = LearningRateMonitor()
        callbacks.append(lr_callback)

    if args.accelerator == "ddp":
        plugins = pl.strategies.DDPStrategy(find_unused_parameters=False)
        print(f"@@@ Using DDP")
    elif args.accelerator == "deepspeed":
        if args.resume_from_checkpoint is None:
            plugins = pl.strategies.deepspeed.DeepSpeedStrategy(stage=2)
        else:
            plugins = pl.strategies.deepspeed.DeepSpeedStrategy(stage=2, load_full_weights=True)
        
        print(f"@@@ Using Deepspeed stage2")
    elif args.accelerator == 'dp':
        plugins = 'dp'
        print(f"@@@ Using dp @@@@")
    else:
        import pdb; pdb.set_trace()
        raise NotImplementedError("** accelerator: Choose between (ddp|dp|deepspeed)")

    if args.run_3b:
        args.precision=16
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator='gpu',
            devices=args.n_gpu,
            #strategy=plugins,
            strategy='deepspeed_stage_2_offload',
            max_epochs=args.num_train_epochs,
            # max_steps=args.num_train_steps,
            precision=args.precision,
            default_root_dir=args.output_dir,
            logger=wandb_logger,
            val_check_interval=args.eval_steps,
            check_val_every_n_epoch=None,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
    else:
        args.precision = 'bf16'
        torch.set_float32_matmul_precision('high')
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator='gpu',
            devices=args.n_gpu,
            strategy=plugins,
            max_epochs=args.num_train_epochs,
            # max_steps=args.num_train_steps,
            precision=args.precision,
            default_root_dir=args.output_dir,
            logger=wandb_logger,
            val_check_interval=None if args.eval_steps is None else args.eval_steps,
            check_val_every_n_epoch=None if args.eval_epochs is None else args.eval_epochs,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
    Path(args.output_dir).parent.mkdir(exist_ok=True, parents=True)
    main(args, train_params)


