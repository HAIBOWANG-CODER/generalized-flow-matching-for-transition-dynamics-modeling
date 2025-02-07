import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Lightning")

    ####### ITERATES IN THE CODE #######
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to iterate over",
    )
    parser.add_argument(
        "--t_exclude",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Time points to exclude (iterating over)",
    )
    parser.add_argument(
        "--resume_flow_model_from",
        type=str,
        default=None,
        help="Path to the model to resume training",
    )
    parser.add_argument(
        "--resume_spline_model_from",
        type=str,
        default=None,
        help="Path to the model to resume training",
    )

    ####################################

    ######### DATASETS #################
    parser = datasets_parser(parser)
    ####################################

    ######### METRICS ##################
    parser = metric_parser(parser)
    ####################################

    ######### General Training #########
    parser = general_training_parser(parser)
    ####################################

    ###### Latent Space Training ######
    vae_parser(parser)
    ###################################

    ##### Training Spline Network #####
    parser = spline_network_parser(parser)
    ####################################

    #### Training Velocity Network ####
    parser = velocity_network_parser(parser)
    ####################################

    ######### Cluster Training #########
    parser = cluster_parser(parser)
    ####################################

    ######### Train Method #############
    parser = train_method_parser(parser)
    ####################################

    return parser.parse_args()


def datasets_parser(parser):
    parser.add_argument("--dim", type=int, default=5, help="Dimension of data")
    parser.add_argument("--dataset_num", type=int, default=2, help="The number of datasets")

    parser.add_argument(
        "--data_type",
        type=str,
        default="scrna",
        help="Type of data, now wither scrna or one of toys",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="cite",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whiten the data",
    )
    parser.add_argument(
        "--whiten_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whiten only during testing",
    )
    return parser


def metric_parser(parser):
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for Spline strength, should be either 1 or 0",
    )
    parser.add_argument(
        "--n_centers",
        type=int,
        default=100,
        help="Number of centers for RBF network",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Kappa parameter for RBF network",
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="kmeans",
        help="Clustering method for RBF network",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.001,
        help="Rho parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.2, 0.2],
        help="Gamma parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--variance_aggregation",
        type=str,
        default="mean",
        help="Variance aggregation method",
    )
    parser.add_argument(
        "--metric_epochs",
        type=int,
        default=50,
        help="Number of epochs for metric learning",
    )
    parser.add_argument(
        "--metric_patience",
        type=int,
        default=5,
        help="Patience for metric learning",
    )
    parser.add_argument(
        "--metric_lr",
        type=float,
        default=1e-2,
        help="Learning rate for metric learning",
    )
    parser.add_argument(
        "--alpha_metric",
        type=float,
        default=1.0,
        help="Alpha parameter for metric learning",
    )
    parser.add_argument(
        "--ambient_space_metric_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calculate (only) image metric in ambient space",
    )
    parser.add_argument(
        "--OT_in_ambient_space",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calculate (only) image metric in ambient space",
    )
    parser.add_argument(
        "--image_size_metric",
        type=int,
        default=64,
        help="Size of the image for metric learning",
    )
    parser.add_argument(
        "--image_hx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Image hx in metric learning",
    )

    return parser


def general_training_parser(parser):
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for CFM training"
    )                               # 128
    parser.add_argument(
        "--batch_size_i", type=int, default=200, help="Batch size for replay buffer training"
    )

    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=256,
        help="Microbatch size for CFM training",
    )

    parser.add_argument(
        "--optimal_transport_method",
        type=str,
        default="exact",
        help="Use optimal transport in CFM training",
    )
    parser.add_argument(
        "--OT_heat_kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use heat kernel in optimal transport",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help="Decay for EMA",
    )
    parser.add_argument(
        "--plot_trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot trajectories at the end of training",
    )
    parser.add_argument(
        "--split_ratios",
        nargs=2,
        type=float,
        default=[0.9, 0.1],
        help="Split ratios for training/validation data in CFM training",
    )
    parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs")
    parser.add_argument("--iter_epochs", type=int, default=3000, help="Number of iteration spline train epochs")
    parser.add_argument(
        "--accelerator", type=str, default="cpu", help="Training accelerator"
    )
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--sim_num_steps",
        type=int,
        default=1000,
        help="Number of steps in simulation",
    )
    parser.add_argument(
        "--save_val_outputs",
        type=str,
        default=None,
        help="Location where to save validation outputs",
    )
    parser.add_argument(
        "--multiply_validation",
        type=int,
        default=4,
        help="Multiply validation data",
    )
    return parser


def spline_network_parser(parser):
    parser.add_argument(
        "--patience_spline",
        type=int,
        default=5,
        help="Patience for training spline model",
    )
    parser.add_argument(
        "--hidden_dims_spline",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for spline model training",
    )
    parser.add_argument(
        "--deepset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deepset for Spline training",
    )
    parser.add_argument(
        "--hidden_dim_deepset",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for Deepset model training",
    )
    parser.add_argument(
        "--time_spline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use time in spline model",
    )
    parser.add_argument(
        "--time_embedding_type",
        type=str,
        default="cat",
        help="Time embedding type for Spline model: cat, mlp, sin",
    )
    parser.add_argument(
        "--activation_spline",
        type=str,
        default="selu",
        help="Activation function for Spline",
    )
    parser.add_argument(
        "--spline_optimizer",
        type=str,
        default="adam",
        help="Optimizer for Spline training",
    )
    parser.add_argument(
        "--spline_lr",
        type=float,
        default=1e-4,
        help="Learning rate for Spline training",
    )
    parser.add_argument(
        "--spline_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for Spline training",
    )
    parser.add_argument(
        "--unet_num_channels_spline",
        type=int,
        default=64,
        help="Number of channels for UNet",
    )
    parser.add_argument(
        "--unet_num_res_blocks_spline",
        type=int,
        default=2,
        help="Number of res blocks for UNet",
    )
    parser.add_argument(
        "--unet_channel_mult_spline",
        nargs="+",
        type=int,
        default=[1, 2, 2],
        help="Channel multiplier for UNet",
    )
    parser.add_argument(
        "--unet_dropout_spline",
        type=float,
        default=0.0,
        help="Dropout for UNet",
    )
    return parser


def velocity_network_parser(parser):
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Sigma parameter for CFM (variance)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping in CFM training",
    )
    parser.add_argument(
        "--hidden_dims_velocity",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for CFM training",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=5,
        help="Check validation every N epochs during CFM training",
    )

    parser.add_argument(
        "--activation_velocity",
        type=str,
        default="selu",
        help="Activation function for CFM",
    )
    parser.add_argument(
        "--velocity_optimizer",
        type=str,
        default="adamw",
        help="Optimizer for Spline training",
    )
    parser.add_argument(
        "--velocity_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Spline training",
    )
    parser.add_argument(
        "--velocity_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for Spline training",
    )
    parser.add_argument(
        "--unet_num_channels",
        type=int,
        default=128,
        help="Number of channels for UNet",
    )
    parser.add_argument(
        "--unet_num_res_blocks",
        type=int,
        default=4,
        help="Number of res blocks for UNet",
    )
    parser.add_argument(
        "--unet_channel_mult",
        nargs="+",
        type=int,
        default=[2, 2, 2],
        help="Channel multiplier for UNet",
    )
    parser.add_argument(
        "--unet_dropout",
        type=float,
        default=0.1,
        help="Dropout for UNet",
    )
    parser.add_argument(
        "--unet_resblock_updown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use resblock updown in UNet",
    )
    parser.add_argument(
        "--unet_use_new_attention_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use new attention order in UNet",
    )
    parser.add_argument(
        "--unet_attention_resolutions",
        type=str,
        default="16",
        help="Resolutions for attention in UNet",
    )
    parser.add_argument(
        "--unet_num_heads",
        type=int,
        default=1,
        help="Number of heads for attention in UNet",
    )

    return parser


def vae_parser(parser):
    parser.add_argument(
        "--vae_epochs",
        type=int,
        default=100,
        help="Number of epochs for VAE training",

    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Latent dimension for UNet",
    )
    parser.add_argument(
        "--activation_encoder",
        type=str,
        default="selu",
        help="Activation function for encoder",
    )
    parser.add_argument(
        "--hidden_dims_encoder",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for encoder",
    )
    parser.add_argument(
        "--vae_lr",
        type=float,
        default=1e-3,
        help="Learning rate for VAE training",
    )
    return parser


def cluster_parser(parser):
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the config file for training",
    )
    parser.add_argument(
        "--data_on_cluster",
        type=str,
        default=None,
        help="data_on_cluster",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="~/scratch/data/",
        help="Directory to save logs",
    )
    return parser


def train_method_parser(parser):
    parser.add_argument(
        "--converge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Key of train staged experiment",
    )
    parser.add_argument(
        "--reflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Key of reflow train",
    )
    parser.add_argument(
        "--reflow_num", type=int, default=0, help="iterative numbers for reflow training"
    )
    parser.add_argument(
        "--resample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Key of resample train",
    )
    parser.add_argument(
        "--resample_num",
        type=int,
        default=1,
        help="Resample interation times",
    )
    parser.add_argument(
        "--direc",
        type=str,
        default=None,
        help="Direction of training",
    )
    parser.add_argument(
        "--save_address",
        type=str,
        default=None,
        help="Address of saving results",
    )
    return parser

