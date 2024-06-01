from train.utils import (train, validation )
from model.MFTC-Net import MultiEncodersSwinUNETR



## datasets 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_loader, train_loader = data_loaders(data_dir, num_samples = 3, device = device)






max_iterations = 30000
eval_num = 300
post_label = AsDiscrete(to_onehot=9)
post_pred = AsDiscrete(argmax=True, to_onehot=9)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
root_dir = "/content/drive/MyDrive/Swin_UNETR_Vanila"
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "MultiEncoders_Swin_UNETR_vanila_tunned_2_FusionBlock_synapse_12Mar.pth")))