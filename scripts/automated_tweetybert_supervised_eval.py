import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
import json
import sys
sys.path.append("src")
os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from data_class import CollateFunction, SongDataSet_Image
from utils import load_model
from linear_probe import LinearProbeModel, LinearProbeTrainer, ModelEvaluator
import datetime

# Load TweetyBERT model
weights_path = "experiments/Goliath-0-No_weight_decay_a1_fp16_CVM_Noise_augmentation_4std/saved_weights/model_step_34000.pth"
config_path = "experiments/Goliath-0-No_weight_decay_a1_fp16_CVM_Noise_augmentation_4std/config.json"
tweety_bert_model = load_model(config_path, weights_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data directories
data_dir = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files"

# Define cross-validation directories
cv_dirs = [
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold1_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold1_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold1_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold1_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold1_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold1_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold2_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold2_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold2_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold2_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold2_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold2_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold3_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold3_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold3_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold3_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold3_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold3_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold4_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold4_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold4_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold4_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold4_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold4_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold5_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold5_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold5_train"),
    os.path.join(data_dir, "llb3_npz_files_train_0.3_fold5_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.03_fold5_val"),
    os.path.join(data_dir, "llb3_npz_files_train_0.003_fold5_val")
]

# Train and evaluate models for each cross-validation split
for idx, (train_dir, val_dir) in enumerate(zip(cv_dirs[::2], cv_dirs[1::2])):
    # Load datasets
    train_dataset = SongDataSet_Image(train_dir, num_classes=21)
    val_dataset = SongDataSet_Image(val_dir, num_classes=21)
    collate_fn = CollateFunction(segment_length=500)  # Adjust the segment length if needed
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn)

    # Initialize and train classifier model, the num classes is a hack and needs to be fixed later on by removing one hot encodings 
    classifier_model = LinearProbeModel(num_classes=513, model_type="neural_net", model=tweety_bert_model,
                                        freeze_layers=False, layer_num=-3, layer_id="attention_output", classifier_dims=384)
    classifier_model = classifier_model.to(device)
    trainer = LinearProbeTrainer(model=classifier_model, train_loader=train_loader, test_loader=val_loader,
                                 device=device, lr=1e-5, plotting=False, batches_per_eval=10, desired_total_batches=1e4, patience=4)
    trainer.train()


    eval_dataset = SongDataSet_Image(val_dir, num_classes=21, infinite_loader=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Evaluate the trained model
    evaluator = ModelEvaluator(model=classifier_model, test_loader=eval_loader, num_classes=21,
                               device='cuda:0', use_tqdm=True, filter_unseen_classes=True, train_dir=train_dir)
    class_frame_error_rates, total_frame_error_rate = evaluator.validate_model_multiple_passes(num_passes=1, max_batches=1250)

    # Generate a unique and descriptive folder name for the results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fold_number = idx // 2 + 1  # Calculate fold number based on index
    results_folder_name = f"evaluation_results_fold{fold_number}_{timestamp}"

    # Save the evaluation results
    results_dir = os.path.join(data_dir, results_folder_name)
    os.makedirs(results_dir, exist_ok=True)
    evaluator.save_results(class_frame_error_rates, total_frame_error_rate, results_dir)