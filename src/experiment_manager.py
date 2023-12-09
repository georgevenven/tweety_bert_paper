import os
import torch
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
from model import TweetyBERT
from trainer import ModelTrainer
from analysis import plot_umap_projection
import hashlib
import json
import shutil

class ExperimentRunner:
    def __init__(self, device, base_save_dir='experiments'):
        self.device = device
        self.base_save_dir = base_save_dir
        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)

    def archive_existing_experiments(self, experiment_name):
        archive_dir = os.path.join(self.base_save_dir, 'archive')
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        # Check if the experiment with the same name already exists
        source = os.path.join(self.base_save_dir, experiment_name)
        if os.path.exists(source):
            base_destination = os.path.join(archive_dir, experiment_name)
            destination = base_destination
            # Check for duplicates and create a unique name for the archive
            copy_number = 1
            while os.path.exists(destination):
                # Append a copy number to the experiment name
                destination = f"{base_destination}_copy{copy_number}"
                copy_number += 1
            
            # Move the current folder to the archive directory with the unique name
            shutil.move(source, destination)
    
    def run_experiment(self, config, i):
        experiment_name = config.get('experiment_name', f"experiment_{i}")
        self.archive_existing_experiments(experiment_name)
        
        # Create a directory for this experiment based on experiment_name
        experiment_dir = os.path.join(self.base_save_dir, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        
        # Save the config as a metadata file
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        # Data Loading
        collate_fn = CollateFunction(segment_length=config['context'])
        train_dataset = SongDataSet_Image(config['train_dir'], num_classes=config['num_clusters'], subsampling=True, subsample_factor=config['subsample'], remove_silences=config['remove_silences'])
        test_dataset = SongDataSet_Image(config['test_dir'], num_classes=config['num_clusters'], subsampling=True, subsample_factor=config['subsample'], remove_silences=config['remove_silences'])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=16)
        
        # Initialize model
        model = TweetyBERT(
            d_transformer=config['d_transformer'], 
            nhead_transformer=config['nhead_transformer'],
            embedding_dim=config['embedding_dim'],
            num_labels=config['num_clusters'],
            tau=config['tau'],
            dropout=config['dropout'],
            dim_feedforward=config['dim_feedforward'],
            transformer_layers=config['transformer_layers'],
            m=config['m'],
            p=config['p'],
            alpha=config['alpha'],
            sigma=config['sigma']
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        saved_weights_dir = os.path.join(experiment_dir, 'saved_weights')

        if not os.path.exists(saved_weights_dir):
            os.makedirs(saved_weights_dir)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Initialize trainer
        trainer = ModelTrainer(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            self.device,  
            weights_save_dir=saved_weights_dir,  # pass the directory for saved weights
            experiment_dir=experiment_dir, 
            max_steps=config['max_steps'], 
            eval_interval=config['eval_interval'], 
            save_interval=config['save_interval'], 
            overfit_on_batch=False, 
            l1_lambda=0,
            loss_function = config['loss_function']
        )        
        # Train the model
        trainer.train()
        
        # Plot the results
        trainer.plot_results(save_plot=True, config=config)

        if config['plot_umap'] == True:
            #UMAP Analysis
            plot_umap_projection(
                model, 
                self.device, 
                data_dir=config['umap_data_dir'], 
                subsample_factor=config['subsample'],  # Using new config parameter
                remove_silences=config['remove_silences'],  # Using new config parameter
                samples=1000, 
                file_path='files/category_colors_llb16.pkl', 
                layer_index=-1, 
                dict_key="feed_forward_output_relu", 
                time_bins_per_umap_point=config['time_bins_umap_point'], 
                context=config['context'],  # Using new config parameter
                save_dir=os.path.join(experiment_dir, 'umap.png')
            )