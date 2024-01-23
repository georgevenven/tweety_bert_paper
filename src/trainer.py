import os
import numpy as np 
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device, 
             max_steps=10000, eval_interval=500, save_interval=1000, 
             weights_save_dir='saved_weights', 
             save_weights=True, overfit_on_batch=False, experiment_dir=None, loss_function=None, early_stopping=True, patience=8, trailing_avg_window=1000, path_to_prototype_clusters=None):

        self.overfit_on_batch = overfit_on_batch
        self.fixed_batch = None  # Will hold the batch data when overfitting
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=1)
        self.early_stopping = early_stopping
        self.patience = patience 
        self.trailing_avg_window = trailing_avg_window  # Window size for trailing average calculation

        self.loss_list = []
        self.val_loss_list = []
        self.sum_squared_weights_list = []
        self.masked_sequence_accuracy_list = []
        self.unmasked_sequence_accuracy_list = []
        self.val_masked_sequence_accuracy_list = []
        self.val_unmasked_sequence_accuracy_list = []
        self.avg_seq_acc = [] 

        self.save_interval = save_interval
        self.weights_save_dir = weights_save_dir
        self.save_weights = save_weights
        self.experiment_dir = experiment_dir  # Assuming the experiment dir is the parent of visualizations_save_dir
        
        # Create directories if they do not exist
        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)

        # Create a subfolder for predictions under visualizations_save_dir
        self.predictions_subfolder_path = os.path.join(experiment_dir, "predictions")
        if not os.path.exists(self.predictions_subfolder_path):
            os.makedirs(self.predictions_subfolder_path)

        if loss_function == None:
            self.loss_function = 'cross_entropy'
        else:
            self.loss_function = loss_function

        if path_to_prototype_clusters is not None:
            self.path_to_prototype_clusters = path_to_prototype_clusters

    def sum_squared_weights(self):
        sum_of_squares = sum(torch.sum(p ** 2) for p in self.model.parameters())
        return sum_of_squares   
 
    def save_model(self, step):
        if self.save_weights:
            filename = f"model_step_{step}.pth"
            filepath = os.path.join(self.weights_save_dir, filename)
            torch.save(self.model.state_dict(), filepath)

    def create_large_canvas(self, intermediate_outputs, image_idx=0, spec_shape=None):
        """
        Create a large canvas that places all items in the dictionary.
        Make sure the x-axis is aligned for all of them.
        """
        num_layers = len(intermediate_outputs)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))  # All images in one column

        # Determine the y-axis limit based on the 'spec' shape if provided
        y_axis_limit = spec_shape[1] if spec_shape else None

        for ax, (name, tensor) in zip(axes, intermediate_outputs.items()):
            # Convert PyTorch tensor to NumPy array
            image_data = tensor[image_idx].cpu().detach().numpy()

            # Remove singleton dimensions
            image_data = np.squeeze(image_data)

            # If the tensor is 1D, reshape it to 2D
            if len(image_data.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)

            # Now, make sure it's 2D before plotting
            if len(image_data.shape) == 2:
                ax.imshow(image_data)  # Adjust based on your actual data
                if y_axis_limit:
                    ax.set_ylim(bottom=[0, y_axis_limit])  # Set the y-axis limit to match the 'spec'
                ax.set_aspect('auto')  # This will ensure that the y-axis size is the same in pixels for all plots
                ax.invert_yaxis()  # Invert the y-axis
            else:
                print(f"Skipping {name}, as it is not 1D or 2D after squeezing. Shape is {image_data.shape}")

            ax.set_title(name)
            ax.axis('off')

        plt.tight_layout()

    def loss(self, model, output, label, mask, spec):
        # train_loss, masked_seq_acc, unmasked_seq_acc, predicted_probs, normal_dist, energy, csim = model.mse_loss(output, label, mask, spec)
        if self.loss_function == "cross_entropy":
            train_loss, masked_seq_acc, unmasked_seq_acc, predicted_probs, normal_dist, energy, csim = model.cross_entropy_loss(output, label, mask)
            return train_loss, masked_seq_acc, unmasked_seq_acc, predicted_probs, normal_dist, energy, csim

        elif self.loss_function == "mse_loss":
            combined_loss, masked_loss, unmasked_loss, mse_loss = model.mse_loss(output, mask, spec)
            return combined_loss, masked_loss, unmasked_loss, mse_loss

        else:
            raise Exception("Error: incorrect loss function has been chosen")
        
    def visualize_cross_entropy(self, output, label, mask, spec, path_to_prototype_clusters, step):
        mask_bar_height = 15
        # Compute loss and obtain targets and predicted_labels
        _, _, _, targets, predicted_labels, loss_heatmap, softmax_csim = self.loss(self.model, output, label, mask, spec)
        loss_heatmap = loss_heatmap.squeeze().cpu().numpy()  # Assuming loss_heatmap has shape [1, 1000, 1]

        # Process the inputs
        spec = spec[0].squeeze(1).cpu().numpy()
        mask_np = mask[:, 0, :].cpu().numpy()

        # Prepare plots
        fig, axs = plt.subplots(3, 1, figsize=(30, 30))  # 3 rows, 1 column
        axs = axs.ravel()

        # Labels for X and Y axes
        x_label = 'Time Bins'
        y_label = 'Frequency Bins'

        # Adjust spacing between figures, and between titles and figures
        plt.subplots_adjust(hspace=0.33)  # Adjust vertical space between plots

        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)  # Adjust tick length and width
            ax.set_xlabel(x_label, fontsize=25)
            ax.set_ylabel(y_label, fontsize=25)

        softmax_probs = softmax_csim.cpu().numpy()

        # Convert targets and predicted_labels to CPU and to numpy arrays if they are tensors
        targets = targets[0].cpu().numpy()
        predicted_labels = predicted_labels[0].cpu().numpy()
            
        # Load prototype clusters
        prototype_clusters = np.load(path_to_prototype_clusters)

        combined_predictions_img = np.tensordot(softmax_probs[0], prototype_clusters, axes=([1],[0]))

        # Process targets and predicted labels
        targets_img = np.array([prototype_clusters[label] for label in targets])

        axs[0].imshow(spec[0], aspect='auto', origin='lower')
        axs[0].set_title('Spectrogram with Mask Overlay', fontsize=35, pad=20)
        self._add_mask_overlay(axs[0], mask_np[0], spec, mask_bar_height)

        # Plot for Targets Image
        axs[1].imshow(targets_img.T, aspect='auto', origin='lower')
        axs[1].set_title('Targets Image Reconstructed From K-means Centroids', fontsize=35, pad=20)
        self._add_mask_overlay(axs[1], mask_np[0], targets_img[0], mask_bar_height)

        # Plot for Predicted Labels with Absolute Loss Opacity Overlay
        axs[2].imshow(combined_predictions_img.T, aspect='auto', origin='lower')
        axs[2].set_title('Predicted Image Reconstructed From K-means Centroids', fontsize=35, pad=20)
        self._add_mask_overlay(axs[2], mask_np[0], combined_predictions_img[0], mask_bar_height)

        plt.savefig(os.path.join(self.predictions_subfolder_path, f'Spectrogram_{step}.png'))
        plt.close(fig)

    def visualize_mse(self, output, mask, spec, step):
        mask_bar_height = 15
        # Compute loss
        _, _, _, loss_grid = self.loss(output=output, mask=mask, spec=spec, label=None, model=self.model)
        loss_grid = loss_grid.cpu().numpy()  # Assuming loss_grid has shape [1, seq_len, 196]
        output = output.cpu().numpy()  # Assuming output has shape [1, seq_len, 196]

        # Process the inputs
        spec_np = spec.squeeze(1).cpu().numpy()
        mask_np = mask[:, 0, :].cpu().numpy()

        # Prepare plots
        fig, axs = plt.subplots(3, 1, figsize=(30, 30))  # 3 rows, 1 column
        axs = axs.ravel()

        # Labels for X and Y axes
        x_label = 'Time Bins'
        y_label = 'Frequency Bins'

        # Adjust spacing between figures, and between titles and figures
        plt.subplots_adjust(hspace=0.33)  # Adjust vertical space between plots

        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)  # Adjust tick length and width
            ax.set_xlabel(x_label, fontsize=25)
            ax.set_ylabel(y_label, fontsize=25)

        # Plot 1: Original Spectrogram with Mask
        axs[0].imshow(spec_np[0], aspect='auto', origin='lower')
        axs[0].set_title('Original Spectrogram with Mask', fontsize=35, pad=20)  # Added pad for title
        self._add_mask_overlay(axs[0], mask_np[0], spec_np[0], mask_bar_height)

        # Plot 2: Model Prediction with Mask
        axs[1].imshow(output[0].T, aspect='auto', origin='lower')
        axs[1].set_title('Model Prediction with Mask', fontsize=35, pad=20)  # Added pad for title
        self._add_mask_overlay(axs[1], mask_np[0], loss_grid[0], mask_bar_height)

        # Plot 3: Areas of Loss with Mask
        axs[2].imshow(loss_grid[0].T, aspect='auto', origin='lower', cmap='hot')
        axs[2].set_title('Areas of Loss with Mask', fontsize=35, pad=20)  # Added pad for title
        self._add_mask_overlay(axs[2], mask_np[0], loss_grid[0], mask_bar_height)

        # Save the figure
        plt.savefig(os.path.join(self.predictions_subfolder_path, f'MSE_Visualization_{step}.png'))
        plt.close(fig)

    def _add_mask_overlay(self, axis, mask, data, mask_bar_height):
        # Get the current y-axis limit to position the mask bar within the plot area
        y_min, y_max = axis.get_ylim()
        mask_bar_position = y_max - mask_bar_height  # Position at the top inside the plot

        mask_colormap = ['red' if m == 1 else 'none' for m in mask]
        for idx, color in enumerate(mask_colormap):
            if color == 'red':
                # Create a rectangle with the bottom left corner at (idx, mask_bar_position)
                axis.add_patch(plt.Rectangle((idx, mask_bar_position), 1, mask_bar_height, 
                                            edgecolor='none', facecolor=color))

    def visualize_masked_predictions(self, step):
        self.model.eval()
        with torch.no_grad():
            # Choose the batch for visualization
            if self.overfit_on_batch:
                spec, label, _ = self.fixed_batch
            else:
                spec, label, _ = next(iter(self.test_loader))

            spec = spec.to(self.device)
            label = label.to(self.device)

            # Forward pass through the model
            output, mask, _, all_outputs = self.model.train_forward(spec)

            if self.loss_function == "cross_entropy":
                self.visualize_cross_entropy(output=output, label=label, mask=mask, spec=spec, path_to_prototype_clusters=self.path_to_prototype_clusters, step=step)
            elif self.loss_function == "mse_loss":
                self.visualize_mse(output=output, mask=mask, spec=spec, step=step)

            # Create a large canvas of intermediate outputs
            self.create_large_canvas(all_outputs)

            # Save the large canvas
            plt.savefig(os.path.join(self.predictions_subfolder_path, f'Intermediate Outputs_{step}.png'))
            plt.close()
                
    def validate_model(self, step):
        self.model.eval()
        with torch.no_grad():
            # Fetch the next batch from the validation set
            spec, label, _ = next(iter(self.test_loader))
            spec = spec.to(self.device)
            label = label.to(self.device)

            # Forward pass
            output, mask, *rest = self.model.train_forward(spec)

            # Calculate loss and accuracy
            val_loss, masked_seq_acc, unmasked_seq_acc, *rest = self.loss(self.model, output, label, mask, spec)

            # Convert to scalar values
            avg_val_loss = val_loss.item()
            avg_masked_seq_acc = masked_seq_acc.item()
            avg_unmasked_seq_acc = unmasked_seq_acc.item()

        return avg_val_loss, avg_masked_seq_acc, avg_unmasked_seq_acc

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self):
        step = 0
        best_val_loss = float('inf')  # Best validation loss seen so far
        steps_since_improvement = 0  # Counter for steps since last improvement

        train_iter = iter(self.train_loader)

        # Initialize lists for storing metrics
        raw_loss_list = []
        raw_masked_seq_acc_list = []
        raw_unmasked_seq_acc_list = []
        raw_val_loss_list = []

        smoothed_val_loss_list = []  # If you plan to use it

        while step < self.max_steps:
            if self.overfit_on_batch and self.fixed_batch:
                spec, label, ground_truth = self.fixed_batch
            else:
                try:
                    spec, label, ground_truth = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    spec, label, ground_truth = next(train_iter)

            # Store this batch to continually train on, if overfit_on_batch is True
            if self.overfit_on_batch and self.fixed_batch is None:
                self.fixed_batch = (spec, label, ground_truth)

            spec = spec.to(self.device)
            label = label.to(self.device)
            ground_truth = ground_truth.to(self.device)

            output, mask, *rest = self.model.train_forward(spec)

            # there can be a variable number of variables returned 
            loss, masked_seq_acc, unmasked_seq_acc, *rest = self.loss(self.model, output, label, mask, spec)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Store metrics after each step
            raw_loss_list.append(loss.item())

            # Your existing code where validation loss is computed
            val_loss, avg_masked_seq_acc, avg_unmasked_seq_acc = self.validate_model(step)
            raw_masked_seq_acc_list.append(avg_masked_seq_acc)
            raw_unmasked_seq_acc_list.append(avg_unmasked_seq_acc)

            # Store the raw validation loss after each evaluation
            raw_val_loss_list.append(val_loss)

            # Evaluate and log metrics
            if step % self.eval_interval == 0 or step == 0:
                self.visualize_masked_predictions(step)

                # Smoothed metrics computation
                if len(raw_loss_list) >= self.eval_interval:
                    # Smooth training metrics
                    smooth_loss = self.moving_average(raw_loss_list, self.eval_interval)
                    smooth_masked_seq_acc = self.moving_average(raw_masked_seq_acc_list, self.eval_interval)
                    smooth_unmasked_seq_acc = self.moving_average(raw_unmasked_seq_acc_list, self.eval_interval)

                    # Smooth validation loss
                    smooth_val_loss = self.moving_average(raw_val_loss_list, self.eval_interval)
                    smoothed_val_loss_list.append(smooth_val_loss[-1])  # Store smoothed validation loss

                    # Logging with smoothed metrics
                    print(f'Step [{step}/{self.max_steps}], '
                        f'Training Loss: {smooth_loss[-1]:.4e}, '
                        f'Masked Seq Acc: {smooth_masked_seq_acc[-1]:.4f}, '
                        f'Unmasked Seq Acc: {smooth_unmasked_seq_acc[-1]:.4f}, '
                        f'Validation Loss: {smooth_val_loss[-1]:.4e}')  # Smoothed validation loss is used here
                else:
                    # Logging the raw values if not enough data points for smoothing
                    print(f'Step [{step}/{self.max_steps}], '
                        f'Training Loss: {raw_loss_list[-1]:.4e}, '
                        f'Masked Seq Acc: {raw_masked_seq_acc_list[-1]:.4f}, '
                        f'Unmasked Seq Acc: {raw_unmasked_seq_acc_list[-1]:.4f}, '
                        f'Validation Loss: {raw_val_loss_list[-1]:.4e}')  # Raw validation loss is used here
                
                # Check for early stopping using smoothed validation loss
                if len(smoothed_val_loss_list) > 0:
                    current_smoothed_val_loss = smoothed_val_loss_list[-1]
                    is_best = current_smoothed_val_loss < best_val_loss

                    if is_best:
                        best_val_loss = current_smoothed_val_loss
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1

                    if self.early_stopping and steps_since_improvement >= self.patience:
                        print(f"Early stopping triggered at step {step}. No improvement for {self.patience} evaluation intervals.")
                        self.save_model(step)
                        break  # Exit the training loop

            # Update validation lists with the latest validation metrics
            self.val_masked_sequence_accuracy_list.append(avg_masked_seq_acc)
            self.val_unmasked_sequence_accuracy_list.append(avg_unmasked_seq_acc)
            self.masked_sequence_accuracy_list.append(masked_seq_acc.item())
            self.unmasked_sequence_accuracy_list.append(unmasked_seq_acc.item())

            self.val_loss_list.append(val_loss)
            self.loss_list.append(loss.item())
            self.sum_squared_weights_list.append(self.sum_squared_weights().item())

            if step % self.save_interval == 0:
                self.save_model(step)
            
            step += 1

            if step >= self.max_steps:
                self.save_model(step)
            
    def plot_results(self, save_plot=False, config=None, smoothing_window=100):
        # Calculate smoothed curves for the metrics
        smoothed_training_loss = self.moving_average(self.loss_list, smoothing_window)
        smoothed_validation_loss = self.moving_average(self.val_loss_list, smoothing_window)
        smoothed_masked_seq_acc = self.moving_average(self.masked_sequence_accuracy_list, smoothing_window)
        smoothed_unmasked_seq_acc = self.moving_average(self.unmasked_sequence_accuracy_list, smoothing_window)
        smoothed_val_masked_seq_acc = self.moving_average(self.val_masked_sequence_accuracy_list, smoothing_window)
        smoothed_val_unmasked_seq_acc = self.moving_average(self.val_unmasked_sequence_accuracy_list, smoothing_window)

        plt.figure(figsize=(24, 6))

        # Plot 1: Training and Validation Loss
        plt.subplot(1, 3, 1)
        plt.plot(smoothed_training_loss, label='Smoothed Training Loss')
        plt.plot(smoothed_validation_loss, label='Smoothed Validation Loss')
        plt.legend()
        plt.title('Smoothed Training and Validation Loss')

        # Plot 2: Sum of Squared Weights per Step
        plt.subplot(1, 3, 2)
        plt.plot(self.sum_squared_weights_list, color='red', label='Sum of Squared Weights')
        plt.legend()
        plt.title('Sum of Squared Weights per Step')

        # Plot 3: Training and Validation Sequence Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(smoothed_masked_seq_acc, label='Smoothed Training Masked Sequence Accuracy')
        plt.plot(smoothed_unmasked_seq_acc, label='Smoothed Training Unmasked Sequence Accuracy')
        plt.plot(smoothed_val_masked_seq_acc, linestyle='dashed', label='Smoothed Validation Masked Sequence Accuracy')
        plt.plot(smoothed_val_unmasked_seq_acc, linestyle='dashed', label='Smoothed Validation Unmasked Sequence Accuracy')
        plt.legend()
        plt.title('Smoothed Training and Validation Sequence Accuracy')

        plt.tight_layout()
        
        if save_plot:
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
            plt.savefig(os.path.join(self.experiment_dir, 'smoothed_loss_accuracy_curves.png'))
        else:
            plt.show()

        # Prepare training statistics dictionary
        training_stats = {
            'smoothed_training_loss': smoothed_training_loss,
            'smoothed_validation_loss': smoothed_validation_loss,
            'smoothed_masked_seq_acc': smoothed_masked_seq_acc,
            'smoothed_unmasked_seq_acc': smoothed_unmasked_seq_acc,
            'smoothed_val_masked_seq_acc': smoothed_val_masked_seq_acc,
            'smoothed_val_unmasked_seq_acc': smoothed_val_unmasked_seq_acc,
            'sum_squared_weights': self.sum_squared_weights_list
        }

        # Save the training statistics as JSON
        stats_file_path = os.path.join(self.experiment_dir, 'training_statistics.json')
        with open(stats_file_path, 'w') as json_file:
            json.dump(training_stats, json_file, indent=4)

        print(f"Training statistics saved to {stats_file_path}")