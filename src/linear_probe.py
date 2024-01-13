import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import json 
import numpy as np 
from sklearn.decomposition import PCA


class LinearProbeModel(nn.Module):
    def __init__(self, num_classes, model_type="neural_net", model=None, freeze_layers=True, layer_num=-1, layer_id="feed_forward_output_relu", classifier_dims=2):
        super(LinearProbeModel, self).__init__()
        self.model_type = model_type
        self.freeze_layers = freeze_layers
        self.layer_num = layer_num
        self.layer_id = layer_id
        self.model = model
        self.classifier_dims = classifier_dims
        self.num_classes = num_classes 

        self.classifier = nn.Linear(classifier_dims, num_classes)

        if freeze_layers and model_type == "neural_net":
            self.freeze_all_but_classifier(self.model)
        if model_type == "pca":
            self.pca = PCA(n_components=classifier_dims, random_state=42)

    def forward(self, input):
        if self.model_type == "neural_net":
            outputs, layers = self.model.inference_forward(input)

            if self.layer_id == "embedding":
                features = outputs 
            else:
                features = layers[self.layer_num][self.layer_id]
            logits = self.classifier(features)

        elif self.model_type == "umap":
            # reformat for UMAP 
            # remove channel dim intended for conv network 
            input = input[:,0,:,:]
            output_shape = input.shape
            input = input.reshape(-1,input.shape[1])
            input = input.detach().cpu().numpy()
            reduced = self.model.transform(input)
            reduced = torch.Tensor(reduced).to(self.device)
            logits = self.classifier(reduced)
            
            # shape is batch x num_classes (how many classes in the dataset) x sequence length 
            logits = logits.reshape(output_shape[0],output_shape[2],self.num_classes)

        elif self.model_type == "pca":
            # reformat for UMAP 
            # remove channel dim intended for conv network 
            input = input[:,0,:,:]
            output_shape = input.shape
            input = input.reshape(-1,input.shape[1])
            input = input.detach().cpu().numpy()
            reduced = self.pca.fit_transform(input)
            reduced = torch.Tensor(reduced).to(self.device)
            logits = self.classifier(reduced)
            # shape is batch x num_classes (how many classes in the dataset) x sequence length
            logits = logits.reshape(output_shape[0],output_shape[2],self.num_classes)

        elif self.model_type == "raw":
            # reformat for UMAP 
            # remove channel dim intended for conv network 
            input = input[:,0,:,:]
            output_shape = input.shape
            input = input.reshape(-1,input.shape[1])
            logits = self.classifier(input)
            # shape is batch x num_classes (how many classes in the dataset) x sequence length
            logits = logits.reshape(output_shape[0],output_shape[2],self.num_classes)

        return logits
    
    def cross_entropy_loss(self, predictions, targets):
        loss = nn.CrossEntropyLoss()
        return loss(predictions, targets)

    def freeze_all_but_classifier(self, model):
        for name, module in model.named_modules():
            if name != "classifier":
                for param in module.parameters():
                    param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    # overwrite way we have access to the models device state 
    def to(self, device):
        self.device = device
        return super(LinearProbeModel, self).to(device)

class LinearProbeTrainer():
    def __init__(self, model, train_loader, test_loader, device, lr=1e-2, plotting=False, batches_per_eval=100, desired_total_batches=1e4, patience=8, use_tqdm=True, moving_avg_window = 200):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-2, weight_decay=0.0)
        self.plotting = plotting
        self.batches_per_eval = batches_per_eval
        self.desired_total_batches = desired_total_batches
        self.patience = patience
        self.use_tqdm = use_tqdm
        self.moving_avg_window = moving_avg_window  # Window size for moving average

    def frame_error_rate(self, y_pred, y_true):
        y_pred = y_pred.permute(0,2,1).argmax(-1)
        mismatches = (y_pred != y_true).float()
        error = mismatches.sum() / y_true.numel()
        return error * 100

    def validate_model(self):
        self.model.eval()
        total_val_loss = 0
        total_frame_error = 0
        num_val_batches = 0

        with torch.no_grad():
            for i, (spectrogram, label, _) in enumerate(self.test_loader):
                if i > self.batches_per_eval:
                    break
                spectrogram, label = spectrogram.to(self.device), label.to(self.device)
                output = self.model.forward(spectrogram)
                label = label.squeeze(1).argmax(dim=-1)
                output = output.permute(0,2,1)
                loss = self.model.cross_entropy_loss(predictions=output, targets=label)
                total_val_loss += loss.item()
                total_frame_error += self.frame_error_rate(output, label).item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        avg_frame_error = total_frame_error / num_val_batches
        return avg_val_loss, avg_frame_error

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self):
        total_batches = 0
        best_val_loss = float('inf')
        num_val_no_improve = 0
        stop_training = False

        raw_loss_list, raw_val_loss_list, raw_frame_error_rate_list = [], [], []

        while total_batches < self.desired_total_batches:
            for i, (spectrogram, label, _) in enumerate(self.train_loader):
                if total_batches >= self.desired_total_batches:
                    break

                spectrogram, label = spectrogram.to(self.device), label.to(self.device)
                output = self.model.forward(spectrogram)
                label = label.squeeze(1).permute(0,2,1).argmax(dim=-2)
                output = output.permute(0,2,1)
                loss = self.model.cross_entropy_loss(predictions=output, targets=label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_batches += 1
                if total_batches % self.batches_per_eval == 0:
                    avg_val_loss, avg_frame_error = self.validate_model()

                    raw_loss_list.append(loss.item())
                    raw_val_loss_list.append(avg_val_loss)
                    raw_frame_error_rate_list.append(avg_frame_error)

                    if len(raw_val_loss_list) >= self.moving_avg_window:
                        smooth_val_loss = self.moving_average(raw_val_loss_list, self.moving_avg_window)[-1]
                        if smooth_val_loss < best_val_loss:
                            best_val_loss = smooth_val_loss
                            num_val_no_improve = 0
                        else:
                            num_val_no_improve += 1
                            if num_val_no_improve >= self.patience:
                                print("Early stopping triggered")
                                stop_training = True
                                break

                    if self.use_tqdm: 
                        print(f'Batch {total_batches}: FER = {avg_frame_error:.2f}%, Train Loss = {loss.item():.4f}, Val Loss = {avg_val_loss:.4f}')

                if stop_training:
                    break
            if stop_training:
                break

        if self.plotting:
            self.plot_results(raw_loss_list, raw_val_loss_list, raw_frame_error_rate_list)

    def plot_results(self, loss_list, val_loss_list, frame_error_rate_list):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_list, label='Training Loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.title('Loss over Batches')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(frame_error_rate_list, label='Frame Error Rate', color='red')
        plt.title('Frame Error Rate over Batches')
        plt.xlabel('Batches')
        plt.ylabel('Error Rate (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

class ModelEvaluator:
    def __init__(self, model, test_loader, num_classes=21, device='cuda:0', use_tqdm=True):
        self.model = model
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_tqdm = use_tqdm

    def validate_model_multiple_passes(self, num_passes=1, max_batches=100):
        self.model.eval()
        errors_per_class = [0] * self.num_classes
        correct_per_class = [0] * self.num_classes
        total_frames = 0
        total_errors = 0

        total_iterations = num_passes * min(max_batches, len(self.test_loader))
        progress_bar = tqdm(total=total_iterations, desc="Evaluating", unit="batch") if self.use_tqdm else None

        for _ in range(num_passes):
            with torch.no_grad():
                for i, (waveform, label, _) in enumerate(self.test_loader):
                    if i >= max_batches:
                        break

                    waveform, label = waveform.to(self.device), label.to(self.device)
                    output = self.model.forward(waveform)
                    label = label.squeeze(1).permute(0, 2, 1)
                    output = output.permute(0, 2, 1)

                    predicted_labels = output.argmax(dim=-2)
                    true_labels = label.argmax(dim=-2)

                    correct = (predicted_labels == true_labels)
                    incorrect = ~correct

                    for cls in range(self.num_classes):
                        class_mask = (true_labels == cls)
                        incorrect_class = incorrect & class_mask

                        errors_per_class[cls] += incorrect_class.sum().item()
                        correct_per_class[cls] += (correct_class := correct & class_mask).sum().item()

                        total_frames += class_mask.sum().item()
                        total_errors += incorrect_class.sum().item()

                    if progress_bar is not None:
                        progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        class_frame_error_rates = {
            cls: (errors / (errors + correct) * 100 if errors + correct > 0 else float('nan'))
            for cls, (errors, correct) in enumerate(zip(errors_per_class, correct_per_class))
        }
        total_frame_error_rate = (total_errors / total_frames * 100 if total_frames > 0 else float('nan'))
        return class_frame_error_rates, total_frame_error_rate

    def save_results(self, class_frame_error_rates, total_frame_error_rate, folder_path, layer_id=None, layer_num=None):
        # Conditional filename based on whether layer_id and layer_num are provided
        if layer_id is not None and layer_num is not None:
            suffix = f'_{layer_id}_{layer_num}'
        else:
            suffix = ''

        # Save plot
        plot_filename = f'frame_error_rate_plot{suffix}.png'
        self.plot_error_rates(class_frame_error_rates, plot_filename, folder_path)

        # Save data to JSON
        results_data = {
            'class_frame_error_rates': class_frame_error_rates,
            'total_frame_error_rate': total_frame_error_rate
        }
        json_filename = f'results{suffix}.json'
        with open(os.path.join(folder_path, json_filename), 'w') as file:
            json.dump(results_data, file)

    def plot_error_rates(self, class_frame_error_rates, plot_filename, save_path):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(class_frame_error_rates)), class_frame_error_rates.values(), color='skyblue')
        plt.xlabel('Class', fontsize=15)
        plt.ylabel('Frame Error Rate (%)', fontsize=15)
        plt.title(f'Frame Error Rates - {plot_filename.replace(".png", "")}', fontsize=15)
        plt.xticks(range(len(class_frame_error_rates)), class_frame_error_rates.keys(), fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()