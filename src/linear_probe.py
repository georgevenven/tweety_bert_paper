import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import pandas as pd 

class LinearProbeModel(nn.Module):
    def __init__(self, num_classes, model_type="neural_net", model=None, freeze_layers=True, layer_num=-1, layer_id="feed_forward_output_relu", classifier_dims=2):
        super(LinearProbeModel, self).__init__()
        self.model_type = model_type
        self.freeze_layers = freeze_layers
        self.layer_num = layer_num
        self.layer_id = layer_id

        self.classifier = nn.Linear(classifier_dims, num_classes)

        if self.model_type == "neural_net":
            self.model = model
            if freeze_layers:
                self.freeze_all_but_classifier(self.model)

    def forward(self, input):
        if self.model_type == "neural_net":
            outputs, layers = self.model.inference_forward(input)

            if self.layer_id == "embedding":
                features = outputs 
            else:
                features = layers[self.layer_num][self.layer_id]
            logits = self.classifier(features)
        else:
            logits = self.classifier(input)

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

class LinearProbeTrainer():
    def __init__(self, model, train_loader, test_loader, device, lr=1e-2, plotting=False, batches_per_eval=100, desired_total_batches=1e4, patience=8):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-2, weight_decay=0.0)
        self.plotting = plotting
        self.batches_per_eval = batches_per_eval
        self.desired_total_batches = desired_total_batches
        self.patience = patience

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

    def train(self):
        total_batches = 0
        best_val_loss = float('inf')
        num_val_no_improve = 0
        stop_training = False

        loss_list, val_loss_list, frame_error_rate_list = [], [], []

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
                    print(f'Batch {total_batches}: FER = {avg_frame_error:.2f}%, Train Loss = {loss.item():.4f}, Val Loss = {avg_val_loss:.4f}')

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        num_val_no_improve = 0
                    else:
                        num_val_no_improve += 1
                        if num_val_no_improve >= self.patience:
                            print("Early stopping triggered")
                            stop_training = True
                            break

                    loss_list.append(loss.item())
                    val_loss_list.append(avg_val_loss)
                    frame_error_rate_list.append(avg_frame_error)

                if stop_training:
                    break

        if self.plotting:
            self.plot_results(loss_list, val_loss_list, frame_error_rate_list)


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

class ModelEvaluator():
    def __init__(self, model, test_loader, num_classes=21, device='cuda:0'):
        self.model = model
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def validate_model_multiple_passes(self, num_passes=1, max_batches=100):
        self.model.eval()
        errors_per_class = [0] * self.num_classes
        correct_per_class = [0] * self.num_classes
        total_frames = 0
        total_errors = 0

        total_iterations = num_passes * min(max_batches, len(self.test_loader))
        progress_bar = tqdm(total=total_iterations, desc="Evaluating", unit="batch")

        for _ in range(num_passes):
            with torch.no_grad():
                for i, (waveform, label, _) in enumerate(self.test_loader):
                    if i >= max_batches:
                        break

                    waveform, label = waveform.to(self.device), label.to(self.device)
                    output = self.model.forward(waveform)
                    label = label.squeeze(1).permute(0, 2, 1)
                    output = output.permute(0,2,1)

                    predicted_labels = output.argmax(dim=-2)
                    true_labels = label.argmax(dim=-2)

                    correct = (predicted_labels == true_labels)
                    incorrect = ~correct

                    for cls in range(self.num_classes):
                        class_mask = (true_labels == cls)
                        incorrect_class = incorrect & class_mask

                        errors_per_class[cls] += incorrect_class.sum().item()
                        total_frames += class_mask.sum().item()
                        total_errors += incorrect_class.sum().item()

                    progress_bar.update(1)

        progress_bar.close()

        class_frame_error_rates = {cls: (errors / total_frames * 100 if total_frames > 0 else float('nan')) for cls, errors in enumerate(errors_per_class)}
        total_frame_error_rate = total_errors / total_frames * 100 if total_frames > 0 else float('nan')

        return class_frame_error_rates, total_frame_error_rate

    def save_results(self, class_frame_error_rates, total_frame_error_rate, experiment_name):
        # Create directory for the experiment
        experiment_path = os.path.join('/results', experiment_name)
        os.makedirs(experiment_path, exist_ok=True)

        # Save plot
        self.plot_error_rates(class_frame_error_rates, experiment_path)

        # Save data to CSV
        data = {'Class': list(class_frame_error_rates.keys()), 'Frame Error Rate (%)': list(class_frame_error_rates.values())}
        data['Class'].append('Overall')
        data['Frame Error Rate (%)'].append(total_frame_error_rate)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(experiment_path, 'frame_error_rates.csv'), index=False)

    def plot_error_rates(self, class_frame_error_rates, save_path):
        plt.figure(figsize=(10, 6))
        plt.bar(class_frame_error_rates.keys(), class_frame_error_rates.values(), color='skyblue')
        plt.xlabel('Class', fontsize=15)
        plt.ylabel('Frame Error Rate (%)', fontsize=15)
        plt.title('Class-wise Frame Error Rates', fontsize=15)
        plt.xticks(list(class_frame_error_rates.keys()), fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 100)
        plt.savefig(os.path.join(save_path, 'frame_error_rate_plot.png'))
        plt.close()
