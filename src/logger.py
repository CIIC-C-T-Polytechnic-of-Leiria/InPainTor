import datetime
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")


class Logger:
    def __init__(self, model_name, log_dir='logs'):
        self.model_name = model_name
        self.log_dir = log_dir
        self.metrics = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')  # Initialize best_val_loss to positive infinity
        self.best_model_path = None

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_metrics(self, epoch, train_loss, val_loss):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.save_log_to_file(epoch, train_loss, val_loss)

    def save_log_to_file(self, epoch, train_loss, val_loss):
        log_path = os.path.join(self.log_dir, f'metrics_{self.model_name}.txt')
        with open(log_path, 'a') as f:
            f.write(
                f'Epoch {epoch + 1}, Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, Train '
                f'Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: '
                f'{self.best_val_loss:.4f}\n')


class Plotter:
    """Class to read and plot training and validation loss from a log file.

        Usage:
            plotter = Plotter('logs/metrics_model_name.txt')
            plotter.plot(log_scale=True)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_data()

    def read_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                epoch, timestamp, train_loss, val_loss, best_val_loss = self.parse_line(line)
                data.append((epoch, train_loss, val_loss, best_val_loss))
        return data

    @staticmethod
    def parse_line(line):
        parts = line.split(',')
        epoch = int(parts[0].split()[1])
        timestamp = parts[1].split()[2]
        train_loss = float(parts[2].split()[2])
        val_loss = float(parts[3].split()[2])
        best_val_loss = float(parts[4].split()[3])
        return epoch, timestamp, train_loss, val_loss, best_val_loss

    def plot(self, log_scale=False):
        epochs, train_losses, val_losses, _ = zip(*self.data)

        # Set font family to serif
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['cmr10']

        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-', linewidth=0.5, markersize=2)
        plt.plot(epochs, val_losses, label='Val Loss', marker='o', linestyle='-', linewidth=0.5, markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        if log_scale:
            plt.yscale('log')
        y_major_locator = MultipleLocator(0.25)  # Set the major grid interval to 0.1
        y_minor_locator = MultipleLocator(0.1)  # Set the minor grid interval to 0.01
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)
        plt.show()
