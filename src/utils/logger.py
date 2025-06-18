import os
import csv
import json
from datetime import datetime

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_log_file = os.path.join(log_dir, f'train_log_{timestamp}.csv')
        self.val_log_file = os.path.join(log_dir, f'val_log_{timestamp}.csv')
        
        # Initialize CSV files
        self._init_train_log()
        self._init_val_log()
        
    def _init_train_log(self):
        """Initialize training log CSV"""
        with open(self.train_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'loss_total', 'loss_lse', 'loss_softmax'])
    
    def _init_val_log(self):
        """Initialize validation log CSV"""
        with open(self.val_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 
                'i2t_r1', 'i2t_r5', 'i2t_r10', 'i2t_mean_r',
                't2i_r1', 't2i_r5', 't2i_r10', 't2i_mean_r',
                'rsum'
            ])
    
    def log_step(self, epoch, step, loss_dict):
        """Log training step"""
        with open(self.train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                step,
                loss_dict['loss_total'],
                loss_dict.get('loss_lse', 0),
                loss_dict.get('loss_softmax', 0)
            ])
    
    def log_validation(self, epoch, results):
        """Log validation results"""
        # Calculate R@sum
        rsum = results['i2t']['R@1'] + results['i2t']['R@5'] + results['i2t']['R@10'] + \
               results['t2i']['R@1'] + results['t2i']['R@5'] + results['t2i']['R@10']
        
        with open(self.val_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                results['i2t']['R@1'], results['i2t']['R@5'], results['i2t']['R@10'], results['i2t']['mean_r'],
                results['t2i']['R@1'], results['t2i']['R@5'], results['t2i']['R@10'], results['t2i']['mean_r'],
                rsum
            ])
        
        # Also save as JSON for easy reading
        json_file = os.path.join(self.log_dir, f'val_epoch_{epoch}.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=4)