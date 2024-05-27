###########################
# Author: Lam Thai Nguyen #
###########################

import statistics
import pandas as pd
from .RicePanicle import RicePanicle


class AccuracyManager:
    num_entries = 0
    fold_A_tracker = None
    fold_B_tracker = None

    def __init__(self):
        self.scores = dict()
        self.names = list()
        self.f1_1 = list()
        self.pr_1 = list()
        self.rc_1 = list()
        self.f1_2 = list()
        self.pr_2 = list()
        self.rc_2 = list()
        self.f1_3 = list()
        self.pr_3 = list()
        self.rc_3 = list()

    def show(self):
        if self.num_entries == 0:
            print("No entry!")
            return
        print(self.scores)

    def add(self, detection_accuracy: RicePanicle.DetectionAccuracy):
        self.num_entries += 1
        self.names.append(detection_accuracy.name)
        self._assign_accuracy(detection_accuracy)
        self._update_scores()

    def _assign_accuracy(self, detection_accuracy: RicePanicle.DetectionAccuracy):
        self.f1_1.append(detection_accuracy.f1_1)
        self.pr_1.append(detection_accuracy.pr_1)
        self.rc_1.append(detection_accuracy.rc_1)
        self.f1_2.append(detection_accuracy.f1_2)
        self.pr_2.append(detection_accuracy.pr_2)
        self.rc_2.append(detection_accuracy.rc_2)
        self.f1_3.append(detection_accuracy.f1_3)
        self.pr_3.append(detection_accuracy.pr_3)
        self.rc_3.append(detection_accuracy.rc_3)

    def _update_scores(self):
        self.scores[("All junctions", "f1_score")] = self.f1_1
        self.scores[("All junctions", "precision")] = self.pr_1
        self.scores[("All junctions", "recall")] = self.rc_1
        self.scores[("Main axis junctions", "f1_score")] = self.f1_2
        self.scores[("Main axis junctions", "precision")] = self.pr_2
        self.scores[("Main axis junctions", "recall")] = self.rc_2
        self.scores[("High order junctions", "f1_score")] = self.f1_3
        self.scores[("High order junctions", "precision")] = self.pr_3
        self.scores[("High order junctions", "recall")] = self.rc_3

    def save_as_csv(self, save_path: str):
        df = pd.DataFrame(self.scores, index=self.names)
        df.index.name = "image_name"
        df.to_csv(save_path)
                
    def read_fold(self, fold_path: str) -> None:
            
        self._read_fold_A(fold_path)
        self._read_fold_B(fold_path)
        
    def _read_fold_A(self, fold_path: str) -> None:
        if self.fold_A_tracker is None:
            main_keys = ['DEEPCRACK', 'FCN', 'SEGNET', 'UNET', 'U2CRACKNET', 'ACS', 'RUC_NET']
            subkeys = ['precision', 'recall', 'F1']
            self.fold_A_tracker = {key: {subkey: [] for subkey in subkeys} for key in main_keys}
        
        df = pd.read_excel(fold_path)
        for _, row in df.iterrows():
            name_model = row['Name model']
            precision = row['precision']
            recall = row['recall']
            f1 = row['F1']
            self.fold_A_tracker[name_model]['precision'].append(precision)
            self.fold_A_tracker[name_model]['recall'].append(recall)
            self.fold_A_tracker[name_model]['F1'].append(f1)
            
    def _read_fold_B(self, fold_path: str) -> None:
        if self.fold_B_tracker is None:
            self.fold_B_tracker = list()
            
        fold_id = fold_path.split('/')[-1].split('.')[0][-1]
            
        df = pd.read_excel(fold_path)
        
        idx_max = df["F1"].idxmax()
        name_model = df["Name model"][idx_max]
        F1 = df["F1"][idx_max]
        precision = df["precision"][idx_max]
        recall = df["recall"][idx_max]
        
        aver_F1 = statistics.mean(df["F1"])
        entry = [fold_id, name_model, (precision, recall, F1), aver_F1]
        self.fold_B_tracker.append(entry)
        
    def _model_A_selection(self) -> str:
        if self.fold_A_tracker is None:
            return
        
        mean_dict = {model: {metric: statistics.mean(values) for metric, values in metrics.items()} for model, metrics in self.fold_A_tracker.items()}
        model_A = max(mean_dict, key=lambda x: mean_dict[x]['F1'])
        print(f"Highest performance model: {model_A} | Aver_F1: {mean_dict[model_A]['F1']} | Aver_Precision: {mean_dict[model_A]['precision']} | Aver_Recall: {mean_dict[model_A]['recall']}")
        
        return model_A
            
    def _model_B_selection(self) -> str:
        if self.fold_B_tracker is None:
            return
        
        (fold_id, model_B, (precision, recall, F1), aver_F1) = max(self.fold_B_tracker, key=lambda x: x[-1])
        print(f"Highest averaged fold: #{fold_id} | Aver_F1: {aver_F1} | Model: {model_B} | F1: {F1} | Precision: {precision} | Recall: {recall}")
        
        return model_B
        
    def model_selection(self) -> list[str]:
        return [self._model_A_selection(), self._model_B_selection()]