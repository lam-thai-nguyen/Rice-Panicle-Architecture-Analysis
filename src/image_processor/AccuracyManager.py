###########################
# Author: Lam Thai Nguyen #
###########################

import pandas as pd
from .RicePanicle import RicePanicle


class AccuracyManager:
    num_entries = 0

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
                