import h5py
import pandas as pd
import numpy as np
import torch

class KeypointDataset():
    def __init__(self, h5Path, labelsCSV, max_seq_len, transform = None):
        self.h5Path = h5Path
        self.labelsCSV = labelsCSV

        self.max_seq_len = max_seq_len
        self.transform = transform

        self.processData()

    def processData(self):
        self.labels = pd.read_csv(self.labelsCSV)

        with h5py.File(self.h5Path, 'r') as f:
            self.clips_ids = list(f.keys())
            #print(self.clips_ids)

            self.mapping = []
            for clip in self.clips_ids:
                clip_group = f[clip]
                signers = list(clip_group.keys())
                #print("Clip: ", clip, "\nClip Group:", list(clip_group.keys()))

                for signer in signers:
                    #print("Signer:", signer)

                    signer_group = clip_group[signer]
                    #print("Datasets in signer:", list(signer_group.keys()))

                    if "keypoints" in signer_group:
                        keypoints = signer_group["keypoints"][:]
                        self.mapping.append((clip,signer))

            self.valid_index = []
            for idx, (clip, signer) in enumerate(self.mapping):
                
                clip_name = clip.split(".")[0]
                #dfRow = self.labels.loc[self.labels["id"] == clip_name]56567885678
                labelRow = self.labels.loc[(self.labels["id"] == clip_name) & (self.labels["infered_signer"] == signer) & (self.labels["duration"] < 30)]
                
                if not labelRow.empty:
                    self.valid_index.append(idx)

    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx):
        mapped_idx = self.valid_index[idx]

        clip, signer = self.mapping[mapped_idx]
        label_row = self.labels.loc[self.labels["id"] == (clip.split(".")[0])]
        label = label_row['label'].values[0]

        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[clip][signer]["keypoints"][:]
            
            num_frames = keypoint.shape[0]
            num_joints = keypoint.shape[1] // 4

            keypoint = keypoint.reshape(num_frames, num_joints, 4)
            #print(keypoint.shape)
            keypoint = keypoint[:,:,:2]
            #print(keypoint.shape)

        if num_frames > self.max_seq_len:
            keypoint = keypoint[:self.max_seq_len]
        #elif num_frames < self.max_seq_len:
        #    padding = np.zeros((self.max_seq_len - num_frames, num_joints, 2)) 
        #    keypoint = np.concatenate([keypoint, padding], axis = 0)

        if self.transform:
            keypoint = self.transform(keypoint)

        keypoint = torch.tensor(keypoint, dtype=torch.float32)

        return keypoint, label
