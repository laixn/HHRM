
'''
Document Introduction

This Python file is for the HHRM (Hallmark-based Hierarchical Recursive Model) model

Input files :
        1. cleaned_data_out.csv     (11476, 6448)   11476 samples, with 6448 gene expressions per sample
        2. adjacency_matrix     (6448, 6448)*10     10 Hallmarks, adjacency matrices constructed separately
        3. cleaned_label.xlsx       (11476, 15)     Multi level labels for 11476 samples
Output files :
        1. saved_models_hlc     This folder is used to save the trained models

Code Introduction
    
    1. Label processing: For each sample, fill in its labels up to 8 layers. If not enough, use the last label to 
        supplement and create a label dictionary (a total of 1713 labels, including 405 original labels and 1308 copied labels)
    2. Relieve label imbalance: If the number of samples for the label is less than 20, fill it up to 20 through interpolation
    3. HHRM Model: Referring to the HARNN model, a hierarchical recursive model based on Hallmark was constructed
    4. Loss function: Weighted Focal Cross Entropy Loss function
            Label weight calculation: Use the log method, W-c=1/log (smooth+class_comunt), and then normalize it
            Focal cross entropy: focal=alpha * ((1-pt) * * gamma) * ce-loss
    5. Training: 60% training set, 40% testing set, independently trained multiple times (5+), saved and recorded the optimal model and metrics after each training session

'''
# python 3.8.20  pytorch 2.3.0  numpy 1.24.3  pandas 2.0.3  scikit-learn 1.3.0  SHAP 0.41.0


import torch

from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from pytorch_model_summary import summary
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import load_npz, csc_matrix, diags
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




# Data file address
data_path = '../cleaned_data_out.csv'         # ../cleaned_data_out.csv
graph_path = '../adjacency_matrix/Undirected_'     # ../adjacency_matrix/Undirected_
target_path = '../cleaned_label.xlsx'     # ../cleaned_data_test.xlsx

cancer_hallmarks_name_list = ['0_Sustaining Proliferative Signal',
                              '1_Evading Growth Suppressor',
                              '2_Resist Cell Death',
                              '3_Enabling Replicative Immortalit',
                              '4_Inducing Angiogenesis',
                              '5_Activating Invasion and Metasta',
                              '6_Genome Instability and Mutation',
                              '7_Tumor promoting Inflammation',
                              '8_Deregulating Cellular Energetic',
                              '9_Avoiding Immune Destruction']

# Read data
all_data = pd.read_csv(data_path, delimiter='\t', header=0, index_col=0).T  # (11476, 6448)

# Read tags/targets/labels
all_label = pd.read_excel(target_path, sheet_name='A∩C_cleaned', header=0)  # (11476, 15)

# Read graph adjacency matrix
Adjacency_matrix = [None] * len(cancer_hallmarks_name_list)
Adjacency_matrix_tensor = []
for index, chname in enumerate(cancer_hallmarks_name_list):
    Adj_matrix = load_npz(f'{graph_path}{chname}_matrix.npz')
    Adj = Adj_matrix.astype(np.float32)
    # Adding self loops and degree normalization to adjacency matrices
    A = Adj + csc_matrix(np.eye(Adj.shape[0], dtype=np.float32))
    degrees = np.array(A.sum(axis=1)).flatten()
    D_sqrt_inv = diags(1 / np.sqrt(degrees))
    Adjacency_matrix[index] = D_sqrt_inv @ A @ D_sqrt_inv
    # Convert adjacency matrix into tensor
    # Convert CSR matrix to COO format
    coo_matrix = Adjacency_matrix[index].tocoo()

    # Create SparseTensor(COO format)
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    L_tensor = torch.sparse_coo_tensor(indices, coo_matrix.data, coo_matrix.shape)
    Adjacency_matrix_tensor.append(L_tensor)

# Label processing
def Cleaning_data(sample_data, label, test_size, random_state=42):        # 42    None
    # Clean the Xth level label of the data based on the provided X
    filtered = label.set_index('(A∩C)_cleaned')
    True_data = filtered[filtered['check']]    # check
    False_data = filtered[filtered['check'] == False]  # check

    True_data_labels = True_data.filter(like='target', axis=1)  # (10589,8)

    def fill_labels(row):   # Reset the labels of each sample to 8
        # Remove null values (such as NaN or empty strings)
        valid_labels = [x for x in row if pd.notna(x) and str(x).strip() != '']
        num_labels = len(valid_labels)

        if num_labels < 8 and num_labels > 0:       # Fill in the labels of each sample to 8 layers
            last_label = valid_labels[-1]
            for i in range(1, 8 - num_labels + 1):
                valid_labels.append(f"{last_label}_copy_{i}")

        while len(valid_labels) < 8:
            valid_labels.append(None)

        return pd.Series(valid_labels[:8])

    filled_labels_df = True_data_labels.apply(fill_labels, axis=1)      #    [10589,8]

    # Perform a simple statistical analysis on the completed labels
    def count_label(input_df):
        True_data_counts = []
        for col_true in input_df.columns:
            # col_X = True_data_labels.columns[col_true]
            label_counts_X = input_df[col_true].value_counts().sort_values().reset_index()
            label_counts_X.columns = ['label', 'count']
            # label_counts_X['column_name'] = col
            True_data_counts.append(label_counts_X)
        combined_stats = pd.concat(True_data_counts, ignore_index=True)
        return combined_stats

    true_data_count = count_label(True_data_labels)
    filled_labels_count = count_label(filled_labels_df)


    def Clean_low_frequency(data_file, label_file_all, label_file, col, frequency, clean_mode, noise_scale=0.1):
        # 1. Count the frequency of a column of labels appearing
        col_X = label_file.columns[col]
        label_counts_X = label_file[col_X].value_counts().sort_values().reset_index()
        label_counts_X.columns = ['label', 'count']

        # 2. Identify labels with frequencies below the threshold
        low_freq_labels = label_counts_X[label_counts_X['count'] < frequency]['label'].tolist()

        # 3. Record the sample name corresponding to the low-frequency label
        low_freq_rows = label_file[label_file[col_X].isin(low_freq_labels)].index.tolist()

        # 4. Processing low-frequency samples and their labels      use interpolation mode
        clean_data_df = []
        clean_label_all = []
        clean_label_df = []
        if clean_mode == 'Delete':
            clean_data_df = data_file.drop(index=low_freq_rows)
            clean_label_all = label_file_all.drop(index=low_freq_rows)
            clean_label_df = label_file.drop(index=low_freq_rows)
        elif clean_mode == 'Polling_copy':
            for label in low_freq_labels:
                # Calculate the number of times it needs to be copied
                original_rows = label_file[label_file[col_X] == label].index        # Filter out the sample names for this label
                current_count = len(original_rows)      # Filter out the number of samples for this label
                needed_copies = frequency - current_count   # The number of times it needs to be copied

                if needed_copies <= 0:
                    continue
                if current_count == 1:      # Scenario 1: When there is only one sample, directly copy the needed pieces times
                    src_idx = original_rows[0]
                    for i in range(needed_copies):
                        new_data_rows = data_file.loc[[src_idx]].copy()
                        new_label_all = label_file_all.loc[[src_idx]].copy()
                        new_label_rows = label_file.loc[[src_idx]].copy()
                        new_index = [f"{src_idx}_copy_{i + 1}"]
                        new_data_rows.index = new_index
                        new_label_all.index = new_index
                        new_label_rows.index = new_index
                        data_file = pd.concat([data_file, new_data_rows])
                        label_file_all = pd.concat([label_file_all, new_label_all])
                        label_file = pd.concat([label_file, new_label_rows])
                else:        # Scenario 2: When there are multiple samples, randomly poll and replicate until the requirements are met
                    copies_per_sample = needed_copies // current_count      # The integer quotient rounded down, which is the number of times each sample is polling
                    remainder = needed_copies % current_count               # Take remainder
                    for i, idx in enumerate(original_rows):
                        total_copies = copies_per_sample + (1 if i < remainder else 0)
                        for j in range(total_copies):
                            new_data_rows = data_file.loc[[idx]].copy()
                            new_label_all = label_file_all.loc[[idx]].copy()
                            new_label_rows = label_file.loc[[idx]].copy()
                            new_index = [f"{idx}_copy_{j + 1}"]
                            new_data_rows.index = new_index
                            new_label_all.index = new_index
                            new_label_rows.index = new_index
                            data_file = pd.concat([data_file, new_data_rows])
                            label_file_all = pd.concat([label_file_all, new_label_all])
                            label_file = pd.concat([label_file, new_label_rows])
                clean_data_df = data_file
                clean_label_all = label_file_all
                clean_label_df = label_file
        elif clean_mode == 'interpolation':
            for label in low_freq_labels:
                # Calculate the number of times it needs to be processed
                original_rows = label_file[label_file[col_X] == label].index        # Filter out the sample names for this label
                current_count = len(original_rows)      # Filter out the number of samples for this label
                needed_copies = frequency - current_count   # The number of times it needs to be processed

                if needed_copies <= 0:
                    continue
                if current_count == 1:      # Scenario 1: When there is only one sample, noise needs to be added for data augmentation
                    src_idx = original_rows[0]
                    new_data_rows = data_file.loc[[src_idx]].copy()
                    new_label_all = label_file_all.loc[[src_idx]].copy()
                    new_label_rows = label_file.loc[[src_idx]].copy()

                    for i in range(0, needed_copies):
                        # Randomly generate addition and subtraction symbols (50% probability)
                        sign = 1 if np.random.rand() > 0.5 else -1  # Randomly select+1 or -1
                        noise = sign * np.random.randn(len(new_data_rows.columns)) * noise_scale * new_data_rows.abs().values[0]
                        noisy_data = new_data_rows + noise
                        new_index = [f"{src_idx}_noisy_{i + 1}"]
                        new_data_rows.index = new_index
                        new_label_all.index = new_index
                        new_label_rows.index = new_index
                        data_file = pd.concat([data_file, noisy_data])
                        label_file_all = pd.concat([label_file_all, new_label_all])
                        label_file = pd.concat([label_file, new_label_rows])
                else:       # Scenario 2: Linear interpolation until the requirement is met for multiple samples
                    # Extract feature data
                    new_data_rows = data_file.loc[original_rows].copy()         # Retrieve all data from this tag
                    new_label_all = label_file_all.loc[[original_rows[0]]].copy()      # Just take one label
                    new_label_rows = label_file.loc[[original_rows[0]]].copy()

                    feature_candidates = {}     # Generate candidate data
                    for col in new_data_rows.columns:
                        # Calculate the min/max for each feature
                        min_vals = new_data_rows[col].min()
                        max_vals = new_data_rows[col].max()
                        # Generate interpolation points (each feature is independent)
                        # Generate 'needed_copies' candidate values (linear interpolation)
                        candidates = np.random.uniform(min_vals, max_vals, size=needed_copies)
                        feature_candidates[col] = candidates
                    # Randomly combine candidate values of each feature to form a new sample
                    for i in range(needed_copies):
                        new_interpolation_sample = {}
                        new_index = [f"{label}_interpolation_{i + 1}"]
                        # For each feature, randomly select a candidate value (reproducible)
                        for col in feature_candidates:
                            # Randomly select an index (duplicate selection allowed)
                            chosen_idx = np.random.randint(0, needed_copies)
                            new_interpolation_sample[col] = feature_candidates[col][chosen_idx]
                        new_interpolation_sample_DF = pd.DataFrame(new_interpolation_sample, index=new_index)
                        # new_interpolation_sample_DF.index = new_index
                        new_label_all.index = new_index
                        new_label_rows.index = new_index
                        data_file = pd.concat([data_file, new_interpolation_sample_DF])
                        label_file_all = pd.concat([label_file_all, new_label_all])
                        label_file = pd.concat([label_file, new_label_rows])
                clean_data_df = data_file
                clean_label_all = label_file_all
                clean_label_df = label_file
        else:
            raise ValueError("clean_mode must be 'Delete', 'Polling_copy', or 'interpolation'")
            # print('Please choose the cleaning mode: delete or copy ')
        return clean_data_df, clean_label_all, clean_label_df

    sample_data, True_data, filled_labels_df = Clean_low_frequency(sample_data, True_data, filled_labels_df, -1, 20, 'interpolation')

    # Building a Label Dictionary
    def creat_label_dictionary(label_data,
                               dropna=True,
                               conver_str=True,
                               sort_columns=True,
                               start_index=0):

        ordered_unique_labels = []
        num_classes_list = []
        # Process column by column
        for col in label_data.columns:
            # Retrieve current column data
            col_labels = label_data[col].values
            if dropna:
                col_labels = col_labels[~pd.isnull(col_labels)]
            if conver_str:
                col_labels = col_labels.astype(str)
            unique_labels_col = pd.unique(col_labels)
            if sort_columns:
                unique_labels_col = np.sort(unique_labels_col)
            # Add to the overall list (keep column order)
            ordered_unique_labels.extend(unique_labels_col)
            num_classes_list.append(len(unique_labels_col))
        label_dict = {lab: idx + start_index for idx, lab in enumerate(ordered_unique_labels)}
        return label_dict, ordered_unique_labels, num_classes_list

    label_dict, label_list, num_classes_list= creat_label_dictionary(filled_labels_df)      # There are a total of 1713 labels, including 405 original labels and 1308 duplicate labels

    # Add index to hierarchical labels
    sample_combined_label = True_data.filter(regex='combined', axis=1)  # Obtain the combined labels of each sample
    combined_filled_labels_df = pd.concat([sample_combined_label, filled_labels_df], axis=1)
    combined_filled_labels_df.columns = ['combined_Labels'] + [f"target_{i}" for i in range(1, 9)]

    # Process the reset labels line by line
    True_data_multi_labels_toindex = []
    for row_name, row in combined_filled_labels_df.iterrows():
        # Obtain the overall hierarchical label
        full_label = row[0]  # The first column is the overall hierarchical label
        targets = row[1:9].tolist()  # target1-8
        filled_labels = ",,,".join(str(t) for t in row[1:9])    # Splicing together the completed labels
        # Initialize index list
        target_index = []
        # Check each level
        for i in range(8):
            # Retrieve the index by querying the dictionary
            target_index.append(label_dict[targets[i]])
        multi_hot = np.zeros(len(label_dict), dtype=int)
        for idx in target_index:
            multi_hot[idx] = 1

        '''
        Label processing example:
        sample: THR24_1459_S01, 
        overall label: T005 LEUK,,,T120 AML,,,T143 AMKL,,,T156 AMKL HOX,
        overall hierarchical label: T005 LEUK,,,T120 AML,,,T143 AMKL,,,T156 AMKL HOX,,,T156 AMKL HOX_copy_1,,,T156 AMKL HOX_copy_2,,,T156 AMKL HOX_copy_3,,,T156 AMKL HOX_copy_4,
        Index of labels after completion: [5, 46, 134, 313, 587, 888, 1195, 1507], 
        '''
        # Store the processed multi-level labels in a list format
        result = [
            row_name,  # Sample Name
            full_label,  # overall label
            filled_labels,  # overall hierarchical label
            target_index,  # Index of labels after completion
            multi_hot,  # Multi hot encoding of Index of labels after completion
        ]
        True_data_multi_labels_toindex.append(result)
    # Print it out and take a look
    if True_data_multi_labels_toindex:
        item = True_data_multi_labels_toindex[0]  # Print the first line
        print(f"Sample Name: {item[0]}, \noverall label: {item[1]}, \noverall hierarchical label: {item[2]}, \nIndex of labels after completion: {item[3]}")

    # Build digital labels to restore the original label names
    def crear_digital_labels(label_file):   # filled_labels_df [10589,8]
        level_label_index_dict = {}
        one_hot_encoded_list = []
        for col in label_file.columns:
            sort_unique_labels_col = np.sort(pd.unique(label_file[col]))
            level_label_index_dict[col] = {label: idx for idx, label in enumerate(sort_unique_labels_col)}
            one_hot = pd.get_dummies(label_file[col], prefix=col).astype(int).to_numpy()
            one_hot_encoded_list.append(one_hot)
        return level_label_index_dict, one_hot_encoded_list

    level_label_index_dict, one_hot_list = crear_digital_labels(filled_labels_df)

    arr = np.array(True_data_multi_labels_toindex, dtype=object)
    multihot_labels_list = (arr[:, 0],               # sample name
                            one_hot_list[0],         # One-hot encoding of level_1
                            one_hot_list[1],         # One-hot encoding of level_2
                            one_hot_list[2],         # One-hot encoding of level_3
                            one_hot_list[3],         # One-hot encoding of level_4
                            one_hot_list[4],         # One-hot encoding of level_5
                            one_hot_list[5],         # One-hot encoding of level_6
                            one_hot_list[6],         # One-hot encoding of level_7
                            one_hot_list[7],         # One-hot encoding of level_8
                            np.stack([row[4] for row in True_data_multi_labels_toindex]))         # Overall multi hot encoding

    cleaned_data = sample_data[sample_data.index.isin(multihot_labels_list[0])]

    # Split training data and testing data according to the index
    train_idx, test_idx = train_test_split(
        np.arange(len(cleaned_data)),  # Generate index
        test_size=test_size,
        random_state=random_state,      # random_state
        stratify=multihot_labels_list[8]  # Stratified Sampling Based on the Labels of Target 8 # Stratified Sampling
    )

    # Split cleaned_data
    X_train = cleaned_data.iloc[train_idx]
    X_test = cleaned_data.iloc[test_idx]

    # Split all parts of multihot_labels_list (a total of 9 tuples)
    train_labels = [arr[train_idx] for arr in multihot_labels_list]
    test_labels = [arr[test_idx] for arr in multihot_labels_list]

    print('X_train : ', X_train.shape)
    print('X_test : ', X_test.shape)
    print('train_labels(level_1) : ', np.array(train_labels[1]).shape)
    print('train_labels(level_2) : ', np.array(train_labels[2]).shape)
    print('train_labels(level_3) : ', np.array(train_labels[3]).shape)
    print('train_labels(level_4) : ', np.array(train_labels[4]).shape)
    print('train_labels(level_5) : ', np.array(train_labels[5]).shape)
    print('train_labels(level_6) : ', np.array(train_labels[6]).shape)
    print('train_labels(level_7) : ', np.array(train_labels[7]).shape)
    print('train_labels(level_8) : ', np.array(train_labels[8]).shape)
    print('train_multi_hot : ', np.array(train_labels[9]).shape)
    print('False_data : ', False_data.shape)
    return X_train, X_test, train_labels, test_labels, False_data, num_classes_list, level_label_index_dict

# X_train, X_test, train_labels, test_labels, hard_samples = Cleaning_data(all_data, all_label, 0.4, num_labels)

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_ten = Adjacency_matrix_device1         # [N,N] * 10

        self.W1 = nn.Parameter(torch.randn(self.in_features, self.in_features))
        self.W2 = nn.Parameter(torch.randn(self.in_features, self.in_features))
        self.bn = torch.nn.BatchNorm1d(10)

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        self.out_fc = nn.Linear(self.in_features, self.out_features)

    def forward(self, input_data):      # [B,1,N]
        data_att = input_data.squeeze(1)      # [B,N]
        all_GCN_out = []
        for adj in self.adj_ten:
            adj = adj.to(input_data.device)
            support = torch.mm(data_att, self.W1)       # [B,N]
            h_prime = torch.mm(support, adj)     # [B,N]
            h_prime_out = F.relu(h_prime)
            bn_h = data_att + h_prime_out
            all_GCN_out.append(bn_h.unsqueeze(1))
        out = torch.cat(all_GCN_out, dim=1)  # [B,10,N]
        # out net
        out = self.out_fc(out)
        return out

class GAT_layer(nn.Module):     #
    def __init__(self, in_features, out_features):
        super(GAT_layer, self).__init__()

class SelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(SelfAttention, self).__init__()

class CNN(nn.Module):
    def __init__(self, in_features, fc_hidden_size):
        super(CNN, self).__init__()

class Highway_net(nn.Module):
    def __init__(self, in_features, out_features, drop=0.2):
        super(Highway_net, self).__init__()
        self.highway_fc = nn.Linear(in_features, out_features)
        self.highway_lin = nn.Linear(out_features, out_features)
        self.highway_gate = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(drop)
    def forward(self, x):
        x = self.highway_fc(x)
        g = torch.relu(self.highway_lin(x))
        t = torch.sigmoid(self.highway_gate(x))
        output = g * t + x * (1 - t)
        return self.dropout(output)

class my_model(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(my_model, self).__init__()
        self.input_dim = input_dim      # 6448
        self.embed_dim = embed_dim      # fc
        self.GCN_out = GCN(input_dim, embed_dim)
        self.fc_out = nn.Linear(self.embed_dim * 10, self.embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)

        GCN_out = self.GCN_out(x)
        GCN_out = GCN_out.view(batch_size, self.embed_dim * 10)

        output = self.fc_out(GCN_out)

        return output




class LocalLayer(nn.Module):
    def __init__(self, num_units, num_classes):     # fc --> num_class
        super(LocalLayer, self).__init__()
        self.fc1 = Highway_net(num_units, num_classes * 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_classes * 4, num_classes * 2)
        self.relu2 = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_classes * 2)
        self.fc3 = nn.Linear(num_classes * 2, num_classes)

    def forward(self, input_x):
        logits = self.fc3(self.bn(self.relu2(self.fc2(self.relu1(self.fc1(input_x))))))  # [B,C]
        scores = torch.sigmoid(logits)  # [B,C]
        return logits, scores       # [B,C], [B,C]


class my_HHRM_test_model(nn.Module):
    def __init__(self, fc_hidden_size, num_classes_list):
        super(my_HHRM_test_model, self).__init__()

        self.first_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_0)
        self.second_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_0)
        self.third_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_1)
        self.fourth_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_1)
        self.five_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_2)
        self.six_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_2)
        self.seven_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_3)
        self.eight_self_att_layer = my_model(num_nodes, fc_hidden_size).to(device_3)

        # local layer
        self.first_local = LocalLayer(fc_hidden_size, num_classes_list[0]).to(device_0)
        self.second_local = LocalLayer(fc_hidden_size + num_classes_list[0], num_classes_list[1]).to(device_0)
        self.third_local = LocalLayer(fc_hidden_size + sum(num_classes_list[:2]), num_classes_list[2]).to(device_1)
        self.fourth_local = LocalLayer(fc_hidden_size + sum(num_classes_list[:3]), num_classes_list[3]).to(device_1)
        self.five_local = LocalLayer(fc_hidden_size + sum(num_classes_list[:4]), num_classes_list[4]).to(device_2)
        self.six_local = LocalLayer(fc_hidden_size + sum(num_classes_list[:5]), num_classes_list[5]).to(device_2)
        self.seven_local = LocalLayer(fc_hidden_size + sum(num_classes_list[:6]), num_classes_list[6]).to(device_3)
        self.eight_local = LocalLayer(fc_hidden_size + sum(num_classes_list[:7]), num_classes_list[7]).to(device_3)

    def forward(self, input_data):      # B,N

        input_data = input_data.to(device_0)

        first_att_out = self.first_self_att_layer(input_data)
        first_logits, first_scores = self.first_local(first_att_out)         

        second_att_out = self.second_self_att_layer(input_data)
        second_local_in = torch.cat((second_att_out, first_logits), dim=1)       
        second_logits, second_scores = self.second_local(second_local_in)

        input_data_gpu1 = input_data.to(device_1)
        second_logits_gpu1 = second_logits.to(device_1)
        first_logits_gpu1 = first_logits.to(device_1)

        third_att_out = self.third_self_att_layer(input_data_gpu1)
        third_local_in = torch.cat((third_att_out, first_logits_gpu1, second_logits_gpu1), dim=1)      
        third_logits, third_scores = self.third_local(third_local_in)

        fourth_att_out = self.fourth_self_att_layer(input_data_gpu1)
        fourth_local_in = torch.cat((fourth_att_out, first_logits_gpu1, second_logits_gpu1, third_logits), dim=1)       
        fourth_logits, fourth_scores = self.fourth_local(fourth_local_in)

        input_data_gpu2 = input_data.to(device_2)
        fourth_logits_gpu2 = fourth_logits.to(device_2)
        third_logits_gpu2 = third_logits.to(device_2)
        second_logits_gpu2 = second_logits.to(device_2)
        first_logits_gpu2 = first_logits.to(device_2)

        five_att_out = self.five_self_att_layer(input_data_gpu2)
        five_local_in = torch.cat((five_att_out, first_logits_gpu2, second_logits_gpu2, third_logits_gpu2, fourth_logits_gpu2), dim=1)       
        five_logits, five_scores = self.five_local(five_local_in)

        six_att_out = self.six_self_att_layer(input_data_gpu2)
        six_local_in = torch.cat((six_att_out, first_logits_gpu2, second_logits_gpu2, third_logits_gpu2, fourth_logits_gpu2, five_logits), dim=1)      
        six_logits, six_scores = self.six_local(six_local_in)

        input_data_gpu3 = input_data.to(device_3)
        six_logits_gpu3 = six_logits.to(device_3)
        five_logits_gpu3 = five_logits.to(device_3)
        fourth_logits_gpu3 = fourth_logits.to(device_3)
        third_logits_gpu3 = third_logits.to(device_3)
        second_logits_gpu3 = second_logits.to(device_3)
        first_logits_gpu3 = first_logits.to(device_3)

        seven_att_out = self.seven_self_att_layer(input_data_gpu3)
        seven_local_in = torch.cat((seven_att_out, first_logits_gpu3, second_logits_gpu3, third_logits_gpu3, fourth_logits_gpu3, five_logits_gpu3, six_logits_gpu3), dim=1)       
        seven_logits, seven_scores = self.seven_local(seven_local_in)

        eight_att_out = self.eight_self_att_layer(input_data_gpu3)
        eight_local_in = torch.cat((eight_att_out, first_logits_gpu3, second_logits_gpu3, third_logits_gpu3, fourth_logits_gpu3, five_logits_gpu3, six_logits_gpu3, seven_logits), dim=1)       
        eight_logits, eight_scores = self.eight_local(eight_local_in)

        global_local_logits = torch.cat((first_logits.to(device_1),
                                          second_logits.to(device_1),
                                          third_logits.to(device_1),
                                          fourth_logits.to(device_1),
                                          five_logits.to(device_1),
                                          six_logits.to(device_1),
                                          seven_logits.to(device_1),
                                          eight_logits.to(device_1),), dim=1)
        local_scores = torch.cat((first_scores.to(device_1),
                                          second_scores.to(device_1),
                                          third_scores.to(device_1),
                                          fourth_scores.to(device_1),
                                          five_scores.to(device_1),
                                          six_scores.to(device_1),
                                          seven_scores.to(device_1),
                                          eight_scores.to(device_1),), dim=1)

        return (first_logits, second_logits, third_logits, fourth_logits, five_logits, six_logits, seven_logits, eight_logits, global_local_logits), \
            (first_scores, second_scores, third_scores, fourth_scores, five_scores, six_scores, seven_scores, eight_scores, local_scores)

class Loss(nn.Module):
    def __init__(self, weight_list):
        super(Loss, self).__init__()
        self.weight_list = weight_list
        self.CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='sum')
        # self.BCEWithLogitsLoss = None

    def forward(self, num_classes_list,
                first_logits, second_logits, third_logits, fourth_logits, five_logits, six_logits, seven_logits, eight_logits, global_logits,
                first_scores, second_scores, third_scores, fourth_scores, five_scores, six_scores, seven_scores, eight_scores, global_scores,
                y_true_first, y_true_second, y_true_third, y_true_fourth, y_true_five, y_true_six, y_true_seven, y_true_eight, y_true_global):

        def focal_loss(logits, labels, weight, alpha=1.0, gamma=2.0, reduction='mean'):
            # ce_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
            ce_loss = F.cross_entropy(logits, labels, weight=weight, reduction='none', label_smoothing=0.3)
            pt = torch.exp(-ce_loss)
            focal = alpha * ((1 - pt) ** gamma) * ce_loss
            return focal.mean() if reduction == 'mean' else focal.sum()

        losses_1 = focal_loss(first_logits, y_true_first.argmax(dim=1), self.weight_list[0].to(device_1))
        losses_2 = focal_loss(second_logits, y_true_second.argmax(dim=1), self.weight_list[1].to(device_1))
        losses_3 = focal_loss(third_logits, y_true_third.argmax(dim=1), self.weight_list[2].to(device_1))
        losses_4 = focal_loss(fourth_logits, y_true_fourth.argmax(dim=1), self.weight_list[3].to(device_1))
        losses_5 = focal_loss(five_logits, y_true_five.argmax(dim=1), self.weight_list[4].to(device_1))
        losses_6 = focal_loss(six_logits, y_true_six.argmax(dim=1), self.weight_list[5].to(device_1))
        losses_7 = focal_loss(seven_logits, y_true_seven.argmax(dim=1), self.weight_list[6].to(device_1))
        losses_8 = focal_loss(eight_logits, y_true_eight.argmax(dim=1), self.weight_list[7].to(device_1))

        local_losses = losses_1 + losses_2 + losses_3 * 2 + losses_4 * 2 + losses_5 * 3 + losses_6 * 3 + losses_7 * 4 + losses_8 * 4
        # local_losses = losses_1 + losses_2 + losses_3 + losses_4 + losses_5 + losses_6 + losses_7 + losses_8

        global_losses = self.BCEWithLogitsLoss(global_logits, y_true_global.float())

        return local_losses, global_losses, local_losses + global_losses

def split_batchs(data, label, batch_size, shuffle=True):
    batches = []
    num_samples, _ = data.shape

    # Generate index
    indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)

    data_shuffled = data.iloc[indices]
    label_shuffled = [arr[indices] for arr in label]

    for i in range(0, num_samples, batch_size):
        batch_data = data_shuffled[i:i + batch_size]
        batch_tensor = torch.tensor(batch_data.values, dtype=torch.float32)
        batch_labels = [lbl[i:i + batch_size] for lbl in label_shuffled]
        batches.append((batch_tensor,
                        torch.tensor(batch_labels[1], dtype=torch.int64),
                        torch.tensor(batch_labels[2], dtype=torch.int64),
                        torch.tensor(batch_labels[3], dtype=torch.int64),
                        torch.tensor(batch_labels[4], dtype=torch.int64),
                        torch.tensor(batch_labels[5], dtype=torch.int64),
                        torch.tensor(batch_labels[6], dtype=torch.int64),
                        torch.tensor(batch_labels[7], dtype=torch.int64),
                        torch.tensor(batch_labels[8], dtype=torch.int64),
                        torch.tensor(batch_labels[9], dtype=torch.int64),))
    return batches

def compute_class_weights(y_true_onehot, method='log', smooth=1.02):
    weights_list = []
    for i in range(8):
        y_true_tensor = torch.from_numpy(y_true_onehot[i + 1])
        labels = y_true_tensor.argmax(dim=1)
        num_classes = y_true_onehot[i+1].shape[1]

        counts = torch.bincount(labels, minlength=num_classes).float()
        if method == 'inverse':
            weights = 1.0 / counts
        elif method == 'log':
            weights = 1.0 / torch.log(smooth + counts)
        elif method == 'proportional':
            weights = labels.size(0) / (counts + 1e-6)
        else:
            raise ValueError("method must be 'inverse', 'log', or 'proportional'")

        # normalize weights to sum to num_classes
        weights = weights * num_classes / weights.sum()
        weights_list.append(weights)
    return weights_list

def evaluation(scores, labels, threshold=0.5):
    # Please ensure the order, this method only calculates evaluation indicators

    eval_all = []
    # Calculate evaluation
    for i in range(8):
        y_true = labels[i].cpu().numpy()
        y_score = scores[i].cpu().numpy()

        eval_result = {
            'model': 'test_model',
            'task': f'label_{i+1}',
            'num_labels': y_score.shape[1],
            "accuracy": accuracy_score(y_true.argmax(axis=1), y_score.argmax(axis=1)),
            "precision": precision_score(y_true.argmax(axis=1), y_score.argmax(axis=1), average='macro', zero_division=0),
            "recall": recall_score(y_true.argmax(axis=1), y_score.argmax(axis=1), average='macro', zero_division=0),
            "f1_score": f1_score(y_true.argmax(axis=1), y_score.argmax(axis=1), average='macro', zero_division=0),
        }

        eval_all.append(eval_result)

    y_true_global = labels[8].cpu().numpy()
    y_score_global = scores[8].cpu().numpy()
    y_pred_binary = (y_score_global >= threshold).astype(int)
    eval_all.append({
        'model': 'test_model',
        'task': f'Global',
        'num_labels': y_pred_binary.shape[1],
        'accuracy': accuracy_score(y_true_global, y_pred_binary),
        'precision': precision_score(y_true_global, y_pred_binary, average='macro', zero_division=0),
        'recall': recall_score(y_true_global, y_pred_binary, average='macro', zero_division=0),
        'f1_score': f1_score(y_true_global, y_pred_binary, average='macro', zero_division=0)
    })
    y_true_global2 = torch.cat(labels[:8], dim=1).cpu().numpy()
    y_score_global2 = torch.cat([F.one_hot(s.argmax(dim=1), num_classes=s.size(1)).to(torch.int) for s in scores[:8]], dim=1).cpu().numpy()
    global2 = {
        'model': 'test_model',
        'task': f'Global_2',
        'num_labels': y_score_global2.shape[1],
        'accuracy': accuracy_score(y_true_global2, y_score_global2),
        'precision': precision_score(y_true_global2, y_score_global2, average='macro', zero_division=0),
        'recall': recall_score(y_true_global2, y_score_global2, average='macro', zero_division=0),
        'f1_score': f1_score(y_true_global2, y_score_global2, average='macro', zero_division=0)
    }
    eval_all.append(global2)

    new_metrics = {
        'accuracy': accuracy_score(y_true_global2, y_score_global2),
        'precision': precision_score(y_true_global2, y_score_global2, average='macro', zero_division=0),
        'recall': recall_score(y_true_global2, y_score_global2, average='macro', zero_division=0),
        'f1_score': f1_score(y_true_global2, y_score_global2, average='macro', zero_division=0)
    }

    for result in eval_all:
        print(f"Model: {result['model']}")
        print(f"Task: {result['task']}")
        print(f"  num_labels : {result['num_labels']}")
        print(f"  Accuracy : {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall   : {result['recall']:.4f}")
        print(f"  F1-score : {result['f1_score']:.4f}")

    return eval_all, new_metrics

def save_best_model(model, old_metrics, new_metrics):
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y%m%d")

    if new_metrics['accuracy'] > old_metrics['accuracy']:
        # Delete old model
        old_acc = f"{old_metrics['accuracy']:.4f}".replace('.', '_')
        old_name = f"best_model_{model_name}({old_acc}_{timestamp}).pth"
        old_model_path = model_save_dir / old_name
        if old_model_path.exists():
            os.remove(old_model_path)
        # Generate a new file name (with 4 decimal places)
        new_acc = f"{new_metrics['accuracy']:.4f}".replace('.', '_')
        new_name = f"best_model_{model_name}({new_acc}_{timestamp}).pth"

        # Save new model
        torch.save(model, model_save_dir / new_name)   
        print(f"\nBest model has been saved! version: {new_name}")

        for metric in best_metrics:
            best_metrics[metric] = new_metrics[metric]
    # Print the best historical indicators
    print("\nBest Metrics So Far:")
    for metric, value in best_metrics.items():
        print(f"Best {metric.capitalize()}: {value:.4f}")

    return best_metrics

def load_best_model(model_path, best_metrics):
    # Note: This is the load best model code used during training, which is different from the load best model code used during prediction
    # model_path = model_save_dir / best_model_name
    best_acc = f"{best_metrics['accuracy']:.4f}".replace('.', '_')
    best_name = f"best_model_{model_name}({best_acc}_*).pth"
    model_files = list(model_path.glob(best_name))
    if not model_files:
        raise FileNotFoundError(f"No matching model file found {best_acc}")
    saved_content = torch.load(model_files[0])
    print(f"Model loaded successfully! Version number: {model_files[0]}")
    return saved_content

# Model parameters
epochs = 1000
batch_size = 200
num_nodes = 6448
in_features = 1
hidden_size = 8
attention_unit_size = 32
fc_hidden_size = 128        # 128
num_adj = 10     # 10
num_classes_filled_list = [26, 64, 155, 253, 295, 301, 307, 312]
learning_rate = 1e-5
threshold = 0.5

model_name = 'HHRM_out_outLN'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device_0 = 'cuda:0'
device_1 = 'cuda:0'
device_2 = 'cuda:0'
device_3 = 'cuda:0'

Adjacency_matrix_device = torch.stack([adj.to_dense() for adj in Adjacency_matrix_tensor], dim=0)
Adjacency_matrix_device1 = Adjacency_matrix_device.to(device_0)

X_train, X_test, train_labels, test_labels, hard_samples, num_classes_list, label_index_dict = Cleaning_data(all_data, all_label, 0.4)

class_weight = compute_class_weights(train_labels)

model = my_HHRM_test_model(fc_hidden_size, num_classes_list)

criterion = Loss(class_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2, eps=1e-8)

#  
input_tensor = torch.rand(batch_size, num_nodes)
model_summary = summary(model, input_tensor)
print(model_summary)

best_metrics = {
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1_score': 0.0
}

model_save_dir = Path("../saved_models_hlc")
model_save_dir.mkdir(exist_ok=True)
best_model_name = 'best_model.pth'

for epoch in range(epochs):
    # Training  
    model.train()
    epoch_loss = 0

    batches = tqdm(split_batchs(X_train, train_labels, batch_size=batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for batch_id, batch in enumerate(batches):
        batch_data, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_global = batch

        batch_data = batch_data.to(device_0)
        optimizer.zero_grad()
        # forward propagation
        logits, scores = model(batch_data)

        # calculate the loss
        local_loss, global_losses, all_loss = criterion(num_classes_list,
                         logits[0].to(device_1), logits[1].to(device_1), logits[2].to(device_1), logits[3].to(device_1), logits[4].to(device_1), logits[5].to(device_1), logits[6].to(device_1), logits[7].to(device_1), logits[8].to(device_1),
                         scores[0].to(device_1), scores[1].to(device_1), scores[2].to(device_1), scores[3].to(device_1), scores[4].to(device_1), scores[5].to(device_1), scores[6].to(device_1), scores[7].to(device_1), scores[8].to(device_1),
                         label_1.to(device_1), label_2.to(device_1), label_3.to(device_1), label_4.to(device_1), label_5.to(device_1), label_6.to(device_1), label_7.to(device_1), label_8.to(device_1), label_global.to(device_1))

        torch.cuda.empty_cache()

        all_loss.backward()
        optimizer.step()

        epoch_loss += all_loss.item()

        batches.set_postfix(
                            local_loss=f"{local_loss.item():.4f}",
                            global_losses=f"{global_losses.item():.4f}",
                            loss=f"{all_loss:.4f}",
                            epoch_loss=f"{epoch_loss:.4f}")
    print(f"Model name: {model_name}, Epoch [{epoch + 1}/{epochs}], epoch_loss: {epoch_loss:.4f}")

    if (epoch+1) % 99 == 0 or (epoch+1) > 900:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            all_scores = []
            all_labels = []
            test_batches = split_batchs(X_test, test_labels, batch_size=batch_size, shuffle=False)
            for batch_id, (batch_data, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_global) in enumerate(test_batches):
                batch_data = batch_data.to(device_0)
                # Start testing
                test_logits, test_scores = model(batch_data)

                test_local_loss, test_global_losses, test_all_loss = criterion(num_classes_list,
                    test_logits[0].to(device_1), test_logits[1].to(device_1), test_logits[2].to(device_1), test_logits[3].to(device_1), test_logits[4].to(device_1), test_logits[5].to(device_1), test_logits[6].to(device_1), test_logits[7].to(device_1), test_logits[8].to(device_1),
                    test_scores[0].to(device_1), test_scores[1].to(device_1), test_scores[2].to(device_1), test_scores[3].to(device_1), test_scores[4].to(device_1), test_scores[5].to(device_1), test_scores[6].to(device_1), test_scores[7].to(device_1), test_scores[8].to(device_1),
                    label_1.to(device_1), label_2.to(device_1), label_3.to(device_1), label_4.to(device_1), label_5.to(device_1), label_6.to(device_1), label_7.to(device_1), label_8.to(device_1), label_global.to(device_1))

                val_loss += test_all_loss.item()
                labels = [label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_global]
                all_labels.append(labels)
                all_scores.append(test_scores)

            labels_concat = [torch.cat([batch[i] for batch in all_labels], dim=0).cpu() for i in range(9)]
            scores_concat = [torch.cat([batch[i] for batch in all_scores], dim=0).cpu() for i in range(9)]
            test_eval, new_metrics = evaluation(scores_concat, labels_concat, threshold=threshold)
            best_metrics = save_best_model(model, best_metrics, new_metrics)

    if (epoch + 1) == 1000:
        print(f"\nImporting the best model ..")
        loaded_model = load_best_model(model_save_dir, best_metrics)
        with torch.no_grad():
            val_loss = 0.0
            all_scores = []
            all_labels = []
            test_batches = split_batchs(X_test, test_labels, batch_size=batch_size, shuffle=False)
            for batch_id, (batch_data, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_global) in enumerate(test_batches):
                batch_data = batch_data.to(device_0)
                # Start testing
                test_logits, test_scores = loaded_model(batch_data)

                test_local_loss, test_global_losses, test_all_loss = criterion(num_classes_list,
                    test_logits[0].to(device_1), test_logits[1].to(device_1), test_logits[2].to(device_1), test_logits[3].to(device_1), test_logits[4].to(device_1), test_logits[5].to(device_1), test_logits[6].to(device_1), test_logits[7].to(device_1), test_logits[8].to(device_1),
                    test_scores[0].to(device_1), test_scores[1].to(device_1), test_scores[2].to(device_1), test_scores[3].to(device_1), test_scores[4].to(device_1), test_scores[5].to(device_1), test_scores[6].to(device_1), test_scores[7].to(device_1), test_scores[8].to(device_1),
                    label_1.to(device_1), label_2.to(device_1), label_3.to(device_1), label_4.to(device_1), label_5.to(device_1), label_6.to(device_1), label_7.to(device_1), label_8.to(device_1), label_global.to(device_1))

                val_loss += test_all_loss.item()
                labels = [label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_global]
                all_labels.append(labels)
                all_scores.append(test_scores)

            labels_concat = [torch.cat([batch[i] for batch in all_labels], dim=0).cpu() for i in range(9)]
            scores_concat = [torch.cat([batch[i] for batch in all_scores], dim=0).cpu() for i in range(9)]
            test_eval, new_metrics = evaluation(scores_concat, labels_concat, threshold=threshold)



