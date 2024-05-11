import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import XLMRobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import optuna
import csv
from torch.utils.data import WeightedRandomSampler
import numpy as np

torch.cuda.empty_cache()

BATCH_SIZE = 32
NUM_EPOCHS = 3
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_WARMUP_STEPS = 500

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        
        # Couche d'embedding: Convertit les IDs de tokens en vecteurs d'embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Couche LSTM: Prend les embeddings en entrée et produit des états cachés et une cellule d'état
        # Note: Le paramètre batch_first=True indique que l'entrée et la sortie seront de forme (batch_size, seq_length, features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Couche de sortie: Projette la sortie du LSTM (état caché) sur l'espace de sortie (par ex., binaire pour la classification)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text, text_lengths, attention_mask=None):
        # Texte -> embeddings
        embedded = self.embedding(text)
        
        # Empaquetage des embeddings: Permet au LSTM de traiter efficacement les séquences de différentes longueurs
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Passage à travers le LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Utilisation du dernier état caché pour la classification
        # -1 indexe le dernier layer; hidden[-1,:,:] donne le dernier état caché de toutes les instances du batch
        hidden = hidden[-1,:,:]
        
        # Passage à travers la couche linéaire pour obtenir les logits/prédictions
        output = self.fc(hidden)
        
        return output

class TextDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        # Chargement et prétraitement initial des données
        self.dataframe = pd.read_csv(filename, sep='\t', header=None, names=['sentence', 'label'], on_bad_lines='skip')
        self.dataframe.dropna(inplace=True)  # Supprime les lignes avec des valeurs NaN
        self.dataframe = self.dataframe[self.dataframe['label'].apply(lambda x: x.isdigit())]
        self.dataframe['label'] = self.dataframe['label'].astype(int)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sentence, label = self.dataframe.iloc[idx]
        # Tokenisation et encodage du texte
        encoded = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze(0)  # Retire une dimension superflue
        attention_mask = encoded['attention_mask'].squeeze(0)
        length = attention_mask.sum().item()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long), length
    
    @staticmethod
    def collate_fn(batch):
        input_ids, attention_mask, labels, lengths = zip(*batch)
        # Empilement des tensors pour créer un batch
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.tensor(labels, dtype=torch.long)  # Assure la cohérence du type pour les étiquettes
        lengths = torch.tensor(lengths, dtype=torch.long)
        return input_ids, attention_mask, labels, lengths



def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training')
    for step, (inputs, attention_mask, labels, lengths) in progress_bar:
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs, lengths, attention_mask).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Mise à jour de la description de la barre de progression avec les informations actuelles sur la perte
        progress_bar.set_description(f"Training - Step {step+1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f'\nAverage Training Loss: {avg_loss}')
    return avg_loss



def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, total=len(dataloader), desc='Evaluating')
    with torch.inference_mode():
        for inputs, attention_mask, labels, lengths in progress_bar:
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(inputs, lengths, attention_mask).squeeze(1)  
            
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f'\nAccuracy: {accuracy}, F1 Score: {f1}, Macro F1 Score: {macro_f1}')
    return accuracy, f1, macro_f1



# Chargement des données

# Noms des fichiers TSV
fichier_tsv1 = 'train.tsv'
fichier_tsv2 = 'train_aug.tsv'

# Chemin du fichier de sortie
fichier_sortie = 'train_lstm_aug.tsv'

# Ouvrir le fichier de sortie
with open(fichier_sortie, 'w', newline='') as fichier_out:
    writer = csv.writer(fichier_out, delimiter='\t')
    
    # Traiter le premier fichier
    with open(fichier_tsv1, 'r', newline='') as fichier_in:
        reader = csv.reader(fichier_in, delimiter='\t')
        
        # Écrire chaque ligne du premier fichier dans le fichier de sortie
        for ligne in reader:
            writer.writerow(ligne)
    
    # Traiter le deuxième fichier
    with open(fichier_tsv2, 'r', newline='') as fichier_in:
        reader = csv.reader(fichier_in, delimiter='\t')
        
        # Écrire chaque ligne du deuxième fichier dans le fichier de sortie
        for ligne in reader:
            writer.writerow(ligne)

def create_weighted_sampler(fichier_sortie, weight_aug):
    # Initialiser une liste pour stocker les poids
    weights = []

    # Lire le fichier combiné
    dataframe = pd.read_csv(fichier_sortie, sep='\t', header=None, names=['sentence', 'label'], on_bad_lines='skip')
    dataframe.dropna(inplace=True)
    with open('train_aug.tsv', 'r') as train_aug:
        list_train_aug = np.genfromtxt(train_aug, delimiter='\t')
        for index, row in dataframe.iterrows():
            if row['sentence'] in list_train_aug[:, 0]:
                weights.append(weight_aug)  # Poids plus élevé pour train_aug.tsv
            else:
                weights.append(1.0)  # Poids normal pour train.tsv

    # Créer un WeightedRandomSampler avec les poids calculés
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return sampler

# train_dataset = TextDataset(fichier_sortie, tokenizer)
# sampler = create_weighted_sampler(fichier_sortie, weight_aug)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=TextDataset.collate_fn)
#cdev_dataset = TextDataset("dev.tsv", tokenizer)
# dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TextDataset.collate_fn)
test_dataset = TextDataset("test.tsv", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TextDataset.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(tokenizer), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

# Définition de la fonction de perte
def count_labels(filenames):
    # Initialiser un dictionnaire pour stocker les comptes pour chaque fichier
    label_counts = {filename: {0: 0, 1: 0} for filename in filenames}

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # Ici, on suppose que le label est toujours à la fin de la ligne
                # et qu'il est séparé par une tabulation (\t)
                label = line.strip().split('\t')[-1]
                if label in ['0', '1']:  # S'assurer que le label est valide
                    label_counts[filename][int(label)] += 1

    return label_counts

filenames = ['train_lstm_aug.tsv']
label_counts = count_labels(filenames)

total_examples = label_counts['train_lstm_aug.tsv'][0] + label_counts['train_lstm_aug.tsv'][1]
weight_for_0 = total_examples / (label_counts['train_lstm_aug.tsv'][0] * 2)
weight_for_1 = total_examples / (label_counts['train_lstm_aug.tsv'][1] * 2)
pos_weight = torch.tensor([weight_for_1 / weight_for_0], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# total_steps = len(train_loader) * NUM_EPOCHS
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=total_steps)

'''
# Boucle principale d'entraînement
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    train_epoch(model, train_loader, optimizer, scheduler, device)
    evaluate(model, dev_loader, device)

# Calcul des métriques sur l'ensemble de test
test_accuracy, test_f1, test_macro_f1 = evaluate(model, test_loader, device)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test Macro F1 Score: {test_macro_f1:.4f}')
'''

def objective(trial):
    # Suggère des valeurs pour les hyperparamètres
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    embedding_dim = trial.suggest_categorical('embedding_dim', [100, 200, 300])
    num_epochs = trial.suggest_int('num_epochs', 1, 10)
    num_warmup_steps = trial.suggest_categorical('num_warmup_steps', [400, 500, 600, 700, 800])
    weight_aug = trial.suggest_categorical('weight_aug', [10, 100, 1000])

    train_dataset = TextDataset(fichier_sortie, tokenizer)
    sampler = create_weighted_sampler(fichier_sortie, weight_aug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=TextDataset.collate_fn)
    dev_dataset = TextDataset("dev.tsv", tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=TextDataset.collate_fn)

    model = LSTMClassifier(len(tokenizer), embedding_dim, hidden_dim, OUTPUT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    # Boucle d'entraînement avec les hyperparamètres suggérés
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, scheduler, device)
        val_accuracy, val_f1, val_macro_f1 = evaluate(model, dev_loader, device)

    # Retourne la métrique que vous voulez maximiser/minimiser
    return val_macro_f1  # ou autre métrique de votre choix

# Création de l'étude Optuna
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())  # ou 'minimize' selon le cas
study.optimize(objective, n_trials=40)  # Définit le nombre d'essais

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")