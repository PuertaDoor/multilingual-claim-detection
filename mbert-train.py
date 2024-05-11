from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

torch.cuda.empty_cache()

# Charger le tokenizer
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')

# Définition de la classe de Dataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, filename):
        df = pd.read_csv(filename, sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip')
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df.dropna(subset=['label'], inplace=True)
        df['label'] = df['label'].astype(int)

        # Pré-tokenisation des textes
        texts = df['text'].tolist()
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        # Conversion des labels en Tensor
        self.labels = torch.tensor(df['label'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item



# Préparer les ensembles de données
train_dataset = TextDataset(tokenizer, 'train.tsv')
dev_dataset = TextDataset(tokenizer, 'dev.tsv')
test_dataset = TextDataset(tokenizer, 'test.tsv')

# Charger le modèle
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-cased', num_labels=2)

# Geler tous les paramètres sauf le classificateur
for param in model.parameters():
    param.requires_grad = False

# Réactiver le calcul du gradient pour le classificateur
# Accéder aux paramètres du classificateur spécifiquement
for param in model.classifier.parameters():
    param.requires_grad = True

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',  # Evaluer à la fin de chaque époque
)

# Fonction pour calculer la précision
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1_macro': f1_macro, 'f1': f1}

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# Entraînement
model.train()
trainer.train()

# Évaluation sur l'ensemble de test
model.eval()
with torch.no_grad():
    trainer.evaluate(test_dataset)
