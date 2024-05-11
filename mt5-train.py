from transformers import MT5ForSequenceClassification, MT5Tokenizer, MT5Config
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments, Trainer

torch.cuda.empty_cache()

# Charger le tokenizer et le modèle
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
config = MT5Config.from_pretrained('google/mt5-small', num_labels=1)
model = MT5ForSequenceClassification(config)

# Geler tous les paramètres
for param in model.parameters():
    param.requires_grad = False

# Décongeler la classification_head
for param in model.classification_head.parameters():
    param.requires_grad = True

class TextDataset(Dataset):
    def __init__(self, tokenizer, filename, max_length=512):
        self.dataframe = pd.read_csv(filename, sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip')
        self.dataframe['label'] = pd.to_numeric(self.dataframe['label'], errors='coerce')
        self.dataframe.dropna(subset=['label'], inplace=True)
        self.dataframe['label'] = self.dataframe['label'].astype(int)
        
        self.max_length = max_length
        self.encodings = tokenizer(self.dataframe['text'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(self.dataframe['label'].values, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
# Préparer les ensembles de données
train_dataset = TextDataset(tokenizer, 'train.tsv')
dev_dataset = TextDataset(tokenizer, 'dev.tsv')
test_dataset = TextDataset(tokenizer, 'test.tsv')

# Configurez ici vos arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,
    gradient_accumulation_steps=4,
)

# Fonction pour calculer la précision
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1_macro': f1_macro, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# Entraînement
model.train()
trainer.train()

# Évaluation sur l'ensemble de test
model.eval()
with torch.inference_mode():
    trainer.evaluate(test_dataset)