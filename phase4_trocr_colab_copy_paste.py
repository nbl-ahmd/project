# ===== Cell 1 =====
!pip -q install -U transformers datasets evaluate jiwer accelerate sentencepiece kagglehub

# ===== Cell 2 =====
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

PROJECT_DIR = Path('/content/drive/MyDrive/mtech_phase4_trocr')
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
print('PROJECT_DIR:', PROJECT_DIR)

# ===== Cell 3 =====
import os
import random
import inspect
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
import evaluate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('Torch:', torch.__version__)

# ===== Cell 4 =====
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mamun1113/doctors-handwritten-prescription-bd-dataset")

print("Path to dataset files:", path)
KAGGLE_DATASET_PATH = Path(path)

# ===== Cell 5 =====
CFG = {
    'model_name': 'microsoft/trocr-base-handwritten',
    'target_col': 'MEDICINE_NAME',
    'image_col': 'IMAGE',
    'max_target_len': 24,
    'num_train_epochs': 8,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'grad_accum_steps': 1,
    'learning_rate': 3e-5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'num_beams': 1,
    'early_stopping_patience': 3,
    'output_dir': str(PROJECT_DIR / 'checkpoints'),
    'best_dir': str(PROJECT_DIR / 'best_model'),
    'pred_csv': str(PROJECT_DIR / 'phase4_test_predictions_trocr.csv'),
    'max_steps': 1200,
}
CFG

# ===== Cell 6 =====
MANUAL_DATASET_BASE = None

DATA_ROOT_HINTS = [
    '/content/data/doctors-handwritten-prescription-bd-dataset',
    '/content/drive/MyDrive/data/doctors-handwritten-prescription-bd-dataset',
    '/content/drive/MyDrive',
    str(PROJECT_DIR / 'data'),
    str(PROJECT_DIR / 'data' / 'doctors-handwritten-prescription-bd-dataset'),
]
if 'KAGGLE_DATASET_PATH' in globals():
    DATA_ROOT_HINTS += [str(KAGGLE_DATASET_PATH), str(KAGGLE_DATASET_PATH.parent)]


def is_dataset_base(p: Path) -> bool:
    return (
        (p / 'Training' / 'training_labels.csv').exists()
        and (p / 'Validation' / 'validation_labels.csv').exists()
        and (p / 'Testing' / 'testing_labels.csv').exists()
    )


def find_dataset_base(hints):
    if MANUAL_DATASET_BASE:
        m = Path(MANUAL_DATASET_BASE)
        if is_dataset_base(m):
            return m
        print('MANUAL_DATASET_BASE invalid:', m)

    for hint in hints:
        p = Path(hint)
        if not p.exists():
            continue
        if is_dataset_base(p):
            return p
        for cand in p.rglob('*'):
            if cand.is_dir() and is_dataset_base(cand):
                return cand

    for f in glob('/content/**/training_labels.csv', recursive=True):
        cand = Path(f).parent.parent
        if is_dataset_base(cand):
            return cand
    return None

DATASET_BASE = find_dataset_base(DATA_ROOT_HINTS)
print('DATASET_BASE:', DATASET_BASE)
if DATASET_BASE is None:
    print('Debug candidates (training_labels.csv):')
    for p in glob('/content/**/training_labels.csv', recursive=True)[:30]:
        print(' -', p)
    raise FileNotFoundError('Dataset base not found. Set MANUAL_DATASET_BASE.')

# ===== Cell 7 =====
SPLITS = {
    'train': {
        'csv': DATASET_BASE / 'Training' / 'training_labels.csv',
        'img_dir': DATASET_BASE / 'Training' / 'training_words',
    },
    'val': {
        'csv': DATASET_BASE / 'Validation' / 'validation_labels.csv',
        'img_dir': DATASET_BASE / 'Validation' / 'validation_words',
    },
    'test': {
        'csv': DATASET_BASE / 'Testing' / 'testing_labels.csv',
        'img_dir': DATASET_BASE / 'Testing' / 'testing_words',
    },
}

train_df = pd.read_csv(SPLITS['train']['csv'])
val_df = pd.read_csv(SPLITS['val']['csv'])
test_df = pd.read_csv(SPLITS['test']['csv'])

print('train:', train_df.shape, 'val:', val_df.shape, 'test:', test_df.shape)
print('columns:', train_df.columns.tolist())
print('unique medicine names:', train_df[CFG['target_col']].nunique())
train_df.head(3)

# ===== Cell 8 =====
def count_missing(df, img_dir, image_col):
    img_dir = Path(img_dir)
    missing = [x for x in df[image_col].astype(str) if not (img_dir / x).exists()]
    return missing

for split_name, split_obj, df in [
    ('train', SPLITS['train'], train_df),
    ('val', SPLITS['val'], val_df),
    ('test', SPLITS['test'], test_df),
]:
    miss = count_missing(df, split_obj['img_dir'], CFG['image_col'])
    print(split_name, 'missing images:', len(miss))

# ===== Cell 9 =====
sample_df = train_df.sample(8, random_state=SEED).reset_index(drop=True)
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, ax in enumerate(axes.ravel()):
    row = sample_df.iloc[i]
    img = Image.open(SPLITS['train']['img_dir'] / row[CFG['image_col']]).convert('RGB')
    ax.imshow(img)
    ax.set_title(str(row[CFG['target_col']]), fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()

# ===== Cell 10 =====
processor = TrOCRProcessor.from_pretrained(CFG['model_name'])
model = VisionEncoderDecoderModel.from_pretrained(CFG['model_name'])

# token IDs stay in model.config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# generation params must go to generation_config
model.generation_config.max_length = CFG['max_target_len']
model.generation_config.no_repeat_ngram_size = 0
model.generation_config.length_penalty = 1.0
model.generation_config.num_beams = CFG['num_beams']
# only valid for beam search (num_beams > 1)
model.generation_config.early_stopping = CFG['num_beams'] > 1

# speed-up on T4: freeze encoder
for p in model.encoder.parameters():
    p.requires_grad = False


# ===== Cell 11 =====
class PrescriptionWordOCRDataset(Dataset):
    def __init__(self, df, img_dir, processor, target_col, image_col='IMAGE', max_target_len=24):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.target_col = target_col
        self.image_col = image_col
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.img_dir / str(row[self.image_col])).convert('RGB')

        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        text = str(row[self.target_col]).strip()
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_target_len,
            return_tensors='pt',
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {'pixel_values': pixel_values, 'labels': labels}

train_ds = PrescriptionWordOCRDataset(train_df, SPLITS['train']['img_dir'], processor, CFG['target_col'], CFG['image_col'], CFG['max_target_len'])
val_ds = PrescriptionWordOCRDataset(val_df, SPLITS['val']['img_dir'], processor, CFG['target_col'], CFG['image_col'], CFG['max_target_len'])
test_ds = PrescriptionWordOCRDataset(test_df, SPLITS['test']['img_dir'], processor, CFG['target_col'], CFG['image_col'], CFG['max_target_len'])

print('dataset sizes:', len(train_ds), len(val_ds), len(test_ds))

# ===== Cell 12 =====
cer_metric = evaluate.load('cer')
wer_metric = evaluate.load('wer')

def norm_text(s):
    return ' '.join(str(s).strip().split())

def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

    pred_texts = [norm_text(x) for x in pred_texts]
    label_texts = [norm_text(x) for x in label_texts]

    exact = np.mean([p == y for p, y in zip(pred_texts, label_texts)])
    cer = cer_metric.compute(predictions=pred_texts, references=label_texts)
    wer = wer_metric.compute(predictions=pred_texts, references=label_texts)

    return {'exact_match': float(exact), 'cer': float(cer), 'wer': float(wer)}

# ===== Cell 13 =====
def build_training_args(cfg):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    args_dict = {
        'output_dir': cfg['output_dir'],
        'predict_with_generate': True,
        'save_strategy': 'epoch',
        'logging_strategy': 'steps',
        'logging_steps': 50,
        'per_device_train_batch_size': cfg['train_batch_size'],
        'per_device_eval_batch_size': cfg['eval_batch_size'],
        'gradient_accumulation_steps': cfg['grad_accum_steps'],
        'num_train_epochs': cfg['num_train_epochs'],
        'learning_rate': cfg['learning_rate'],
        'warmup_ratio': cfg['warmup_ratio'],
        'weight_decay': cfg['weight_decay'],
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'exact_match',
        'greater_is_better': True,
        'fp16': torch.cuda.is_available(),
        'dataloader_num_workers': 2,
        'report_to': 'none',
        'remove_unused_columns': False,
        'seed': SEED,
        'max_steps': cfg['max_steps'],
    }
    if 'evaluation_strategy' in sig:
        args_dict['evaluation_strategy'] = 'epoch'
    elif 'eval_strategy' in sig:
        args_dict['eval_strategy'] = 'epoch'
    return Seq2SeqTrainingArguments(**args_dict)

training_args = build_training_args(CFG)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG['early_stopping_patience'])],
)
training_args

# ===== Cell 14 =====
train_result = trainer.train()
print(train_result)

# ===== Cell 15 =====
val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix='val')
test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix='test')
print('Validation:', val_metrics)
print('Test:', test_metrics)

# ===== Cell 16 =====
best_dir = Path(CFG['best_dir'])
best_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(best_dir))
processor.save_pretrained(str(best_dir))
print('Saved best model to:', best_dir)

# ===== Cell 17 =====
pred = trainer.predict(test_ds)
pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
pred_texts = [norm_text(x) for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]

pred_df = test_df.copy()
pred_df['PREDICTED_MEDICINE_NAME'] = pred_texts
pred_df['IS_EXACT'] = (
    pred_df[CFG['target_col']].astype(str).map(norm_text) == pred_df['PREDICTED_MEDICINE_NAME']
).astype(int)

pred_csv = Path(CFG['pred_csv'])
pred_csv.parent.mkdir(parents=True, exist_ok=True)
pred_df.to_csv(pred_csv, index=False)
print('Saved predictions to:', pred_csv)
pred_df.head(15)

# ===== Cell 18 =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_saved_model(model_dir=None):
    if model_dir is None:
        model_dir = CFG['best_dir']
    proc = TrOCRProcessor.from_pretrained(model_dir)
    mdl = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    mdl.eval()
    return proc, mdl

def predict_single_crop(image_path, proc=None, mdl=None):
    if proc is None or mdl is None:
        proc, mdl = load_saved_model()

    image = Image.open(image_path).convert('RGB')
    pixel_values = proc(images=image, return_tensors='pt').pixel_values.to(device)

    with torch.no_grad():
        gen_ids = mdl.generate(
            pixel_values,
            num_beams=CFG['num_beams'],
            max_length=CFG['max_target_len'],
            early_stopping=(CFG['num_beams'] > 1),
        )
    return norm_text(proc.batch_decode(gen_ids, skip_special_tokens=True)[0])

def predict_folder(folder_path, exts=('.png', '.jpg', '.jpeg', '.webp')):
    proc, mdl = load_saved_model()
    rows = []
    for p in sorted(Path(folder_path).iterdir()):
        if p.suffix.lower() in exts:
            rows.append({'file': p.name, 'prediction': predict_single_crop(p, proc, mdl)})
    return pd.DataFrame(rows)


# ===== Cell 19 =====
# Example inference
# print(predict_single_crop('/content/some_crop.png'))
# df_pred = predict_folder('/content/some_folder_with_crops')
# df_pred.head()

# ===== Cell 20 =====
# Optional: download artifacts from Colab runtime
# from google.colab import files
# files.download(CFG['pred_csv'])
# !zip -qr /content/phase4_best_model.zip "{CFG['best_dir']}"
# files.download('/content/phase4_best_model.zip')

