# ============================================================== IMPORTS ============================================================== #
### DATA HANDLING ###
import numpy as np
import pandas as pd

### DATA VISUALISATION - PLOTTING ###
import matplotlib.pyplot as plt

### NEURAL NETWORK LIBRARIES ###
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, accuracy_score

### HUGGING FACE TRANSFORMERS FOR BERT MODEL ###
import accelerate
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertConfig

### TENSOR AND MODEL TRAINING ###
import torch
from torch.utils.data import Dataset

### REGEX FROM CLEANSING ###
import re

# DEBUG LINE #
print("\nIMPORTS SUCCESSFUL!")
# ===================================================================================================================================== #

# ======================================================== DATASET PREPARATION ======================================================== #
### CLEANINSING TWEET TEXT - SINGLE UNIT ###
def clean_tweet(tweet):
    original_tweet = tweet.lower().strip()        # CONVERTS TO LOWERCASE & REMOVES SPACES (LEADING + TRAILING)
    tweet = re.sub(r'http\S+', '', tweet)         # REMOVES URLs
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)     # REMOVES NON-ALPHA CHARs
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)  # REMOVES MENTIONS
    return tweet

### LOADS AND PREPARES DATASETS ###
def loadDataset(train_path, test_path, seed):
    ### SETTING RANDOM SEED: For reproducibility ###
    np.random.seed(seed)
    
    # DEBUG LINES #
    print("SEED:",seed, "\n")
    print("LOADING DATASET...\n")
    
    ### LOADS TRAINING AND TEST DATASET ###
    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["label", "text"])
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["label", "text"])

    ### Check for unexpected data types in the text column
    if not pd.api.types.is_string_dtype(train_df["text"]) or not pd.api.types.is_string_dtype(test_df["text"]):
        raise ValueError("Non-string data found in the text column.")

    ### SAMPLES A RANDOM TWEET BEFORE CLEANSING ###
    random_index = np.random.choice(train_df.index)
    original_tweet = train_df.loc[random_index, "text"]
    
    ### APPLIES CLEANSING: only to text column ###
    train_df["text"] = train_df["text"].apply(clean_tweet)
    test_df["text"] = test_df["text"].apply(clean_tweet)

    ### SAMPLES A RANDOM TWEET AFTER CLEANSING ###
    cleaned_tweet = train_df.loc[random_index, "text"]

    # DEBUG PRINT: One tweet before and after cleansing.
    print("\nBefore Cleaning:")
    print(original_tweet)
    print("\nAfter Cleaning:")
    print(cleaned_tweet)
    
    ### SHUFFLES DATASETS ###
    train_df = shuffle(train_df, random_state = seed)
    test_df = shuffle(test_df, random_state = seed)

    ### RETURNS PREPARED DATASETS: Returns cleansed and shuffled datasets ###
    return train_df, test_df

### SEED ####
seed = 27081999

### FILE PATHS & FUNCTION CALL ###
train_path = "dataset/train_150k.txt"
test_path = "dataset/test_62k.txt"

### FUNCTION CALL WITH SEED ###
train_df, test_df = loadDataset(train_path, test_path, seed)

# DEBUG LINE #
print("\nFINISHED\n")
# ===================================================================================================================================== #

# ======================================================== FUNCTION DEFINITION ======================================================== #
### PLOTTING FUNCTION: Generates plots for confusion matrix/ROC/recall ###
def plot_graphs(y_true, y_pred, model_name="Model", save_dir="plots"):

    ### 3 SUBPLOTS FIGURE ###
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    
    ### CONFUSION MATRIX ###
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax[0], cmap="Blues", values_format="d")
    ax[0].set_title(f'{model_name} - Confusion Matrix')

    # =================================== ROC CURVE =================================== #
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, color="darkorange", lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title(f'{model_name} - ROC Curve')
    ax[1].legend(loc="lower right")

    # ============================= PRECISION-RECALL CURVE ============================= #
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ax[2].plot(recall, precision, color="blue", lw=2)
    ax[2].set_xlabel("Recall")
    ax[2].set_ylabel("Precision")
    ax[2].set_xlim([0.0, 1.0])
    ax[2].set_ylim([0.0, 1.05])
    ax[2].set_title(f'{model_name} - Precision-Recall Curve')

    plt.tight_layout()
    
    ### SAVES PLOTS ###
    plot_filename = f"{save_dir}/{model_name.replace(' ', '_')}_Evaluation_Plots.png"
    plt.savefig(plot_filename)
    plt.show()

    # DEDUG LINE #
    print(f"\nPlot Saved: {plot_filename}\n")
# ===================================================================================================================================== #

# =========================================================== TOKENIZATION ============================================================ #
# ============================= DROPOUT CONFIGURATION ============================= #
config = DistilBertConfig(
    num_labels=2,                       # BINARY CLASSIFICATION
    hidden_dropout_prob=0.65,           # HIDDEN LAYER DROPOUT - INITIAL 0.3
    attention_probs_dropout_prob=0.65   # ATTENTION LAYER DROPOUT - INITIAL 0.3
)

### DEVICE SELECTION: Selects the appropriate device for Training: GPU, MPS (Apple Silicon) or CPU ###
if torch.cuda.is_available():
    print("Using CUDA (GPU) for Training\n\n")
    device = "cuda"
elif torch.backends.mps.is_available():
    print("Using MPS for Training\n\n")
    device = "mps"
else:
    print("Using CPU for Training\n\n")
    device = "cpu"

### LOAD TOKENIZER & MODEL ###
model_name = "distilbert-base-uncased"   # FASTER...
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config) 

# ============================ TOKENIZATION FOR DATASET ============================ #
def tokenize_dataset(texts, labels):
    encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=128)
    return TweetDataset(encodings, labels)

# ========================== CONVERSION TO PyTorch TENSORS ========================== #
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

### TOKENIZES DATASETS ###
train_dataset = tokenize_dataset(train_df['text'].tolist(), train_df['label'].tolist())
print("\nTraining dataset tokenization completed!\n")

test_dataset = tokenize_dataset(test_df['text'].tolist(), test_df['label'].tolist())
print("\nTesting dataset tokenization completed!\n")
# ===================================================================================================================================== #

# ==================================================== PARAMETERS & HYPERPARAMETERS =================================================== #
### METRIC COMPUTATION FUNCTION ###
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


### TRAINING ARGUMENTS ###
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=128,  # INITIAL 64
    per_device_eval_batch_size=128,   # INITIAL 64
    num_train_epochs=5,               # INITIAL 10
    weight_decay=0.05,                # INITIAL 0.01
    learning_rate=2e-5,               # INITIAL 2e-3
)

# Print debugging information
print("TRAINING PARAMETERS :")
print(f"- Epochs: {training_args.num_train_epochs}")
print(f"- Batch Size (Train): {training_args.per_device_train_batch_size}")
print(f"- Batch Size (Eval): {training_args.per_device_eval_batch_size}")
print(f"- Weight Decay: {training_args.weight_decay}")
print(f"- Learning Rate: {training_args.learning_rate}\n")
# ===================================================================================================================================== #

# ==================================================== PARAMETERS & HYPERPARAMETERS =================================================== #
### INITILISES TRAINER WITH compute_metrics FUNCTION ###
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# DEBUG LINE #
print("\nTraining in progress...\n")

### TRAINS MODEL ###
trainer.train()

# DEBUG LINE #
print("\nTraining Complete!\n")


### EVALUATES MODEL ###
evaluation = trainer.evaluate()
print(evaluation)

### TEST DATASET PREDICTION ###
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

### EVALUATION PLOTS ###
plot_graphs(test_df['label'].tolist(), pred_labels, model_name="DistilBERT Twittos Sentiment Analysis")
# ===================================================================================================================================== #