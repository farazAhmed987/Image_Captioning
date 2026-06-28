# ============================================================
# IMAGE CAPTIONING ON MS COCO — KAGGLE GPU NOTEBOOK
# Optimized for: Kaggle Free GPU (dual T4, 30 GB VRAM)
# Dataset: MS COCO train2017 (via Kaggle datasets)
#
# All 5 optimization techniques implemented:
#   1. Data Generator / Lazy Loading
#   2. Pre-extracted CNN Features (saved to HDF5)
#   3. HDF5 / Memory-Mapped Storage
#   4. Gradient Accumulation
#   5. Mixed Precision (float16)
#   + Checkpoint & Resume across sessions
# ============================================================


# ============================================================
# CELL 1 — Install dependencies & verify GPU
# Run this first every new Kaggle session
# ============================================================
import subprocess
subprocess.run(["pip", "install", "pycocotools", "-q"])

import os
import gc
import json
import pickle
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version : {tf.__version__}")
print(f"GPUs available     : {tf.config.list_physical_devices('GPU')}")

# Verify Kaggle dual-T4 setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memory growth enabled on {len(gpus)} GPU(s)")
else:
    print("WARNING: No GPU detected — training will be slow on CPU")


# ============================================================
# CELL 2 — OPTIMIZATION 5: Enable Mixed Precision (float16)
# Cuts VRAM usage ~40-50%, speeds up T4 by ~20%
# ============================================================
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy.name}")
print(f"Compute dtype : {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}")


# ============================================================
# CELL 3 — Configuration (all paths & hyperparameters)
# ============================================================

# --- Paths ---
# Kaggle working directory persists for 73 GB across sessions
WORK_DIR        = "/kaggle/working"
FEATURES_H5     = os.path.join(WORK_DIR, "coco_features.h5")       # HDF5 feature store
TOKENIZER_PATH  = os.path.join(WORK_DIR, "tokenizer.pkl")
PARAMS_PATH     = os.path.join(WORK_DIR, "model_params.pkl")
CHECKPOINT_DIR  = os.path.join(WORK_DIR, "checkpoints")
FINAL_MODEL     = os.path.join(WORK_DIR, "final_caption_model.keras")
TRAINING_LOG    = os.path.join(WORK_DIR, "training_log.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# MS COCO paths on Kaggle
# Add dataset: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
COCO_DIR        = "/kaggle/input/coco-2017-dataset"
TRAIN_IMG_DIR   = os.path.join(COCO_DIR, "train2017", "train2017")
VAL_IMG_DIR     = os.path.join(COCO_DIR, "val2017", "val2017")
TRAIN_ANN_FILE  = os.path.join(COCO_DIR, "annotations", "annotations",
                               "captions_train2017.json")
VAL_ANN_FILE    = os.path.join(COCO_DIR, "annotations", "annotations",
                               "captions_val2017.json")

# --- Hyperparameters ---
MAX_LENGTH       = 40          # max caption length in tokens
VOCAB_SIZE_LIMIT = 10000       # keep top N words; reduces softmax cost
EMBED_DIM        = 256
LSTM_UNITS       = 256
DENSE_UNITS      = 256
DROPOUT_RATE     = 0.3

# --- Training settings ---
EPOCHS           = 10
BATCH_SIZE       = 32          # actual batch loaded into VRAM per step
ACCUM_STEPS      = 4           # Gradient Accumulation: effective batch = 32*4 = 128
LEARNING_RATE    = 1e-3

# --- Dataset subset (safety valve for time limits) ---
# Set to None to use ALL ~118k training images
# Set to e.g. 20000 to use a 20k subset (faster, good for testing)
MAX_TRAIN_IMAGES = None

START_TOKEN = "<start>"
END_TOKEN   = "<end>"
UNK_TOKEN   = "<unk>"
PAD_TOKEN   = "<pad>"

print("Configuration loaded.")
print(f"  Features HDF5   : {FEATURES_H5}")
print(f"  Checkpoint dir  : {CHECKPOINT_DIR}")
print(f"  Effective batch : {BATCH_SIZE * ACCUM_STEPS}")


# ============================================================
# CELL 4 — Load MS COCO annotations
# ============================================================
print("\n=== Loading MS COCO Annotations ===")

with open(TRAIN_ANN_FILE, "r") as f:
    train_ann_data = json.load(f)

with open(VAL_ANN_FILE, "r") as f:
    val_ann_data = json.load(f)

# Build: {image_id -> filename}
train_id_to_file = {img["id"]: img["file_name"]
                    for img in train_ann_data["images"]}
val_id_to_file   = {img["id"]: img["file_name"]
                    for img in val_ann_data["images"]}

# Build: {filename -> [list of captions]}
train_captions = {}
for ann in train_ann_data["annotations"]:
    fname = train_id_to_file[ann["image_id"]]
    cap   = f"{START_TOKEN} {ann['caption'].strip().lower()} {END_TOKEN}"
    train_captions.setdefault(fname, []).append(cap)

val_captions = {}
for ann in val_ann_data["annotations"]:
    fname = val_id_to_file[ann["image_id"]]
    cap   = f"{START_TOKEN} {ann['caption'].strip().lower()} {END_TOKEN}"
    val_captions.setdefault(fname, []).append(cap)

# Optional subset
if MAX_TRAIN_IMAGES is not None:
    keys = list(train_captions.keys())[:MAX_TRAIN_IMAGES]
    train_captions = {k: train_captions[k] for k in keys}

train_image_names = list(train_captions.keys())
val_image_names   = list(val_captions.keys())

print(f"Training images  : {len(train_image_names)}")
print(f"Validation images: {len(val_image_names)}")
print(f"Sample caption   : {train_captions[train_image_names[0]][0]}")


# ============================================================
# CELL 5 — Build vocabulary & Keras Tokenizer
# Saved to disk so it survives session restarts
# ============================================================
print("\n=== Building Vocabulary ===")

if os.path.exists(TOKENIZER_PATH) and os.path.exists(PARAMS_PATH):
    print("Found existing tokenizer — loading from disk...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(PARAMS_PATH, "rb") as f:
        params = pickle.load(f)
    vocab_size      = params["vocab_size"]
    max_length      = params["max_length"]
    idx_to_word     = params["idx_to_word"]
    word_to_idx     = params["word_to_idx"]
    print(f"Loaded — vocab_size={vocab_size}, max_length={max_length}")

else:
    print("Building tokenizer from scratch...")
    from tensorflow.keras.preprocessing.text import Tokenizer

    all_captions_flat = []
    for caps in train_captions.values():
        all_captions_flat.extend(caps)

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE_LIMIT,
        oov_token=UNK_TOKEN,
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(all_captions_flat)

    # +1 because Keras index starts at 1
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_SIZE_LIMIT + 1)
    max_length = MAX_LENGTH

    # Build fast lookup dictionaries
    word_to_idx = {w: i for w, i in tokenizer.word_index.items()
                   if i < vocab_size}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    # Save tokenizer and params
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(PARAMS_PATH, "wb") as f:
        pickle.dump({
            "vocab_size"      : vocab_size,
            "max_length"      : max_length,
            "idx_to_word"     : idx_to_word,
            "word_to_idx"     : word_to_idx,
            "end_token_string": END_TOKEN,
        }, f)

    print(f"Tokenizer built — vocab_size={vocab_size}, max_length={max_length}")
    print(f"Saved to: {TOKENIZER_PATH}, {PARAMS_PATH}")


# ============================================================
# CELL 6 — OPTIMIZATION 2 & 3: Pre-extract CNN features → HDF5
#
# ResNet50 runs ONCE over the entire dataset.
# Features are saved to a single HDF5 file (~300–500 MB for 118k images).
# All future training epochs read tiny float vectors, not raw images.
# This is the single biggest RAM + speed improvement.
# ============================================================
import h5py
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

def extract_and_save_features(image_names, img_dir, h5_file, split_name):
    """
    Extract ResNet50 features for images not yet in the HDF5 file.
    Skips images already processed (safe to resume after disconnect).
    """
    # Load ResNet50 without the classification head
    feature_extractor = ResNet50(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    feature_extractor.trainable = False

    print(f"\nExtracting features for {split_name} split ({len(image_names)} images)...")

    # OPTIMIZATION 3: Open HDF5 file for append
    with h5py.File(h5_file, "a") as hf:
        # Create group for this split if it doesn't exist
        grp = hf.require_group(split_name)
        already_done = set(grp.keys())
        to_process   = [n for n in image_names if n not in already_done]

        print(f"  Already extracted: {len(already_done)}")
        print(f"  Still to process : {len(to_process)}")

        if not to_process:
            print(f"  All {split_name} features already extracted. Skipping.")
            del feature_extractor
            gc.collect()
            return

        # Process in mini-batches to keep RAM usage low
        EXTRACT_BATCH = 64
        for batch_start in tqdm(range(0, len(to_process), EXTRACT_BATCH),
                                desc=f"Extracting {split_name}"):
            batch_names = to_process[batch_start: batch_start + EXTRACT_BATCH]
            batch_imgs  = []
            valid_names = []

            for img_name in batch_names:
                img_path = os.path.join(img_dir, img_name)
                try:
                    img = keras_image.load_img(img_path, target_size=(224, 224))
                    x   = keras_image.img_to_array(img)
                    x   = preprocess_input(x)
                    batch_imgs.append(x)
                    valid_names.append(img_name)
                except Exception as e:
                    print(f"\n  Warning: skipping {img_name} — {e}")
                    continue

            if not batch_imgs:
                continue

            # Predict features for the whole batch at once (GPU-efficient)
            batch_array    = np.stack(batch_imgs, axis=0)
            batch_features = feature_extractor.predict(batch_array, verbose=0)

            # Write immediately to HDF5
            for name, feat in zip(valid_names, batch_features):
                grp.create_dataset(name, data=feat.astype(np.float32))

        print(f"  Done. {split_name} features saved to {h5_file}")

    del feature_extractor
    gc.collect()


# Run feature extraction (skips if already done — safe to re-run)
extract_and_save_features(train_image_names, TRAIN_IMG_DIR,
                          FEATURES_H5, "train")
extract_and_save_features(val_image_names,   VAL_IMG_DIR,
                          FEATURES_H5, "val")

# Verify the HDF5 file
with h5py.File(FEATURES_H5, "r") as hf:
    n_train = len(hf["train"])
    n_val   = len(hf["val"])
    sample_key  = list(hf["train"].keys())[0]
    sample_feat = hf["train"][sample_key][:]
    print(f"\nHDF5 verified — train: {n_train}, val: {n_val}")
    print(f"Feature shape per image: {sample_feat.shape}")  # should be (2048,)


# ============================================================
# CELL 7 — OPTIMIZATION 1 & 4: Data Generator + Gradient Accumulation
#
# The generator loads ONE BATCH at a time from HDF5.
# RAM usage stays under 1-2 GB regardless of dataset size.
# ============================================================
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence

class COCOCaptionGenerator(Sequence):
    """
    Keras Sequence data generator.
    - Reads image features lazily from HDF5 (never loads full dataset)
    - Generates (image_feature, partial_caption) → next_word pairs
    - Shuffles after every epoch automatically
    """

    def __init__(self, image_names, captions_dict, h5_file, split_name,
                 tokenizer, vocab_size, max_length, batch_size, shuffle=True):
        self.image_names  = image_names
        self.captions     = captions_dict
        self.h5_file      = h5_file
        self.split        = split_name
        self.tokenizer    = tokenizer
        self.vocab_size   = vocab_size
        self.max_length   = max_length
        self.batch_size   = batch_size
        self.shuffle      = shuffle

        # Pre-build a flat list of (image_name, caption_text) pairs
        self.pairs = []
        for img_name in image_names:
            for cap in captions_dict.get(img_name, []):
                self.pairs.append((img_name, cap))

        self.indices = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Keep HDF5 file handle open (faster repeated access)
        self._hf = h5py.File(h5_file, "r")

        print(f"Generator [{split_name}] — {len(self.pairs)} caption pairs, "
              f"{len(self)} batches per epoch")

    def __len__(self):
        return len(self.pairs) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:
                                     (idx + 1) * self.batch_size]
        X_img, X_seq, Y = [], [], []

        for i in batch_indices:
            img_name, caption = self.pairs[i]

            # Load feature from HDF5 (only this one vector, ~8 KB)
            try:
                feat = self._hf[self.split][img_name][:]
            except KeyError:
                continue   # image was skipped during extraction

            # Encode caption to integer sequence
            seq = self.tokenizer.texts_to_sequences([caption])[0]

            # Unroll caption into (partial_seq → next_word) training pairs
            for j in range(1, len(seq)):
                in_seq  = seq[:j]
                out_word = seq[j]

                in_seq = pad_sequences(
                    [in_seq], maxlen=self.max_length, padding="post"
                )[0]
                out_onehot = to_categorical(
                    [out_word], num_classes=self.vocab_size
                )[0]

                X_img.append(feat)
                X_seq.append(in_seq)
                Y.append(out_onehot)

        if not X_img:
            # Return empty batch (safety fallback)
            return (
                [np.zeros((1, 2048)), np.zeros((1, self.max_length))],
                np.zeros((1, self.vocab_size))
            )

        return (
            [np.array(X_img, dtype=np.float32),
             np.array(X_seq, dtype=np.int32)],
            np.array(Y, dtype=np.float32)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __del__(self):
        try:
            self._hf.close()
        except Exception:
            pass


# Instantiate generators
train_gen = COCOCaptionGenerator(
    image_names  = train_image_names,
    captions_dict= train_captions,
    h5_file      = FEATURES_H5,
    split_name   = "train",
    tokenizer    = tokenizer,
    vocab_size   = vocab_size,
    max_length   = max_length,
    batch_size   = BATCH_SIZE,
    shuffle      = True
)

val_gen = COCOCaptionGenerator(
    image_names  = val_image_names,
    captions_dict= val_captions,
    h5_file      = FEATURES_H5,
    split_name   = "val",
    tokenizer    = tokenizer,
    vocab_size   = vocab_size,
    max_length   = max_length,
    batch_size   = BATCH_SIZE,
    shuffle      = False
)


# ============================================================
# CELL 8 — Build the captioning model
# Same architecture as your original, upgraded for COCO scale
# ============================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LSTM,
                                     Embedding, add)

def build_captioning_model(vocab_size, max_length,
                           embed_dim, lstm_units, dense_units, dropout_rate):
    # Image feature branch
    img_input  = Input(shape=(2048,), name="image_features")
    img_drop   = Dropout(dropout_rate)(img_input)
    img_dense  = Dense(dense_units, activation="relu", name="img_dense")(img_drop)

    # Caption sequence branch
    seq_input  = Input(shape=(max_length,), name="caption_sequence")
    seq_embed  = Embedding(vocab_size, embed_dim, mask_zero=True,
                           name="embedding")(seq_input)
    seq_drop   = Dropout(dropout_rate)(seq_embed)
    seq_lstm   = LSTM(lstm_units, name="lstm")(seq_drop)

    # Merge & decode
    merged     = add([img_dense, seq_lstm])
    merged_d   = Dense(dense_units, activation="relu", name="merge_dense")(merged)
    # Output kept in float32 even with mixed precision
    output     = Dense(vocab_size, activation="softmax", dtype="float32",
                       name="word_output")(merged_d)

    model = Model(inputs=[img_input, seq_input], outputs=output,
                  name="coco_captioner")
    return model


# ============================================================
# CELL 9 — OPTIMIZATION 4: Gradient Accumulation Training Step
# Simulates effective batch of BATCH_SIZE * ACCUM_STEPS
# without exceeding VRAM limits
# ============================================================

class GradientAccumulationModel:
    """
    Wraps a Keras model to accumulate gradients over ACCUM_STEPS
    mini-batches before applying a single optimizer update.
    Effective batch size = BATCH_SIZE * ACCUM_STEPS
    """

    def __init__(self, model, accum_steps, learning_rate):
        self.model       = model
        self.accum_steps = accum_steps
        self.optimizer   = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn     = keras.losses.CategoricalCrossentropy()

        # Shadow gradient accumulators (one per trainable weight)
        self.accum_grads = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in model.trainable_variables
        ]

    def reset_accum(self):
        for ag in self.accum_grads:
            ag.assign(tf.zeros_like(ag))

    @tf.function
    def accum_step(self, X_img, X_seq, Y):
        """Compute gradients for one mini-batch and add to accumulators."""
        with tf.GradientTape() as tape:
            y_pred = self.model([X_img, X_seq], training=True)
            loss   = self.loss_fn(Y, y_pred)
            # Scale loss for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_grads = tape.gradient(scaled_loss,
                                     self.model.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)

        for ag, g in zip(self.accum_grads, grads):
            if g is not None:
                ag.assign_add(g / self.accum_steps)

        return loss

    def apply_accum(self):
        """Apply the accumulated gradients to update weights."""
        self.optimizer.apply_gradients(
            zip(self.accum_grads, self.model.trainable_variables)
        )
        self.reset_accum()


# ============================================================
# CELL 10 — Checkpoint & Resume Logic
#
# Saves after every epoch:
#   - Full model weights (.keras)
#   - Epoch number & best val loss (training_log.json)
# On resume: loads the latest checkpoint and continues from
# where training left off — even after Kaggle session timeout.
# ============================================================

def load_training_log():
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, "r") as f:
            return json.load(f)
    return {"epochs_done": 0, "best_val_loss": float("inf"), "history": []}

def save_training_log(log):
    with open(TRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2)

def get_latest_checkpoint():
    """Return path of the latest epoch checkpoint, or None."""
    checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".keras")
    ])
    if checkpoints:
        return os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    return None


def build_or_load_model():
    log = load_training_log()
    ckpt_path = get_latest_checkpoint()

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"\n=== Resuming from checkpoint: {ckpt_path} ===")
        print(f"    Epochs already done : {log['epochs_done']}")
        print(f"    Best val loss so far: {log['best_val_loss']:.4f}")
        model = tf.keras.models.load_model(ckpt_path, compile=False)
    else:
        print("\n=== No checkpoint found — building fresh model ===")
        model = build_captioning_model(
            vocab_size   = vocab_size,
            max_length   = max_length,
            embed_dim    = EMBED_DIM,
            lstm_units   = LSTM_UNITS,
            dense_units  = DENSE_UNITS,
            dropout_rate = DROPOUT_RATE
        )

    model.summary()
    return model, log


captioning_model, training_log = build_or_load_model()
ga_trainer = GradientAccumulationModel(
    captioning_model, ACCUM_STEPS, LEARNING_RATE
)


# ============================================================
# CELL 11 — Training Loop
#
# Combines all 5 optimizations:
#   1. Generator feeds batches lazily (lazy loading)
#   2. Features read from HDF5 (fast sequential SSD reads)
#   3. Gradient accumulation (effective large batch)
#   4. Mixed precision (float16 compute)
#   5. Checkpoint saved every epoch (resume-safe)
# ============================================================

def compute_val_loss(model, val_generator, num_val_batches=50):
    """Quick validation loss over a sample of val batches."""
    loss_fn = keras.losses.CategoricalCrossentropy()
    total_loss = 0.0
    count = 0
    for i in range(min(num_val_batches, len(val_generator))):
        (X_img, X_seq), Y = val_generator[i]
        y_pred = model([X_img, X_seq], training=False)
        total_loss += float(loss_fn(Y, y_pred))
        count += 1
    return total_loss / count if count > 0 else float("inf")


print("\n=== Starting / Resuming Training ===")
print(f"  Total epochs planned  : {EPOCHS}")
print(f"  Already done          : {training_log['epochs_done']}")
print(f"  Remaining             : {EPOCHS - training_log['epochs_done']}")
print(f"  Effective batch size  : {BATCH_SIZE * ACCUM_STEPS}")
print(f"  Mixed precision       : float16")
print(f"  Gradient accum steps  : {ACCUM_STEPS}")
print(f"  Steps per epoch       : {len(train_gen)}")

start_epoch = training_log["epochs_done"]

for epoch in range(start_epoch, EPOCHS):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch + 1} / {EPOCHS}  "
          f"[{datetime.datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    epoch_loss   = 0.0
    step_count   = 0
    accum_loss   = 0.0
    ga_trainer.reset_accum()

    train_gen.on_epoch_end()  # Shuffle at start of each epoch

    for step in range(len(train_gen)):
        (X_img, X_seq), Y = train_gen[step]

        # Convert to tensors
        X_img_t = tf.cast(tf.constant(X_img), tf.float32)
        X_seq_t = tf.cast(tf.constant(X_seq), tf.int32)
        Y_t     = tf.cast(tf.constant(Y),     tf.float32)

        # OPTIMIZATION 4: Accumulate gradients
        batch_loss = ga_trainer.accum_step(X_img_t, X_seq_t, Y_t)
        accum_loss += float(batch_loss)

        # Apply weights every ACCUM_STEPS mini-batches
        if (step + 1) % ACCUM_STEPS == 0:
            ga_trainer.apply_accum()
            epoch_loss += accum_loss / ACCUM_STEPS
            step_count += 1
            accum_loss  = 0.0

        # Print progress every 200 effective steps
        if step_count > 0 and step_count % 200 == 0:
            avg_so_far = epoch_loss / step_count
            print(f"  Step {step_count:>5} / {len(train_gen)//ACCUM_STEPS} "
                  f"— avg loss: {avg_so_far:.4f}  "
                  f"[{datetime.datetime.now().strftime('%H:%M:%S')}]")

    # Apply any remaining accumulated gradients
    if (len(train_gen) % ACCUM_STEPS) != 0:
        ga_trainer.apply_accum()

    avg_train_loss = epoch_loss / max(step_count, 1)

    # Validation loss
    print(f"\n  Computing validation loss...")
    val_loss = compute_val_loss(captioning_model, val_gen)
    print(f"  Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f}")

    # --- CHECKPOINT: Save after every epoch ---
    ckpt_name = (f"checkpoint_epoch_{epoch+1:02d}"
                 f"_valloss{val_loss:.4f}.keras")
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    captioning_model.save(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Save best model separately
    if val_loss < training_log["best_val_loss"]:
        training_log["best_val_loss"] = val_loss
        captioning_model.save(FINAL_MODEL)
        print(f"  *** New best model saved: {FINAL_MODEL} ***")

    # Update training log
    training_log["epochs_done"] = epoch + 1
    training_log["history"].append({
        "epoch"     : epoch + 1,
        "train_loss": round(avg_train_loss, 4),
        "val_loss"  : round(val_loss, 4),
        "timestamp" : datetime.datetime.now().isoformat()
    })
    save_training_log(training_log)
    print(f"  Training log updated: {TRAINING_LOG}")

    # Free memory between epochs
    gc.collect()
    tf.keras.backend.clear_session()

    # Rebuild GA trainer with saved model to free graph memory
    captioning_model = tf.keras.models.load_model(ckpt_path, compile=False)
    ga_trainer = GradientAccumulationModel(
        captioning_model, ACCUM_STEPS, LEARNING_RATE
    )

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"Best val loss  : {training_log['best_val_loss']:.4f}")
print(f"Final model at : {FINAL_MODEL}")
print(f"{'='*60}")


# ============================================================
# CELL 12 — Caption Generation (Greedy & Beam Search)
# Identical logic to your original app.py, adapted for COCO
# ============================================================

def extract_single_image_feature(img_path):
    """Extract ResNet50 feature for one new image at inference time."""
    feature_extractor = ResNet50(
        include_top=False, weights="imagenet",
        pooling="avg", input_shape=(224, 224, 3)
    )
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x   = keras_image.img_to_array(img)
    x   = preprocess_input(np.expand_dims(x, axis=0))
    feat = feature_extractor.predict(x, verbose=0)
    del feature_extractor
    gc.collect()
    return feat.squeeze()


def greedy_search(photo_feat, model, tokenizer, max_length, idx_to_word,
                  end_token=END_TOKEN):
    """Greedy caption generation — same as your original."""
    photo = photo_feat.reshape(1, 2048)
    in_text = START_TOKEN
    result_words = []

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")
        y_pred  = model.predict([photo, seq], verbose=0)
        word_idx = np.argmax(y_pred[0])
        word     = idx_to_word.get(word_idx)

        if word is None or word == end_token:
            break
        result_words.append(word)
        in_text += " " + word

    return " ".join(result_words)


def beam_search(photo_feat, model, tokenizer, max_length, idx_to_word,
                beam_k=3, end_token=END_TOKEN):
    """Beam search caption generation — same logic as your original."""
    photo = photo_feat.reshape(1, 2048)
    word_to_idx_local = {w: i for i, w in idx_to_word.items()}

    # Initial beam: [(text, log_prob)]
    beams = [(START_TOKEN, 0.0)]
    completed = []

    for _ in range(max_length):
        candidates = []
        for text, log_prob in beams:
            if text.split()[-1] == end_token:
                completed.append((text, log_prob))
                continue

            seq = tokenizer.texts_to_sequences([text])[0]
            seq = pad_sequences([seq], maxlen=max_length, padding="post")
            probs = model.predict([photo, seq], verbose=0)[0]

            # Take top beam_k next words
            top_k_idx = np.argsort(probs)[::-1][:beam_k]
            for idx in top_k_idx:
                word = idx_to_word.get(idx)
                if word is None:
                    continue
                new_log = log_prob + np.log(probs[idx] + 1e-9)
                candidates.append((text + " " + word, new_log))

        if not candidates:
            break

        # Keep top beam_k sequences
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_k]

    # Add remaining open beams to completed
    completed.extend(beams)
    if not completed:
        return ""

    best = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
    words = best.split()
    words = [w for w in words
             if w not in (START_TOKEN, END_TOKEN, UNK_TOKEN)]
    return " ".join(words)


# ============================================================
# CELL 13 — Evaluate on validation set (BLEU score)
# ============================================================
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

def evaluate_bleu(model, val_features_dict, val_captions_dict,
                  tokenizer, max_length, idx_to_word,
                  max_eval_images=500, beam_k=3):
    """
    Calculate average BLEU-4 score on validation images.
    max_eval_images: cap to avoid very long eval on free GPU
    """
    chencherry = SmoothingFunction()
    greedy_scores, beam_scores = [], []

    img_ids = list(val_features_dict.keys())[:max_eval_images]

    for img_name in tqdm(img_ids, desc="Evaluating BLEU"):
        feat = val_features_dict[img_name]
        refs = [
            [w for w in cap.split() if w not in (START_TOKEN, END_TOKEN)]
            for cap in val_captions_dict.get(img_name, [])
        ]
        if not refs:
            continue

        g_cap = greedy_search(feat, model, tokenizer, max_length,
                              idx_to_word).split()
        b_cap = beam_search(feat, model, tokenizer, max_length,
                            idx_to_word, beam_k).split()

        try:
            greedy_scores.append(sentence_bleu(
                refs, g_cap, smoothing_function=chencherry.method1))
            beam_scores.append(sentence_bleu(
                refs, b_cap, smoothing_function=chencherry.method1))
        except Exception:
            continue

    avg_greedy = np.mean(greedy_scores) if greedy_scores else 0
    avg_beam   = np.mean(beam_scores)   if beam_scores   else 0
    print(f"\nBLEU Evaluation ({len(img_ids)} images)")
    print(f"  Avg BLEU (Greedy)         : {avg_greedy:.4f}")
    print(f"  Avg BLEU (Beam k={beam_k}) : {avg_beam:.4f}")
    return avg_greedy, avg_beam


# Load val features from HDF5 into a dict for evaluation
print("\nLoading val features for evaluation...")
val_features_dict = {}
with h5py.File(FEATURES_H5, "r") as hf:
    for k in tqdm(list(hf["val"].keys())[:500], desc="Loading val features"):
        val_features_dict[k] = hf["val"][k][:]

# Load best model for evaluation
best_model = tf.keras.models.load_model(FINAL_MODEL, compile=False)

evaluate_bleu(
    model            = best_model,
    val_features_dict= val_features_dict,
    val_captions_dict= val_captions,
    tokenizer        = tokenizer,
    max_length       = max_length,
    idx_to_word      = idx_to_word,
    max_eval_images  = 500,
    beam_k           = 3
)


# ============================================================
# CELL 14 — Quick inference demo on a single image
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def caption_image(img_path, model, tokenizer, max_length, idx_to_word,
                  beam_k=3):
    """Generate captions for any image path and display it."""
    feat = extract_single_image_feature(img_path)

    greedy_cap = greedy_search(feat, model, tokenizer, max_length, idx_to_word)
    beam_cap   = beam_search(feat, model, tokenizer, max_length,
                             idx_to_word, beam_k)

    img = mpimg.imread(img_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Greedy: {greedy_cap}\nBeam (k={beam_k}): {beam_cap}",
              fontsize=10, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "sample_prediction.png"), dpi=150)
    plt.show()
    print(f"Greedy caption : {greedy_cap}")
    print(f"Beam caption   : {beam_cap}")

# Demo on the first val image
sample_img_name = val_image_names[0]
sample_img_path = os.path.join(VAL_IMG_DIR, sample_img_name)
caption_image(sample_img_path, best_model, tokenizer, max_length, idx_to_word)


# ============================================================
# CELL 15 — Update app.py to use the new COCO model
#
# Run this cell to print the minimal changes needed in app.py.
# Your Flask app.py already works — just update these 3 paths.
# ============================================================

print("""
=====================================================
 HOW TO UPDATE YOUR app.py FOR THE COCO MODEL
=====================================================

1. Copy these 3 files from Kaggle /kaggle/working/ to your
   local project folder (model_checkpoints/ & pickled_features/):

     final_caption_model.keras   →  model_checkpoints/
     tokenizer.pkl               →  pickled_features/
     model_params.pkl            →  pickled_features/

2. In app.py, update these 3 path constants (no other changes needed):

     MODEL_PATH     = 'model_checkpoints/final_caption_model.keras'
     TOKENIZER_PATH = 'pickled_features/tokenizer.pkl'
     PARAMS_PATH    = 'pickled_features/model_params.pkl'

3. Your existing generate_caption_greedy() function in app.py
   already reads end_token_string from model_params.pkl,
   so it will work correctly with the COCO model out of the box.

4. (Optional) To also enable beam search in app.py, add the
   beam_search() function from Cell 12 of this notebook.
=====================================================
""")
