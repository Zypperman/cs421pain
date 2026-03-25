# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '19),
August 4-8, 2019, Anchorage, AK, USA. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3292500.3330871

Modifications
-------------
- Added --mode argument: 'train' (default) or 'infer'
  * train : original behaviour — train, evaluate on held-out split, save .h5 weights
  * infer : load saved weights, score a test CSV, write submission_devnet.npz
- Added F1 tracking: f1Performance() called alongside aucPerformance() every run
- writeResults() now includes mean_f1, std_f1, mean_prec, mean_rec columns
- Added --test_path argument: path to unlabelled test CSV for inference mode
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress FutureWarning spam

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.sparse import vstack, csc_matrix
from utils import (
    dataLoading, aucPerformance, f1Performance,
    writeResults, get_data_from_svmlight_file,
)
from sklearn.model_selection import train_test_split

import time
import pandas as pd

import zipfile

MAX_INT = np.iinfo(np.int32).max
data_format = 0


# ── Network architectures ────────────────────────────────────────────────────

def dev_network_d(input_shape):
    """Deeper network: three hidden layers (1000 → 250 → 20 → 1)."""
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name='hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name='hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_s(input_shape):
    """Shallow network: one hidden layer (20 → 1)."""
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
    intermediate = Dense(1, activation='linear', name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_linear(input_shape):
    """Linear mapping: raw inputs → anomaly score (no hidden layer)."""
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear', name='score')(x_input)
    return Model(x_input, intermediate)

def normalise_scores(scores, method="minmax"):
    """
    Map raw DevNet deviation scores (unbounded) to [0, 1].

    method="minmax"  — linear rescale; preserves rank, fast
    method="sigmoid" — maps the 5-sigma confidence margin to ~0.99;
                       more principled for probability-style submissions
    """
    s = scores.ravel().astype(np.float64)

    if method == "sigmoid":
        # Centre on the confidence margin (5.0) so that:
        #   score = 0  → 0.007  (well below normal)
        #   score = 5  → 0.993  (at the anomaly boundary)
        #   score > 5  → > 0.993
        normalised = 1.0 / (1.0 + np.exp(-(s - 5.0)))

    else:  # minmax
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-12:          # all scores identical (degenerate model)
            normalised = np.full_like(s, 0.5)
        else:
            normalised = (s - s_min) / (s_max - s_min)

    return normalised.astype(np.float32)


# ── Loss function ────────────────────────────────────────────────────────────

def deviation_loss(y_true, y_pred):
    """Z-score-based deviation loss (Algorithm 1 in the KDD'19 paper)."""
    confidence_margin = 5.
    # size=5000 is the setting of l in algorithm 1 in the paper
    ref = K.variable(np.random.normal(loc=0., scale=1.0, size=5000), dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss  = K.abs(dev)
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


# ── Model builder ────────────────────────────────────────────────────────────

def deviation_network(input_shape, network_depth):
    """Construct and compile the deviation network."""
    if network_depth == 4:
        model = dev_network_d(input_shape)
    elif network_depth == 2:
        model = dev_network_s(input_shape)
    elif network_depth == 1:
        model = dev_network_linear(input_shape)
    else:
        sys.exit("The network depth is not set properly")
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


# ── Batch generators ─────────────────────────────────────────────────────────

def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """Infinite batch generator used by fit_generator."""
    rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while 1:
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(
                x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(
                x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield (ref, training_labels)
        if counter > nb_batch:
            counter = 0


def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    """Dense CSV batches — alternates normal/anomaly samples."""
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))
    training_labels = []
    n_inliers  = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if i % 2 == 0:
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)


def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    """LibSVM sparse batches — alternates normal/anomaly samples."""
    ref = np.empty((batch_size,))
    training_labels = []
    n_inliers  = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if i % 2 == 0:
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


# ── Scoring ──────────────────────────────────────────────────────────────────

def load_model_weight_predict(model_name, input_shape, network_depth, x_test):
    """
    Load saved weights into a freshly built model and return anomaly scores.
    Works for both dense (data_format=0) and sparse (data_format=1) inputs.
    """
    model = deviation_network(input_shape, network_depth)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)

    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


# ── Noise injection (for contaminated training sets) ─────────────────────────

def inject_noise_sparse(seed, n_out, random_seed):
    """Add synthetic anomalies to sparse training data (5% feature swap)."""
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio  = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed  = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()


def inject_noise(seed, n_out, random_seed):
    """Add synthetic anomalies to dense training data (5% feature swap)."""
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio  = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise


# ── Helper: build the canonical model filename ───────────────────────────────

def _model_name(filename, args, network_depth):
    return (
        "./model/devnet_"
        + filename + "_"
        + str(args.cont_rate) + "cr_"
        + str(args.batch_size) + "bs_"
        + str(args.known_outliers) + "ko_"
        + str(network_depth) + "d.h5"
    )


# ── Main entry points ────────────────────────────────────────────────────────

def run_devnet(args):
    """
    Training mode (--mode train, the default).

    For every dataset and every run:
      1. Split into train / test (80 / 20, stratified).
      2. Optionally cap known outliers and inject noise.
      3. Train the deviation network, saving the best checkpoint.
      4. Score the held-out test split.
      5. Report AUC-ROC, AUC-PR, and best-threshold F1.
      6. Write a submission_devnet.npz with the scores from the last run.
    """
    os.makedirs("./model",   exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    names         = args.data_set.split(',')
    network_depth = int(args.network_depth)
    random_seed   = args.ramdn_seed

    for nm in names:
        runs     = args.runs
        rauc     = np.zeros(runs)
        ap       = np.zeros(runs)
        f1_arr   = np.zeros(runs)
        prec_arr = np.zeros(runs)
        rec_arr  = np.zeros(runs)

        filename = nm.strip()
        global data_format
        data_format = int(args.data_format)

        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()

        outlier_indices = np.where(labels == 1)[0]
        outliers        = x[outlier_indices]
        n_outliers_org  = outliers.shape[0]

        train_time = 0
        test_time  = 0

        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(
                x, labels, test_size=0.2, random_state=42, stratify=labels
            )
            y_train = np.array(y_train)
            y_test  = np.array(y_test)
            print(filename + ': round ' + str(i))

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices  = np.where(y_train == 0)[0]
            n_outliers      = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            n_noise = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)

            rng = np.random.RandomState(random_seed)
            if data_format == 0:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)

                noises  = inject_noise(outliers, n_noise, random_seed)
                x_train = np.append(x_train, noises, axis=0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx  = rng.choice(outlier_indices, mn, replace=False)
                    retain_idx  = list(set(np.arange(x_train.shape[0])) - set(remove_idx))
                    x_train     = x_train[retain_idx]
                    y_train     = y_train[retain_idx]

                noises  = inject_noise_sparse(outliers, n_noise, random_seed)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices  = np.where(y_train == 0)[0]
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)

            input_shape   = x_train.shape[1:]
            n_samples_trn = x_train.shape[0]
            n_outliers    = len(outlier_indices)
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            start_time   = time.time()
            epochs       = args.epochs
            batch_size   = args.batch_size
            nb_batch     = args.nb_batch
            model        = deviation_network(input_shape, network_depth)
            print(model.summary())
            model_name   = _model_name(filename, args, network_depth)
            checkpointer = ModelCheckpoint(
                model_name, monitor='loss', verbose=0,
                save_best_only=True, save_weights_only=True,
            )

            model.fit_generator(
                batch_generator_sup(x_train, outlier_indices, inlier_indices,
                                    batch_size, nb_batch, rng),
                steps_per_epoch=nb_batch,
                epochs=epochs,
                callbacks=[checkpointer],
            )
            train_time += time.time() - start_time

            # ── Evaluation on the held-out test split ────────────────────────
            start_time = time.time()
            scores     = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
            test_time += time.time() - start_time

            rauc[i], ap[i]                   = aucPerformance(scores, y_test)
            f1_arr[i], prec_arr[i], rec_arr[i], _ = f1Performance(scores, y_test)

        # ── Aggregate across runs ────────────────────────────────────────────
        mean_auc    = np.mean(rauc)
        std_auc     = np.std(rauc)
        mean_aucpr  = np.mean(ap)
        std_aucpr   = np.std(ap)
        mean_f1     = np.mean(f1_arr)
        std_f1      = np.std(f1_arr)
        mean_prec   = np.mean(prec_arr)
        mean_rec    = np.mean(rec_arr)
        train_time /= runs
        test_time  /= runs

        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        print("average F1: %.4f (Precision: %.4f, Recall: %.4f)" % (mean_f1, mean_prec, mean_rec))
        print("average runtime: %.4f seconds" % (train_time + test_time))

        writeResults(
            filename + '_' + str(network_depth),
            x.shape[0], x.shape[1],
            n_samples_trn, n_outliers_org, n_outliers,
            network_depth,
            mean_auc, mean_aucpr, std_auc, std_aucpr,
            mean_f1, std_f1, mean_prec, mean_rec,
            train_time, test_time,
            path=args.output,
        )

        # ── Write submission .npz (scores from the final run) ────────────────
        # scores is (n_test, 1); flatten to (n_test,) to match the format
        # produced by the logistic-regression notebooks.
        npz_path = "./results/submission_devnet.npz"
        np.savez(npz_path, predictions=scores.ravel())
        print("Submission scores saved to %s" % npz_path)


def run_devnet_infer(args):
    """
    Inference mode (--mode infer).

    Loads the pre-trained weights that match the naming convention used by
    run_devnet(), scores the CSV at --test_path, and writes:
        ./results/submission_devnet.npz   (predictions array, shape (n_users,))

    The test CSV must have the same feature columns as the training CSV but
    does NOT need a 'class' column — it is silently dropped if present.

    If --test_path is not provided, the function exits with an error message.
    """
    if not args.test_path:
        sys.exit(
            "Inference mode requires --test_path to be set.\n"
            "Example: python devnet.py --mode infer "
            "--test_path ./dataset/user_features_test.csv"
        )

    os.makedirs("./results", exist_ok=True)

    names         = args.data_set.split(',')
    network_depth = int(args.network_depth)

    global data_format
    data_format = int(args.data_format)

    for nm in names:
        filename   = nm.strip()
        model_name = args.model_path

        if not os.path.exists(model_name):
            sys.exit(
                "Model weights not found at: %s\n"
                "Run training first with: python devnet.py --mode train" % model_name
            )

        # ── Load test data ───────────────────────────────────────────────────
        print("Loading test data from: %s" % args.test_path)
        df_test = pd.read_csv(args.test_path)

        # Drop the label column if it was accidentally included
        if 'class' in df_test.columns:
            print("Note: 'class' column found in test CSV — dropping it for scoring.")
            y_test_ground_truth = df_test['class'].values  # kept for optional F1 eval below
            df_test = df_test.drop(['class'], axis=1)
        else:
            y_test_ground_truth = None

        x_test      = df_test.values
        input_shape = x_test.shape[1:]
        print("Test data shape: (%d, %d)" % x_test.shape)

        # ── Score ────────────────────────────────────────────────────────────
        print("Loading weights from: %s" % model_name)
        start_time    = time.time()
        scores        = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
        elapsed       = time.time() - start_time
        anomaly_scores = scores.ravel()   # shape (n_users,)

        anomaly_scores_norm = normalise_scores(anomaly_scores, method=args.score_norm)
        print("Scored %d users in %.4f seconds" % (len(anomaly_scores), elapsed))
        print("Score range (raw)        : [%.4f, %.4f]" % (anomaly_scores.min(), anomaly_scores.max()))
        print("Mean score : %.4f" % anomaly_scores.mean())
        print("Score range (normalised) : [%.4f, %.4f]" % (anomaly_scores_norm.min(), anomaly_scores_norm.max()))
        print("Mean score (normalised)  : %.4f" % anomaly_scores_norm.mean())

        # ── Optional: evaluate if ground truth is available ──────────────────
        if y_test_ground_truth is not None:
            print("\n--- Evaluation against ground-truth labels ---")
            aucPerformance(anomaly_scores, y_test_ground_truth)
            f1Performance(anomaly_scores, y_test_ground_truth)

        # ── Save .npz in the same format as the LR notebooks ─────────────────
        npz_path = args.output_npz
        np.savez(npz_path, predictions=anomaly_scores_norm)
        print("\nSubmission scores saved to %s" % npz_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="DevNet — Deep Anomaly Detection with Deviation Networks (KDD'19)"
)
parser.add_argument(
    "--mode", choices=['train', 'infer'], default='train',
    help=(
        "Execution mode. "
        "'train' (default): train the network, evaluate on a held-out split, "
        "and save weights + submission .npz. "
        "'infer': load saved weights and score the CSV at --test_path, "
        "writing submission_devnet.npz without any training."
    ),
)
parser.add_argument(
    "--test_path", type=str, default='',
    help=(
        "Path to the unlabelled test CSV used in inference mode. "
        "Feature columns must match the training CSV. "
        "A 'class' column is dropped automatically if present, "
        "but its values will be used for optional AUC/F1 reporting."
    ),
)
parser.add_argument(
    "--network_depth", choices=['1', '2', '4'], default='2',
    help="Network depth: 1=linear, 2=shallow (default), 4=deep",
)
parser.add_argument("--batch_size",      type=int,   default=512,  help="Batch size for SGD")
parser.add_argument("--nb_batch",        type=int,   default=20,   help="Number of batches per epoch")
parser.add_argument("--epochs",          type=int,   default=50,   help="Number of training epochs")
parser.add_argument("--runs",            type=int,   default=10,   help="Repetitions for average performance")
parser.add_argument("--known_outliers",  type=int,   default=200,  help="Number of labelled outliers available")
parser.add_argument("--cont_rate",       type=float, default=0.02, help="Outlier contamination rate in training data")
parser.add_argument("--input_path",      type=str,   default='./dataset/', help="Directory containing training CSVs")
parser.add_argument("--data_set",        type=str,   default='user_features_with_labels', help="Dataset name(s), comma-separated")
parser.add_argument("--data_format",     choices=['0', '1'], default='0', help="0=CSV, 1=libsvm")
parser.add_argument("--output",          type=str,   default='./results/results.csv', help="Output CSV for aggregated metrics")
parser.add_argument("--ramdn_seed",      type=int,   default=42,   help="Random seed")
parser.add_argument("--model_path",      type=str,   default='',   help="Path to saved model weights")
parser.add_argument("--output_npz",        type=str,   default='',   help="Path to save .npz submission file")
parser.add_argument("--score_norm", choices=['minmax', 'sigmoid'], default='minmax', help="Normalisation method for scores")

args = parser.parse_args()

if args.mode == 'train':
    run_devnet(args)
else:
    run_devnet_infer(args)