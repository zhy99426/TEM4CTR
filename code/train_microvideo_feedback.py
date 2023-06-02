# coding:utf-8
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model_microvideo import *
import time
import random
import sys
from utils import *
import multiprocessing
import argparse
import pickle as pkl
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p", type=str, default="train", help="train | test")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--gpus", type=str, default="0,", help="GPU IDs")


def generator_queue(generator, max_q_size=20, wait_time=0.1, nb_worker=1):

    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:

        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        # start_time = time.time()
                        generator_output = next(generator)
                        # end_time = time.time()
                        # print end_time - start_time
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception as e:
                    print(e)
                    _stop.set()
                    print("over1")
                    # raise

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception:
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()
        print("over")

    return q, _stop, generator_threads


EMBEDDING_DIM = 16
HIDDEN_SIZE = 16 * 2
best_auc = 0.0
early_count = 0


def prepare_data(src, target):
    nick_id, item_id, cate_id = src
    (
        label,
        hist_click,
        hist_unclick,
        hist_click_cate,
        hist_unclick_cate,
        hist_click_mask,
        hist_unclick_mask,
    ) = target

    return (
        nick_id,
        item_id,
        cate_id,
        label,
        hist_click,
        hist_unclick,
        hist_click_cate,
        hist_unclick_cate,
        hist_click_mask,
        hist_unclick_mask,
    )


def eval(sess, test_data, model, model_path, batch_size):
    loss_sum = 0.0
    accuracy_sum = 0.0

    nums = 0
    stored_arr = []
    test_data_pool, _stop, _ = generator_queue(test_data)

    while True:
        if _stop.is_set() and test_data_pool.empty():
            break
        if not test_data_pool.empty():
            src, tgt = test_data_pool.get()
        else:
            continue
        (
            nick_id,
            item_id,
            cate_id,
            label,
            hist_click,
            hist_unclick,
            hist_click_cate,
            hist_unclick_cate,
            hist_click_mask,
            hist_unclick_mask,
        ) = prepare_data(src, tgt)

        if len(nick_id) < batch_size:
            continue
        nums += 1
        target = label
        prob, loss, acc = model.calculate(
            sess,
            [
                nick_id,
                item_id,
                cate_id,
                hist_click,
                hist_unclick,
                hist_click_cate,
                hist_unclick_cate,
                hist_click_mask,
                hist_unclick_mask,
                label,
            ],
        )
        loss_sum += loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums

    global best_auc, early_count
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
        early_count = 0
    else:
        early_count += 1

    return test_auc, loss_sum, accuracy_sum


def train(
    train_file="./data/processed/MicroVideo-1.7M/microvideo_train_align.txt",
    test_file="./data/processed/MicroVideo-1.7M/microvideo_test_align.txt",
    feature_file="./data/processed/MicroVideo-1.7M/microvideo_feature_align.pkl",
    batch_size=256,
    maxlen_click=30,
    maxlen_unclick=300,
    test_iter=100,
    save_iter=100,
):

    best_model_path = "best_model/microvideo_ckpt/TEM4CTR"

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        train_data = DataIterator(
            train_file, batch_size, maxlen_click, maxlen_unclick, with_cate=True
        )
        test_data = DataIterator(
            test_file, batch_size, maxlen_click, maxlen_unclick, with_cate=True
        )

        feature_num = pkl.load(open(feature_file, "rb"))
        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen_click

        
        model = Model_TEM4CTR(
                n_uid, n_mid, 512, 512, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN
            )

        variables = tf.contrib.framework.get_variables_to_restore()

        variables_to_restore = [
            v
            for v in variables
            if v.name.split("/")[0] in ["mid_embedding_var:0", "emb_mtx"]
        ]
        print(variables_to_restore)
        saver = tf.train.Saver(variables_to_restore)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        feature = np.concatenate(
            [
                np.load("./data/MicroVideo-1.7M/train_cover_image_feature.npy"),
                np.load("./data/MicroVideo-1.7M/test_cover_image_feature.npy")
            ],
            axis=0,
        )
        feature = np.concatenate(
            [
                feature.mean(0,keepdims=True),
                feature
            ],
            axis=0,
        )
        
        model.init_mid_weight(sess, feature)

        sys.stdout.flush()
        print("training begin")
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.005
        for itr in range(1):
            print("epoch" + str(itr))
            loss_sum = 0.0
            accuracy_sum = 0.0
            train_data_pool, _stop, _ = generator_queue(train_data)
            while True:
                if _stop.is_set() and train_data_pool.empty():
                    break
                if not train_data_pool.empty():
                    src, tgt = train_data_pool.get()
                else:
                    continue
                (
                    nick_id,
                    item_id,
                    cate_id,
                    label,
                    hist_click,
                    hist_unclick,
                    hist_click_cate,
                    hist_unclick_cate,
                    hist_click_mask,
                    hist_unclick_mask,
                ) = prepare_data(src, tgt)
                # import pdb
                # pdb.set_trace()
                loss, acc = model.train(
                    sess,
                    [
                        nick_id,
                        item_id,
                        cate_id,
                        hist_click,
                        hist_unclick,
                        hist_click_cate,
                        hist_unclick_cate,
                        hist_click_mask,
                        hist_unclick_mask,
                        label,
                        lr,
                    ],
                )
                loss_sum += loss
                accuracy_sum += acc

                iter += 1
                sys.stdout.flush()
                #                if iter < 2500:
                #                    continue
                if (iter % test_iter) == 0:
                    print(
                        "iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f"
                        % (
                            iter,
                            loss_sum / test_iter,
                            accuracy_sum / test_iter,
                        )
                    )
                    print(
                        "                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f"
                        % eval(sess, test_data, model, best_model_path, batch_size)
                    )
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    test_time = time.time()
                    print(
                        "test interval: "
                        + str((test_time - start_time) / 60.0)
                        + " min"
                    )
                if (iter % save_iter) == 0:
                    print("save model iter: %d" % (iter))
                    # model.save(sess, model_path + "--" + str(iter))

            print(best_auc)


def test(
    train_file="./data/processed/MicroVideo-1.7M/microvideo_train_align.txt",
    test_file="./data/processed/MicroVideo-1.7M/microvideo_test_align.txt",
    feature_file="./data/processed/MicroVideo-1.7M/microvideo_feature_align.pkl",
    batch_size=256,
    maxlen_click=30,
    maxlen_unclick=300,
    test_iter=100,
    save_iter=100,
):

    model_path = "best_model/microvideo_ckpt/TEM4CTR"

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        test_data = DataIterator(
            test_file, batch_size, maxlen_click, maxlen_unclick, with_cate=True
        )
        feature_num = pkl.load(open(feature_file, "rb"))
        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen_click

        model = Model_TEM4CTR(
                n_uid, n_mid, 512, 512, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN
            )
        
        model.restore(sess, model_path)

        print(
            "test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f"
            % eval(sess, test_data, model, model_path, batch_size)
        )


if __name__ == "__main__":
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    if args.p == "train":
        train()
    elif args.p == "test":
        test()
    else:
        print("do nothing...")
