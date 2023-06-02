import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


def din_attention(
    query,
    facts,
    attention_size,
    mask=None,
    stag="null",
    mode="SUM",
    att_func="all",
    softmax_stag=1,
    time_major=False,
    return_alphas=False,
    keep_prob=0.8,
):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("query_size mismatch")
        query = tf.concat(
            values=[
                query,
                query,
            ],
            axis=1,
        )

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    facts_size = facts.get_shape().as_list()[
        -1
    ]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    if att_func == "all":
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
        d_layer_1_all = tf.layers.dense(
            din_all, 80, activation=tf.nn.relu, name="f1_att" + stag
        )
        d_layer_2_all = tf.layers.dense(
            d_layer_1_all, 40, activation=tf.nn.relu, name="f2_att" + stag
        )
        d_layer_3_all = tf.layers.dense(
            d_layer_2_all, 1, activation=None, name="f3_att" + stag
        )
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
    if att_func == "dot":
        din_all = tf.reduce_sum(queries * facts, axis=-1)
        scores = tf.reshape(din_all, [-1, 1, tf.shape(facts)[1]])
    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-(2 ** 32) + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    
    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        # scores = tf.nn.dropout(scores, keep_prob)

    # Weighted sum
    if mode == "SUM":
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))

    if return_alphas:
        return output, scores

    return output

def din_attention_1(
    query,
    facts,
    attention_size,
    mask=None,
    stag="null",
    mode="SUM",
    att_func="all",
    softmax_stag=1,
    return_alphas=False,
    keep_prob=0.8,
):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("query_size mismatch")
        query = tf.concat(
            values=[
                query,
                query,
            ],
            axis=1,
        )

    facts_size = facts.get_shape().as_list()[
        -1
    ]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, 1, tf.shape(facts)[-2]])
    queries = tf.reshape(queries, tf.shape(facts))
    if att_func == "all":
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
        d_layer_1_all = tf.layers.dense(
            din_all, 80, activation=tf.nn.relu, name="f1_att" + stag
        )
        d_layer_2_all = tf.layers.dense(
            d_layer_1_all, 40, activation=tf.nn.relu, name="f2_att" + stag
        )
        d_layer_3_all = tf.layers.dense(
            d_layer_2_all, 1, activation=None, name="f3_att" + stag
        )
        d_layer_3_all = tf.reshape(d_layer_3_all, [tf.shape(facts)[0], tf.shape(facts)[1], 1, -1])
        scores = d_layer_3_all
    if att_func == "dot":
        din_all = tf.reduce_sum(queries * facts, axis=-1)
        scores = tf.reshape(din_all, [tf.shape(facts)[0], tf.shape(facts)[1], 1, -1])
    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 2)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-(2 ** 32) + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    
    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        # scores = tf.nn.dropout(scores, keep_prob)

    # Weighted sum
    if mode == "SUM":
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))

    if return_alphas:
        return output, scores

    return output


def prelu(_x, scope=""):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable(
            "prelu_" + scope,
            shape=_x.get_shape()[-1],
            dtype=_x.dtype,
            initializer=tf.constant_initializer(0.1),
        )
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0.0, 0.0
    for record in arr:
        if record[1] == 1.0:
            pos += 1
        else:
            neg += 1

    fp, tp = 0.0, 0.0
    xy_arr = []
    for record in arr:
        if record[1] == 1.0:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * (y + prev_y) / 2.0
            prev_x = x
            prev_y = y

    return auc


def calc_gauc(raw_arr, nick_index):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    last_index = 0
    gauc = 0.0
    pv_sum = 0
    for idx in xrange(len(nick_index)):
        if nick_index[idx] != nick_index[last_index]:
            input_arr = raw_arr[last_index:idx]
            auc_val = calc_auc(input_arr)
            if auc_val >= 0.0:
                gauc += auc_val * len(input_arr)
                pv_sum += len(input_arr)
            else:
                pv_sum += len(input_arr)
            last_index = idx
    return gauc / pv_sum

