#coding:utf-8
import tensorflow as tf
from utils import *

class Model(object):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, Flag="DNN"):
        self.model_flag = Flag
        self.batch_size = BATCH_SIZE
        with tf.name_scope('Inputs'):
            self.mid_click_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_click_his_batch_ph')
            self.mid_unclick_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_unclick_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.click_mask = tf.placeholder(tf.float32, [None, None], name='click_mask_batch_ph')
            self.unclick_mask = tf.placeholder(tf.float32, [None, None], name='unclick_mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], trainable=True)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_click_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_click_his_batch_ph)
            self.mid_unclick_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_unclick_his_batch_ph)

        with tf.name_scope('init_operation'):    
            self.mid_embedding_placeholder = tf.placeholder(tf.float32,[n_mid, EMBEDDING_DIM], name="mid_emb_ph")
            self.mid_embedding_init = self.mid_embeddings_var.assign(self.mid_embedding_placeholder)

            
        self.item_eb = self.mid_batch_embedded
        self.item_click_his_eb = self.mid_click_his_batch_embedded * tf.reshape(self.click_mask, (BATCH_SIZE, SEQ_LEN, 1))
        self.item_unclick_his_eb = self.mid_unclick_his_batch_embedded * tf.reshape(self.unclick_mask, (BATCH_SIZE, 10*SEQ_LEN, 1))
        self.item_click_his_eb_sum = tf.reduce_sum(self.item_click_his_eb, 1)
        self.item_unclick_his_eb_sum = tf.reduce_sum(self.item_unclick_his_eb, 1)

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, scope='prelu_1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, scope='prelu_2')

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss

            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init,feed_dict={self.uid_embedding_placeholder: uid_weight})
    
    def init_mid_weight(self, sess, mid_weight):
        sess.run([self.mid_embedding_init],feed_dict={self.mid_embedding_placeholder: mid_weight})

    def save_mid_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding                                 
    
    def train(self, sess, inps):

        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_click_his_batch_ph: inps[2],
            self.mid_unclick_his_batch_ph: inps[3],
            self.click_mask: inps[4],
            self.unclick_mask: inps[5],
            self.target_ph: inps[6],
            self.lr: inps[7]
        })

        return loss, accuracy

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_click_his_batch_ph: inps[2],
            self.mid_unclick_his_batch_ph: inps[3],
            self.click_mask: inps[4],
            self.unclick_mask: inps[5],
            self.target_ph: inps[6],
        })

        return probs, loss, accuracy

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_TEM4CTR(Model):
    def __init__(
        self,
        n_uid,
        n_mid,
        EMBEDDING_DIM,
        HIDDEN_SIZE,
        BATCH_SIZE,
        SEQ_LEN=256,
    ):
        super(Model_TEM4CTR, self).__init__(
            n_uid,
            n_mid,
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            BATCH_SIZE,
            SEQ_LEN,
            Flag="TEM4CTR",
        )
        self.view = self.mid_embeddings_var
        with tf.name_scope("Attention_layer"):
            self.item_unclick_his_eb = tf.reshape(self.item_unclick_his_eb, (BATCH_SIZE, SEQ_LEN, -1, EMBEDDING_DIM))
            self.item_unclick_his_eb = din_attention_1(
                self.item_click_his_eb, self.item_unclick_his_eb, HIDDEN_SIZE, tf.reshape(self.unclick_mask, (BATCH_SIZE, SEQ_LEN, -1)), stag="search"
            )
            self.item_unclick_his_eb = tf.squeeze(self.item_unclick_his_eb)
            shared, unique = self.denoise(self.item_unclick_his_eb, self.item_click_his_eb)
            attention_output = din_attention(
                self.item_eb, self.item_click_his_eb+shared, HIDDEN_SIZE, self.click_mask, stag="click"
            )
            
            attention_output_unclick = din_attention(
                self.item_eb, self.item_unclick_his_eb, HIDDEN_SIZE, self.click_mask, stag="unclick"
            )

            att_fea1 = tf.reduce_sum(attention_output, 1)
            att_fea2 = tf.reduce_sum(attention_output_unclick, 1)

        inp = tf.concat([self.item_eb,  att_fea1, att_fea2], -1)
        self.build_fcn_net(inp, use_dice=False)
        
    def denoise(self, origin_emb, target_emb):
        res_array = tf.expand_dims(tf.reduce_sum(tf.multiply(origin_emb,target_emb),axis=-1),-1)*target_emb
        norm_num = tf.norm(target_emb, axis=-1)*tf.norm(target_emb, axis=-1)+1e-12
        clear_emb = res_array/tf.expand_dims(norm_num,-1)
        noise_emb = origin_emb - clear_emb
        return clear_emb, noise_emb