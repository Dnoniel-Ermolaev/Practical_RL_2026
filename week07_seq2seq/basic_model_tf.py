"""TF2 eager Keras port of the original TF1 graph-mode seq2seq model.

The original module relied on tf.placeholder, tf.scan, tf.nn.rnn_cell and an
explicit tf.Session. We replace all of that with eager-mode Keras layers, while
keeping the public API the notebook expects: a model with `symbolic_score`,
`symbolic_translate`, `weights` and helper functions `initialize_uninitialized`,
`infer_length`, `infer_mask`, `select_values_over_last_axis`.
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L


class BasicTranslationModel(keras.Model):
    def __init__(self, name, inp_voc, out_voc, emb_size, hid_size):
        super().__init__(name=name)
        self.inp_voc = inp_voc
        self.out_voc = out_voc

        self.emb_inp = L.Embedding(len(inp_voc), emb_size)
        self.emb_out = L.Embedding(len(out_voc), emb_size)
        self.enc0 = L.GRU(hid_size, return_sequences=False, return_state=True)
        self.dec_start = L.Dense(hid_size)
        self.dec0 = L.GRUCell(hid_size)
        self.logits = L.Dense(len(out_voc))

        # build all weights with a dummy forward pass
        dummy_inp = tf.zeros([1, 2], dtype=tf.int32)
        dummy_out = tf.zeros([1, 2], dtype=tf.int32)
        _ = self.symbolic_score(dummy_inp, dummy_out)

    @property
    def weights_list(self):
        return self.trainable_variables

    def encode(self, inp):
        inp_emb = self.emb_inp(inp)
        _, enc_last = self.enc0(inp_emb)
        return [self.dec_start(enc_last)]

    def decode_step(self, prev_state, prev_tokens):
        [prev_dec] = prev_state
        prev_emb = self.emb_out(prev_tokens)
        new_dec_out, [new_dec_state] = self.dec0(prev_emb, [prev_dec])
        return [new_dec_state], self.logits(new_dec_out)

    def symbolic_score(self, inp, out, eps=1e-30):
        inp = tf.cast(inp, tf.int32)
        out = tf.cast(out, tf.int32)
        first_state = self.encode(inp)
        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_ix)
        first_logits = tf.math.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        logits_seq = [first_logits]
        state = first_state
        T = int(out.shape[1]) if out.shape[1] is not None else tf.shape(out)[1]
        for t in range(int(T) - 1):
            state, logits = self.decode_step(state, out[:, t])
            logits_seq.append(logits)
        logits_stack = tf.stack(logits_seq, axis=1)
        return tf.nn.log_softmax(logits_stack, axis=-1)

    def symbolic_translate(self, inp, greedy=False, max_len=None, eps=1e-30):
        inp = tf.cast(inp, tf.int32)
        first_state = self.encode(inp)
        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_ix)
        first_logits = tf.math.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        if max_len is None:
            max_len = int(inp.shape[1] or 20) * 2

        out_seq = [bos]
        logits_seq = [first_logits]
        state = first_state
        for _ in range(max_len):
            state, logits = self.decode_step(state, out_seq[-1])
            if greedy:
                y_new = tf.cast(tf.argmax(logits, axis=-1), bos.dtype)
            else:
                y_new = tf.cast(tf.random.categorical(logits, 1)[:, 0], bos.dtype)
            logits_seq.append(logits)
            out_seq.append(y_new)
        out_stack = tf.stack(out_seq, axis=1)
        logits_stack = tf.stack(logits_seq, axis=1)
        return out_stack, tf.nn.log_softmax(logits_stack, axis=-1)


def initialize_uninitialized(*_, **__):
    """No-op in TF2 eager mode."""
    return None


def infer_length(seq, eos_ix, time_major=False, dtype=tf.int32):
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_ix), dtype)
    count_eos = tf.cumsum(is_eos, axis=axis, exclusive=True)
    return tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), dtype), axis=axis)


def infer_mask(seq, eos_ix, time_major=False, dtype=tf.float32):
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_ix, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major:
        mask = tf.transpose(mask)
    return mask


def select_values_over_last_axis(values, indices):
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size = tf.shape(indices)[0]
    seq_len = tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0, batch_size)[:, None], [1, seq_len])
    time_i = tf.tile(tf.range(0, seq_len)[None, :], [batch_size, 1])
    indices_nd = tf.stack([batch_i, time_i, tf.cast(indices, tf.int32)], axis=-1)
    return tf.gather_nd(values, indices_nd)
