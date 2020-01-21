from __future__ import absolute_import, division, print_function

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument(
    '--pretrained_model',
    default='./exp_clevr/tfmodel/clevr_gt_layout/00050000')
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))

from models_clevr_adv.nmn3_assembler import Assembler
from models_clevr_adv.nmn3_model import NMN3Model
from util_new.clevr_train.data_reader import DataReader
import skimage.io
import cv2


def extract_image_and_mask(imlist):
    img = []
    mask = []
    for impath in imlist:
         im = skimage.io.imread(impath)[..., :3]
         edges = cv2.Canny(im,100,200)
         M = np.zeros(im.shape,dtype=np.uint8)
         xmin = max(np.min(np.where(edges!=0)[0]) - 2, 0)
         xmax = min(np.max(np.where(edges!=0)[0]) + 2,im.shape[0])
         ymin = max(np.min(np.where(edges!=0)[1]) - 2, 0)
         ymax = min(np.max(np.where(edges!=0)[1]) + 2,im.shape[1])
         M[xmin:xmax,ymin:ymax,:] = 1
         #im = im*M
         M = M.astype(np.bool)
         img.append(im)
         mask.append(M)
    img = np.array(img)
    mask = np.array(mask)
    img = img.reshape((1,320,480,3))
    mask = mask.reshape((1,320,480,3))
    return img,mask

# vgg_net_new.tfmodel is pretrained vgg with tensor names prefixed by 'neural_module_network/image_feature_cnn/'
vgg_net_model = 'exp_clevr/vgg_net_new.tfmodel'
image_basedir = 'exp_clevr/clevr-dataset/images/'


# Module parameters
H = 320
W = 480
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 512
num_layers = 2
# It's better to have the same setting as eval_shapes.py instead of training it here than testing on eval_clevr_new.py
encoder_dropout = False
decoder_dropout = False
decoder_sampling = False
T_encoder = 45
T_decoder = 10
N = 1
prune_filter_module = True

# Training parameters
invalid_expr_loss = 0.5  # loss value when the layout is invalid
lambda_entropy = 0.005
weight_decay = 5e-6
baseline_decay = 0.99
max_grad_l2_norm = 10
max_trials = 3
max_iter = 1500
log_interval = 20
exp_name = "clevr_rl_gt_layout"
snapshot_file = './clevr_checkpoint/clevr_rl_gt_layout/tfmodel/00050000'

# Data files
vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
vocab_answer_file = './exp_clevr/data/answers_clevr.txt'

imdb_file_trn = './exp_clevr/data/imdb/adv_train_v2.npy'

assembler = Assembler(vocab_layout_file)

data_reader_trn = DataReader(imdb_file_trn, shuffle=False, one_pass=False,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file,
                             prune_filter_module=prune_filter_module)

num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab

# Network inputs
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
expr_validity_batch = tf.placeholder(tf.bool, [None])
answer_label_batch = tf.placeholder(tf.int32, [None])
image_batch = tf.placeholder(tf.float32, [None, H, W, 3])
masking_batch = tf.placeholder(tf.bool, [None, H, W, 3])
grad_scaling_batch = tf.placeholder(tf.float32, [None])
# The model for training
nmn3_model_trn = NMN3Model(
    image_batch, masking_batch, grad_scaling_batch,input_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim, num_layers=num_layers,
    assembler=assembler,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    decoder_sampling=decoder_sampling,
    num_choices=num_choices)



compiler = nmn3_model_trn.compiler
scores = nmn3_model_trn.scores
log_seq_prob = nmn3_model_trn.log_seq_prob

# Loss function
softmax_loss_per_sample =     tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=scores, labels=answer_label_batch)
# The final per-sample loss, which is vqa loss for valid expr
# and invalid_expr_loss for invalid expr
final_loss_per_sample = tf.where(expr_validity_batch,
    softmax_loss_per_sample,
    tf.ones_like(softmax_loss_per_sample) * invalid_expr_loss)

# Totoal training loss:
#     loss = E[ (C - b) * \diff[log(p(x))] + \diff[C] ]
# (where C = -R is the cost/loss; b is baseline)
avg_sample_loss = tf.reduce_mean(final_loss_per_sample)
baseline = tf.Variable(invalid_expr_loss, trainable=False, dtype=tf.float32)
baseline_update_op = tf.assign_add(baseline,
    (1-baseline_decay) * (avg_sample_loss-baseline))
policy_gradient_loss = tf.reduce_mean(
    tf.stop_gradient(final_loss_per_sample-baseline)*log_seq_prob)

total_training_loss = policy_gradient_loss + avg_sample_loss
total_loss = tf.add_n([total_training_loss,
                       lambda_entropy * nmn3_model_trn.entropy_reg,
                       weight_decay * nmn3_model_trn.l2_reg])

freeze_vars = []
train_vars = []
count = 0

for v in tf.trainable_variables():
    if count>0:
        freeze_vars.append(v)
    else:
        train_vars.append(v)
    count+=1

for v in train_vars:
    print(v.name,v.get_shape())


# Train with Adam
solver = tf.train.AdamOptimizer()
gradients = solver.compute_gradients(total_loss, var_list = train_vars)

# Clip gradient by L2 norm
# gradients = gradients_part1+gradients_part2


gradients = [(tf.clip_by_norm(g, max_grad_l2_norm), v)
             for g, v in gradients]
solver_op = solver.apply_gradients(gradients)

# Training operation
# Partial-run can't fetch training operations
# some workaround to make partial-run work
with tf.control_dependencies([solver_op, baseline_update_op]):
    train_step = tf.constant(0)

# Write summary to TensorBoard


loss_ph = tf.placeholder(tf.float32, [])
entropy_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
baseline_ph = tf.placeholder(tf.float32, [])
validity_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("avg_sample_loss", loss_ph))
summary_trn.append(tf.summary.scalar("entropy", entropy_ph))
summary_trn.append(tf.summary.scalar("avg_accuracy", accuracy_ph))
# summary_trn.append(tf.summary.scalar("baseline", baseline_ph))
summary_trn.append(tf.summary.scalar("validity", validity_ph))
log_step_trn = tf.summary.merge(summary_trn)

tst_answer_accuracy_ph = tf.placeholder(tf.float32, [])
tst_layout_accuracy_ph = tf.placeholder(tf.float32, [])
tst_layout_validity_ph = tf.placeholder(tf.float32, [])
summary_tst = []
summary_tst.append(tf.summary.scalar("test_answer_accuracy", tst_answer_accuracy_ph))
summary_tst.append(tf.summary.scalar("test_layout_accuracy", tst_layout_accuracy_ph))
summary_tst.append(tf.summary.scalar("test_layout_validity", tst_layout_validity_ph))
log_step_tst = tf.summary.merge(summary_tst)





avg_accuracy = 0
accuracy_decay = 0.99
# Load previous model
count = 0
n2nmn_vars = []
for v in tf.trainable_variables():
	if count>26:
		n2nmn_vars.append(v)
	count += 1

vgg_vars = []
count = 0
# changing <26 to <=26
for v in tf.trainable_variables():
    if count>0 and count<=26:
        vgg_vars.append(v)
    count+=1

vgg_names = [v.name for v in vgg_vars]
print(vgg_names)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=None,var_list=vgg_vars)
saver.restore(sess, vgg_net_model)
snapshot_saver = tf.train.Saver(max_to_keep=None,var_list=n2nmn_vars)
snapshot_saver.restore(sess, snapshot_file)
adv_var = tf.trainable_variables()[0]
print(adv_var.name)
adv_image_array = []
code = []
question = []
adversarial_label = []
pred = []
adv_iters = []
l2_norms = []
layout = []
text_atts = []
correct = 0

for enum, batch in enumerate(data_reader_trn.batches()):
    #if enum<=800:
    #    continue
    if enum >= 1000:
        break
    batch['image_batch'], batch['masking_batch'] = extract_image_and_mask(batch['image_path_list'])
    success = []
    C_val = [25]
    adv_image_j = []
    W_adv3_j = []
    adv_iter_j = []
    score_val_j = []
    l2_distortion = []
    for j in range(max_trials):
        sess.run(tf.variables_initializer([adv_var]))
        #saver = tf.train.Saver(max_to_keep=None,var_list=vgg_vars)
        #saver.restore(sess, vgg_net_model)
        #snapshot_saver = tf.train.Saver(max_to_keep=None,var_list=n2nmn_vars)
        #snapshot_saver.restore(sess, snapshot_file)
        grad_scaling = C_val[j]
        for n_iter in range(max_iter):
            # set up input and output tensors
            h = sess.partial_run_setup(
            [gradients, nmn3_model_trn.adv_input, nmn3_model_trn.W_adv3, nmn3_model_trn.atts,nmn3_model_trn.predicted_tokens, nmn3_model_trn.entropy_reg,
            scores, avg_sample_loss, train_step],
            [grad_scaling_batch, input_seq_batch, seq_length_batch, image_batch,
            masking_batch,compiler.loom_input_tensor, expr_validity_batch,
            answer_label_batch])

            # Part 0 & 1: Run Convnet and generate module layout
            adv_array, W_adv3, att, tokens, entropy_reg_val = sess.partial_run(h,
            (nmn3_model_trn.adv_input, nmn3_model_trn.W_adv3, nmn3_model_trn.atts, nmn3_model_trn.predicted_tokens, nmn3_model_trn.entropy_reg),
            feed_dict={grad_scaling_batch: [grad_scaling],
                       input_seq_batch: batch['input_seq_batch'],
                       seq_length_batch: batch['seq_length_batch'],
                       image_batch: batch['image_batch'],
                       masking_batch: batch['masking_batch']})
            # Assemble the layout tokens into network structure
            expr_list, expr_validity_array = assembler.assemble(tokens)
            # all exprs should be valid (as it's in the decoder)
            assert(np.all(expr_validity_array))

            labels = batch['answer_label_batch']
            actual_labels = batch['actual_answer_label_batch']
            # Build TensorFlow Fold input for NMN
            expr_feed = compiler.build_feed_dict(expr_list)
            expr_feed[expr_validity_batch] = expr_validity_array
            expr_feed[answer_label_batch] = labels

            # Part 2: Run NMN and learning steps
            grad, scores_val, avg_sample_loss_val, _ = sess.partial_run(
            h, (gradients, scores, avg_sample_loss, train_step), feed_dict=expr_feed)

            # compute accuracy
            predictions = np.argmax(scores_val, axis=1)
            accuracy = np.mean(np.logical_and(expr_validity_array,
                                  predictions == labels))
            avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)
            validity = np.mean(expr_validity_array)
            if n_iter % log_interval == 0 or (n_iter+1) == max_iter:
                 print(enum, grad_scaling, scores_val[0,labels],scores_val[0,actual_labels])
            if accuracy == 1:
                success += [True]
                print("Succesful %d"%(enum))
                print(scores_val,labels)
                if any(not _ for _ in success):
                    last_false = len(success) - success[::-1].index(False) - 1
                    C_val+= [0.5 * (C_val[j] + C_val[last_false])]
                else:
                    C_val+= [C_val[j] * 0.5]
                l2_distortion.append(np.linalg.norm(adv_array - batch['image_batch']))
                W_adv3_j.append(W_adv3)
                adv_image_j.append(adv_array)
                adv_iter_j.append(n_iter)
                score_val_j.append(scores_val)
                break
            elif n_iter==max_iter-1:
                success += [False]
                print("Unsuccesful %d"%(enum))
                print(scores_val,labels)
                if any(_ for _ in success):
                    last_true = len(success) - success[::-1].index(True) - 1
                    C_val += [0.5 * (C_val[j] + C_val[last_true])]
                else:
                    C_val += [C_val[j] * 2]
    if any(_ for _ in success):
        correct += 1
        print(enum+1, correct)
        adv_image_array.append(adv_image_j[np.argmin(l2_distortion)])
        code.append(W_adv3_j[np.argmin(l2_distortion)])
        question.append(batch['input_seq_batch'])
        adversarial_label.append(labels)
        pred.append(score_val_j[np.argmin(l2_distortion)])
        adv_iters.append(adv_iter_j[np.argmin(l2_distortion)])
        l2_norms.append(np.min(l2_distortion))
        layout.append(tokens)
        text_atts.append(att)

    if enum%20==0 or enum==999:
        print("Saving model for",enum)
        os.makedirs("adv_clevr_new_wb/%d_iterations"%(enum+1))
        adv_image_array_np = np.array(adv_image_array)
        code_np = np.array(code)
        question_np = np.array(question)
        adversarial_label_np = np.array(adversarial_label)
        pred_np = np.array(pred)
        adv_iters_np = np.array(adv_iters)
        l2_norms_np = np.array(l2_norms)
        layout_np = np.array(layout)
        text_atts_np = np.array(text_atts)
        np.save("adv_clevr_new_wb/%d_iterations/adv_image_array.npy"%(enum+1),adv_image_array_np)
        np.save("adv_clevr_new_wb/%d_iterations/code.npy"%(enum+1),code_np)
        np.save("adv_clevr_new_wb/%d_iterations/question.npy"%(enum+1),question_np)
        np.save("adv_clevr_new_wb/%d_iterations/adversarial_label.npy"%(enum+1),adversarial_label_np)
        np.save("adv_clevr_new_wb/%d_iterations/pred.npy"%(enum+1),pred_np)
        np.save("adv_clevr_new_wb/%d_iterations/adv_iters.npy"%(enum+1),adv_iters_np)
        np.save("adv_clevr_new_wb/%d_iterations/l2_norms.npy"%(enum+1),l2_norms_np)
        np.save("adv_clevr_new_wb/%d_iterations/layout.npy"%(enum+1),layout_np)
        np.save("adv_clevr_new_wb/%d_iterations/text_atts.npy"%(enum+1),text_atts)
        adv_image_array = []
        code = []
        question = []
        adversarial_label = []
        pred = []
        adv_iters = []
        l2_norms = []
        layout = []
        text_atts = []

print(correct)
print(enum)
