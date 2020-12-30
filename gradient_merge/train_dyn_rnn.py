# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import ast
import paddle
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid
import numpy as np
import sys
import math
import argparse
paddle.enable_static()

CLASS_DIM = 2
EMB_DIM = 128
LSTM_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser("dyn_rnn")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu', type=int, default=1, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    parser.add_argument(
        '--max_step', type=int, default=10, help="max training step")
    parser.add_argument(
        '--base_optimizer', type=str, default='sgd', help="the basic optimizers")
    parser.add_argument(
        '--batch_size', type=int, default=64, help="batch size")
    parser.add_argument(
        '--use_gradient_merge',
        type=ast.literal_eval,
        default=False,
        help="Whether to use gradient merge.")
    parser.add_argument(
        '--embedding_type', type=str, default='dense', help="dense or sparse")

    args = parser.parse_args()
    return args


def dynamic_rnn_lstm(data, input_dim, class_dim, emb_dim, lstm_size):
    if args.embedding_type == "dense":
        emb = fluid.embedding(input=data, size=[input_dim, emb_dim], is_sparse=False, dtype="float32")
    elif args.embedding_type == "sparse":
        emb = fluid.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True, dtype="float32")
    else:
        print("not valid embedding type: ", args.embedding_type)
        exit()
    sentence = fluid.layers.fc(input=emb, size=lstm_size * 4, act='tanh')

    lstm, _ = fluid.layers.dynamic_lstm(sentence, size=lstm_size * 4, dtype="float32")
    last = fluid.layers.sequence_last_step(lstm)
    prediction = fluid.layers.fc(input=last, size=class_dim, act="softmax")
    return prediction


def inference_program(dict_dim):
    data = fluid.data(name="words", shape=[None], dtype="int64", lod_level=1)
    pred = dynamic_rnn_lstm(data, dict_dim, CLASS_DIM, EMB_DIM, LSTM_SIZE)
    return pred


def train_program(prediction):
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    if args.base_optimizer == "sgd":
        optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    elif args.base_optimizer == "adam":
        optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    elif args.base_optimizer == "adagrad":
        optimizer = fluid.optimizer.Adagrad(learning_rate=0.002)
    else:
        print("not valid optimizer: ", args.base_optimizer)
        exit()
    return optimizer


def train(use_cuda, params_dirname):
    fleet.init(is_collective=True)

    place = paddle.set_device('gpu') if use_cuda else fluid.CPUPlace()
    print("use_cuda: ", use_cuda)

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        star_program.random_seed = 90

    prediction = inference_program(5147)
    train_func_outputs = train_program(prediction)
    avg_cost = train_func_outputs[0]

    test_program = main_program.clone(for_test=True)

    strategy = fleet.DistributedStrategy()
    if args.use_gradient_merge:
        print("use gradient_merge")
        strategy.gradient_merge = True
        strategy.gradient_merge_configs = {'k_steps': 4}

    sgd_optimizer = optimizer_func()
    dist_optimizer = fleet.distributed_optimizer(sgd_optimizer, strategy=strategy)
    dist_optimizer.minimize(avg_cost)

    exe = fluid.Executor(place)

    feed_order = ['words', 'label']
    pass_num = args.num_epochs
    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len(train_func_outputs) * [0]
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=train_func_outputs)
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():

        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(fluid.default_startup_program())

        with open("main_program.txt", 'w') as f:
            f.write(str(main_program))

        print("Loading IMDB word dict....")
        word_dict = paddle.dataset.imdb.word_dict()
        print("word dict len: ", len(word_dict))

        print("Reading training data....")
        if args.enable_ce:
            train_reader = paddle.batch(
                paddle.dataset.imdb.train(word_dict), batch_size=args.batch_size)
        else:
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.imdb.train(word_dict), buf_size=25000),
                batch_size=args.batch_size)

        import time 
        start=time.time()
        scope = fluid.global_scope()
        for epoch_id in range(pass_num):
            for step_id, data in enumerate(train_reader()):
                metrics = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[var.name for var in train_func_outputs])
                if math.isnan(float(metrics[0])):
                    sys.exit("got NaN loss, training failed.")
                if step_id + 1 == args.max_step:
                    lstm_b0 = np.array(fluid.global_scope().find_var('lstm_0.b_0').get_tensor())
                    print(lstm_b0.flatten()[:10])
                    exit()
                end = time.time()
                if step_id % 10 == 0:
                    print("step: %d, \trnn_train_cost\t%f, %f steps/s" % (
                        step_id, 
                        metrics[0], 
                        (step_id+1) / (end-start)))
            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["words"],
                                              prediction, exe)
            if args.enable_ce and epoch_id == pass_num - 1:
                print("kpis\trnn_train_cost\t%f" % metrics[0])
                print("kpis\trnn_train_acc\t%f" % metrics[1])
                print("kpis\trnn_test_cost\t%f" % avg_cost_test)
                print("kpis\trnn_test_acc\t%f" % acc_test)

    train_loop()


def infer(use_cuda, params_dirname=None):
    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0))) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # Setup input by creating LoDTensor to represent sequence of words.
        # Here each word is the basic element of the LoDTensor and the shape of
        # each word (base_shape) should be [1] since it is simply an index to
        # look up for the corresponding word vector.
        # Suppose the length_based level of detail (lod) info is set to [[3, 4, 2]],
        # which has only one lod level. Then the created LoDTensor will have only
        # one higher level structure (sequence of words, or sentence) than the basic
        # element (word). Hence the LoDTensor will hold data for three sentences of
        # length 3, 4 and 2, respectively.
        # Note that lod info should be a list of lists.
        reviews_str = [
            'read the book forget the movie', 'this is a great movie',
            'this is very bad'
        ]
        reviews = [c.split() for c in reviews_str]

        UNK = word_dict['<unk>']
        lod = []
        for c in reviews:
            lod.append([np.int64(word_dict.get(words, UNK)) for words in c])

        base_shape = [[len(c) for c in lod]]

        tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
        assert feed_target_names[0] == "words"
        results = exe.run(
            inferencer,
            feed={feed_target_names[0]: tensor_words},
            fetch_list=fetch_targets,
            return_numpy=False)
        np_data = np.array(results[0])
        for i, r in enumerate(np_data):
            print("Predict probability of ", r[0], " to be positive and ", r[1],
                  " to be negative for review \'", reviews_str[i], "\'")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_conv.inference.model"
    train(use_cuda, params_dirname)
    infer(use_cuda, params_dirname)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu  # set to True if training with GPU
    main(use_cuda)
