# Test gradient merge on Language Model

## Code
测试gradient merge [PR](https://github.com/PaddlePaddle/Paddle/pull/29784)。
测试code基于mapingshuo同学的[测试代码](https://github.com/mapingshuo/book/tree/sentiment/06.understand_sentiment)，有所修改，加上了fleet分布式。

## 对比
gradient merge与真实的大Batch在数学上理论是一致的。注:`在有batch_norm时，数学上不等价`
下列对比实验的结果理论应该一致。

| | batch size | 迭代次数 | 精度| optimizer | embedding |
| -- | -- | -- | -- | -- | -- |
| baseline | 64 | 10 | fp32 | sgd/adam | dense/sparse |
| gradient merge (avg) | 16 | 40 | fp32 | sgd/adam | dense/sparse |

## 运行
见`run.sh`脚本。
gradient merge:
``` bash
fleetrun --gpus 0,1 train_dyn_rnn.py \
    --num_epochs=1 \
    --enable_ce \
    --use_gradient_merge=True \
    --batch_size=16 \
    --max_step=40 \
    --base_optimizer=sgd \
    --embedding_type="dense"
```

baseline:
``` bash
fleetrun --gpus 0,1 train_dyn_rnn.py \
    --num_epochs=1 \
    --enable_ce \
    --batch_size=64 \
    --max_step=10 \
    --base_optimizer=sgd \
    --embedding_type="dense"
```

## 测试结果
打印lstm_0.b_0的前十个参数，数值精度误差范围内，相等。表中打印出lstm_0.b_0[0]的值。
| | sgd | adam | adagrad |
| -- | -- | -- | -- |
| baseline | -0.03897955 | -0.06657485 | -0.0089463 |
| gradient merge | -0.03897955 | -0.06657486 | -0.0089463 |

结论：在数值精度误差范围内gradient merge和baseline完全打平，说明gradient merge和真实的大Batch在数学上是一致的; 也说明PR代码多卡实现上没有问题。
