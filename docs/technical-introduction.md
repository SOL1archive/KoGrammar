# Related Works
## Transformer Model

Transformer model은 _Attention is All You Need_ (Vaswani, A. et al., 2017)에서 제시된 모델이다. Transformer 모델은 이전에 RNN 계열 모델에서 Long term dependency 문제를 해결하기 위해 고안된 보완 메커니즘인 Attention 메커니즘을 독자적인 메커니즘으로 구현한 모델이다. Transformer 모델에서 사용하는 Self-Attention (혹은 Multi-Head Attention)은 다음과 같이 구현된다.

$$
A(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

$$
\begin{gathered}
\text{MultiHead} (Q, K, V) = \text{Concat}(\text{head} _1, \text{head} _2, \cdots, \text{head} _h) W^O\\
\text{where head}_i = \text{Attention} (Q W_i^Q, KW_i^K, VW_i^V)
\end{gathered}
$$

개별 Attention은 Query vector들을 쌓은 행렬인 $Q$ , Key vector들을 쌓은 행렬의 전치인 $K^T$ 의 행렬곱으로 각 Query vector들과 Key vector들의 내적을 구한다. 그 결과 한 행에 각 query vector들의 내적값이 존재한다. 이를 $\sqrt{d_k}$ 로 나눈다. 이는 내적으로 인해 행렬 원소의 분산이 $d_k$ 이 된 것을 원래대로 되돌리기 위함이다. 이를 Scaled-Dot Attention이라 한다. 그리고 각 행에 대해 Softmax 연산을 수행한다. 그 결과를 Attention Score라고 한다. 이렇게 얻은 행렬을 Value vector들을 쌓은 행렬인 $V$ 와 곱하여 Attention 결과를 구한다. Multi-Head Attention은 여러 query, key, value matrix들에 대해 attention 결과를 구하고 이를 모두 합친 후 $W^O$ 행렬과 곱하여 행렬의 벡터들을 원래 크기로 선형변환시킨다.

그 결과를 self-attention의 입력값과 더한다. 이를 Residual connection이라 하는데, 딥러닝 모델의 고질적인 문제인 gradient vanishing 문제를 해결하기 위한 방법이다. 그 후 layer normalization을 수행하는데, 이는 각 layer 출력 결과의 평균과 표준편차를 일정 파라미터로 변경하는 정규화 연산이다. 그 후 FC layer에 대입되고 다시 residual connection과 layer normalization을 수행한다. 이 전체 과정이 하나의 Attention Block이고, transformer 모델은 기본적으로 이 Attention block을 쌓아 구성한다.

Attention Block에는 두가지 종류가 존재한다. Encoder Block과 Decoder Block이다. Encoder Block의 구조는 앞서 설명한 Attention Block과 동일하다. Decoder Block에는 입력에서 출력해야 하는 지점 이후의 시퀀스를 보지 못하도록 마스킹이 적용된다. 대부분의 transformer 모델들은 Encoder Block만을 활용한 Encoder-Only 모델, Decoder Block만을 활용한 Decoder-Only 모델, 둘 다 활용하는 Encoder-Decoder 모델이 존재한다. Encoder 모델의 대표적인 예로는 BERT 계열, Decoder 모델의 대표적 예로는 GPT 계열이 존재한다. 그리고 Encoder-Decoder 모델의 대표적인 예론 T5, BART가 존재한다. 일반적으로 Encoder-Only 모델은 NLU(Natural Language Understanding) Task에, Decoder-Only 모델은 NLG(Natural Language Generation) Task에서 좋은 성능을 발휘한다. Encoder-Decoder 모델은  

## Instruction Tuning

instruction tuning은 기존의 transfer learning의 한계를 개선하기 위해 고안된 해결책이다. 대다수의 transformer 기반 언어모델은 전체 입력 시퀀스에서 특정 토큰을 가리고(마스킹)이를 예측하는 방향으로 학습된다. 이를 MLE(Maximum Likelihood Estimation)이라 한다. 이는 



## Distillation

오늘날의 최신 언어 모델의 규모는 매우 거대하다. 기본 용량만 수 GB에서 수십 GB를 상회하는 것까지, 모델의 규모가 점점 커지고 있다. 이러한 언어 모델의 성능은 매우 좋지만 추론과 추가 학습에 오랜 시간과 많은 비용을 필요로 하고, Fine-tuning의 난이도가 높다. 따라서 거대 모델이 배운 지식을 정제(Distillation)하여 작은 모델이 이를 배우도록 하는 시도가 많이 생겼다. 



### Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, *30*.



> Reference는 APA Style로 작성됨.