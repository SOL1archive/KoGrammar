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

Attention Block에는 두가지 종류가 존재한다. Encoder Block과 Decoder Block이다. Encoder Block의 구조는 앞서 설명한 Attention Block과 동일하다. Decoder Block에는 입력에서 출력해야 하는 지점 이후의 시퀀스를 보지 못하도록 마스킹이 적용된다. 대부분의 transformer 모델들은 Encoder Block만을 활용한 Encoder-Only 모델, Decoder Block만을 활용한 Decoder-Only 모델, 둘 다 활용하는 Encoder-Decoder 모델이 존재한다. Encoder 모델의 대표적인 예로는 BERT 계열, Decoder 모델의 대표적 예로는 GPT 계열이 존재한다. 그리고 Encoder-Decoder 모델의 대표적인 예론 T5, BART가 존재한다. 일반적으로 Encoder-Only 모델은 NLU(Natural Language Understanding) Task에, Decoder-Only 모델은 NLG(Natural Language Generation) Task에서 좋은 성능을 발휘한다. Encoder-Decoder 모델은  시퀀스를 입력받고 다른 시퀀스를 출력하는 Task(가령 Question-Answering)에 주로 활용된다.



## Instruction Tuning

Instruction Tuning은 기존의 transfer learning의 한계를 개선하기 위해 고안된 해결책이다. Instruction Tuning은 입력 시퀀스에 원하는 Task를 명시적으로 포함시켜 모델의 성능을 높이는 방법이다. Instruction Tuning에 대한 가장 기초적인 아이디어는 _Exploring the limits of transfer learning with a unified text-to-text transformer_ (Raffel, C. et al., 2020)에서 제시되었다. 해당 논문에선 모델 T5 (_Text-to-Text Transfer Transformer_)을 학습시킬 때 입력 시퀀스 앞에 dataset의 이름을 포함시켜 어떤 Task에 대한 데이터셋인지를 명시했다. 그 결과 단순하게 입력 시퀀스만 입력하여 학습시킬 때보다 더 높은 성능을 얻을 수 있었다. Instruction Tuning을 적용시킨 모델은 Transfer Learning에서 좋은 성능을 낼 뿐만 아니라 학습하지 않은 새로운 Task를 매우 적은 데이터(2~3개에서 10개)만을 학습시킨 후 수행하도록 하는 Few-Shot Learning, 아예 데이터를 학습시키지 않고 새로운 Task를 수행하도록 하는 Zero-Shot Learning에서도 더 좋은 성능을 보였다.



## Distillation

오늘날의 최신 언어 모델의 규모는 매우 거대하다. 기본 용량만 수 GB에서 수십 GB를 상회하는 것까지, 모델의 규모가 점점 커지고 있다. 이러한 언어 모델의 성능은 매우 좋지만 추론과 추가 학습에 오랜 시간과 많은 비용을 필요로 하고, Fine-tuning의 난이도가 높다. 따라서 거대 모델이 배운 지식을 정제(Distillation)하여 작은 모델이 이를 배우도록 하는 시도가 많이 나타났다. 가장 기본적인 시도는 거대 모델(Teacher Model 혹은 Baseline Model)의 출력을 Label로 하여 작은 모델을 학습시키는 것이다.

> Distillation 추가 작성 필요

# Project Introduction

한국어, 영어를 포함한 대부분의 자연어의 문법은 맥락에 의존한다. 프로그래밍 언어, 수학과 같은 대부분의 인공어가 문맥 자유 언어(CFL: Context Free Language)인 것과는 대비적이다. 이것이 자연어의 문법 교정을 어렵게 하는 요인이다. 다른 한편으로, 상대적으로 매우 적은 인구만 사용하고, 언어학적으로도 고립어, 교착어에 속하는 한국어와 같은 언어의 언어 교정에서는 문제가 한 층 더 어려워진다. 정형적 언어 구조론이 많이 연구된 다른 언어의 문법 교정 방식을 적용하기 매우 어렵기 때문이다.

그럼에도 대표적인 자연어 처리 Benchmark인 GLUE에도 문법 교정 Task는 포함되어 있지 않다. 또 국내에서는 LSTM Seq2Seq 기반의 문법 교정 모델만 연구되었다. (조승우 등, 2018) 하지만 LSTM Seq2Seq는 Transformer 모델보다 성능이 일반적으로 떨어지고, 최신의 자연어 처리 연구 결과를 적용하기가 다소 어렵다. 따라서 한국어 문법 교정 모델 개발을 주제로 하기로 결정했다. 우선 한국어 데이터셋으로 학습된 KoT5, KoBART와 같은 Encoder-Decoder Transformer 모델을 이용하여 문법 교정 Task에 대한 Baseline 모델을 개발하고 Baseline 모델이 습득한 지식을 작은 모델에 정제한다. 이를 통해 작은 모델에서 더 높은 성능을 기대할 수 있을 뿐만 아니라 모바일 on-device과 같이 다양한 플랫폼에 쉽게 이식시키고, 빠른 시간에 적은 리소스만을 차지하는 문법 교정 모델을 개발하고자 한다.



### Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, *30*.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, *21*(1), 5485-5551.

조승우, 권홍석, 정헌영, & 이종혁. (2018). Encoder-Decoder 기반 한국어 문법 오류 교정을 위한 Encoder 에서의 신경망 언어 모델 도입. *정보과학회 컴퓨팅의 실제 논문지*, *24*(6), 301-306.



> Reference는 APA Style로 작성됨.