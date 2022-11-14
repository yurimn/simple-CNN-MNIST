# simple-CNN-MNIST

MNIST data accuracy 0.99 넘기기!
layer 3개 CNN 사용해서 성공

---

## About CNN : Convolutional Neural Networks (CNN) 요약 정리

원문 link : [https://cs231n.github.io/convolutional-networks/#architectures](https://cs231n.github.io/convolutional-networks/#architectures)

CNN, Convnet 이라고도 불림. Convolutional Neural networks의 줄임말.

## Layers usedto build ConvNets

[CIFAR-10 data : ml을 위한 공개 데이터셋] 사용. 10개의 클래스, 하나의 클래스 당 6000장의 이미지 포함.

여기서는 간단한 CNN 구현을 위해 다음과 같이 INPUT-CONV-RELU-POOL-FC architecture을 가진다고 했다.

-   INPUT : 32*32*3 image. width 32, heigt 32, color 3 (R, G, B)
-   CONV : 12개의 필터를 사용해 가중치를 부여. 32*32*12 로 바뀜
-   RELU : 활성화 함수. 음수면 0, 양수면 기울기 1인 직선.
-   POOL : 이미지의 크기 축소 & 특정 feature 강조 16*16*12로 줄임
-   FC : 1*1*10, 10개의 class에 대한 값으로 변환. ordinary neural network와 동일한 과정 수행.

이러한 방식으로, 원래 이미지 픽셀 값을 최종적으로 각각의 class에 대한 score로 바꾼다.

CONV/FC layer의 경우 activation function을 비롯해 여러 파라미터들 (wieght-가중치, biases-범위 조절을 위한 상수)이 필요하다. 그리고 경사하강법(gradient descent)으로 값을 찾아나간다.

RELU/POOL layers는 고정된 함수로 구현된다.

**요약**

-   ConvNet architecture은 주어진 이미지를 결과(class에 해당하는 score)로 변환하기 위한 가장 간단한 layer들이다.
-   몇개의 구별된 layer type이 존재하고. 위에서 설명한 4가지가 가장 대표적이다.
-   각각의 layer들을 3차원 데이터를 입력받고 미분가능한 함수를 통해 변환된 뒤 3차원으로 출력한다.
-   각각의 layer는 파라미터를 가질 수도 있고(Conv, FC), 아닐 수도 있다.(RELU, POOL)
-   각각의 layer는 추가적인 hyparameter를 가질 수 있다. (Conv,FC, POOL의 경우)

## Convnet architectures

### Convolution Layer

Conv layer는 Convolutional Netwrok를 구현하기 위한 가장 핵심이라고 할 수 있다.

**개요 및 직관적 관찰**

먼저 CONV layer가 어떤 계산과정을 거치는지 직관적으로 살펴보자.

이 레이어의 파라미터들은 learnable filters의 집합이다. 모든 필터는 작은 공간(가로*세로)이다. 예를 들어, 첫 번째 layer는 전형적으로 5*5\*3의 크기이다. 각각의 공간에서 필터를 슬라이싱하며 행렬곱을 계산한다.

**Local Connectivity**

이미지와 같이 높은 차원의 입력을 다룰 때에는, 이전 volume과 모든 뉴런을 연결하는 것이 비효율적이다. 그래서 대신에, 입력 volume의 local region에서의 각각의 뉴런만을 연결시킬 것이다.

뉴런의 receptive field라고 불리는 hyperparameter가 연결의 크기 정도라고 생각하면 된다. filter size와 같은 의미이다. 이 크기의 depth는 입력의 depth 크기와 동일하다. convolution layer는 이차원이지만, depth까지 곱한 값과 같은 connections를갖는다고 생각하면 된다.

**Spatial arrangement**

여기까지 conv layer에서의 input 공간으로 부터의 뉴런의 연결에 대해 설명했지만, 얼마나 많은 뉴런이 output 공간에 있고 어떻게 배열되는지는 설명하지 않았다. output volume을 조절하기 위해서는 depth, stride, zero-padding의 세 가지 hyperparameter가 필요하다.

1. 먼저, output volume의 depth는 hyparameter이다. 입력에서 눈으로 보기에 다른 것들의 개수, filter의 개수에 대응된다. input의 같은 공간의 뉴런들의 집합은 depth column이다. (fibre라고 하기도 한다.)
2. stride란, filter를 슬라이드 하는 범위, 길이라고 생각하면 된다. 예를 들어 stride가 1이면 filter를 1 pixel씩 미는것이다. stride는 output 크기를 줄이는 역할을 한다.
3. input의 가장자리에 0으로 패딩하면 편리하다. zero-padding의 size도 hyperparameter이다.

그러면 output volume을 다음과 같이 계산할 수 있다.

W = input volum size

S = stride

F = receptive field size of Conv Layer neurons (same as filter size)

P = amount of the zero padding

이라고 할 때, output의 뉴런의 개수는 다음과 같이 계산 가능하다.

$$
\frac {W-F+2*P} {S} +1
$$

**Parameter Sharing**

parameter sharing scheme는 파라미터의 개수를 조절하기 위해 사용된다. 현실에서 파라미터의 개수를 그대로 사용하면 첫 번째 레이어에서 뉴런의 개수가 너무 많아지기 때문이다. 그래서 하나의 depthslice에 해당되는 neuron들은 parameter를 sharing하게 된다.

### Pooling Layer

Conv layer 사이에 pooling layer를 삽입한다. overfitting를 컨트롤하고, 크기를 줄이기 위해 사용된다. 가장 흔한 pooling 형태는 MAX operation을 사용해 2\*2의 작은 공간에서부터 최댓값만을 뽑아내는 것이다. 그러면 데이터의 25%를 남기게 된다. 그러나 경우에 따라 pooling은 데이터를 손실하는것이기 때문에 pooling layer 자체를 없애기도 한다.

### Fully-Connected Layer

regular neural networks에서의 작동과 동일.

### **Layer Patterns**

흔히 다음의 패턴을 많이 따름.

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

## Case studies

convolutional networks에서 여러가지 구조가 있고, 대표적으로 다음들이 있다.

LeNet, AlexNet, GoogleNet, VGGNet, ResNet, etc…
