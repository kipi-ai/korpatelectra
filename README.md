# 특허분야 한국어 AI언어모델 KorPatELECTRA
KorPatELECTRA(Korean Patent ELECTRA)는 [한국특허정보원](https://www.kipi.or.kr)이 연구개발한 AI 언어모델입니다. 
<br>특허분야 한국어 자연어처리 문제 해결 및 특허산업분야의 지능정보화 인프라 마련을 위해 기존 [Google ELECTRA](https://github.com/google-research/electra) 모델의 아키텍쳐를 기반으로 대용량 국내 특허문헌(약 만건, 문장, 억 토큰, GB)을 사전학습(pre-training)하였고, 무료로 제공하고 있습니다.

## 
- [1. KorPatELECTRA](#1-korpatelectra)
- [2. KorPatELECTRA 개요](#2-korpatelectra-개요)
  - [2-1. 사전학습 환경](#2-1-사전학습-환경)
  - [2-2. 코퍼스](#2-2-코퍼스)
  - [2-3. 사전 및 토크나이저](#2-3-사전-및-토크나이저)
  - [2-4. 평가](#2-4-평가)
- [3. KorPatELECTRA 사용안내](#3-korpatelectra-사용안내)
  - [3-1. 요구사항](#3-1-요구사항)
  - [3-2. 토크나이저](#3-2-토크나이저)
  - [3-3. 파인튜닝](#3-3-파인튜닝)
- [4. KorPatELECTRA 정책 및 제공](#4-korpatelctra-정책-및-제공)
  - [4-1. 담당부서 및 모델제공 문의](#4-1-담당부서-및-모델제공-문의)
  - [4-2. 사용신청서](#4-2-사용신청서)
  - [4-3. 협약서](#4-3-협약서)
  - [4-4. 라이선스](#4-4-라이선스)
 
&nbsp;
## 1. KorPatBERT
특허분야 특화된 고성능 사전학습(pre-trained) 언어모델로 다양한 자연어처리 태스크에서 활용 할 수 있습니다.

&nbsp;
![KorPatBERT](./imgs/korpatbert.png)

&nbsp;
## 2. KorPatELECTRA 개요
### 2-1. 사전학습 환경
#### 개발환경
- Anaconda >=4.6.8
- Python >= 3.6
- MSP Tokenizer(Mecab-ko Sentencepiece Patent Tokenizer)
- Tensorflow-gpu >= 1.15.0
- Sentencepiece >= 0.1.96
- Horovod >= 0.19.2
#### 학습환경
- 특허문헌 120GB 코퍼스의 4억 6천만 문장 학습
- NVIDIA V100 32GB GPU 16개로 분산학습 라이브러리 Horovod를 이용하여 학습
- NVIDIA AMP(Automated Mixed Precision) 방식을 활용하여, 메모리 최적화
- 128 Sequence 2,300,000 Step 학습 + 512 Sequence 750,000 Step 학습

### 2-2. 코퍼스
- 특허문헌수 : 건
- 문장 수 : 건
- 토큰 수 : 약 억건
- 코퍼스 크기 : 약 GB

### 2-3. 사전 및 토크나이저


### 2-4. 평가
- 특허데이터 기반 CPC 분류 태스크
	- 144 labels, train data 351,487, dev data 39,053, test data 16,316


&nbsp;
## 3. KorPatELECTRA 사용안내
### 3-1. 요구사항

### 3-2. 토크나이저

	
### 3-3. 파인튜닝
※ [Google electra](https://github.com/google-research/electra) 학습 방식과 동일하며, 사용 예시는

&nbsp;
## 4. KorPatELECTRA 정책 및 제공
### 4-1. 담당부서 및 모델제공 문의
- 담당부서 : IP디지털혁신센터 지능정보전략팀
- 모델제공 및 기타문의 : ai_support@kipi.or.kr

#### 제공 순서
1. 사용신청서를 내려받아 작성하여 이메일(ai_support@kipi.or.kr)을 통해 제출합니다.
2. 담당자로부터 회신이 오면 협약서에 서명하여 이메일로 송부합니다.
3. 언어모델 및 사용자 매뉴얼을 제공받습니다.
4. 추후 본 언어모델을 활용하여 상업적 용도로의 사용을 위해서는 라이센스부분을 추가 작성하여 담당자에 보내주시면 관련내용 협의 후 진행이 가능합니다.
   
####  제공 파일
### 4-2. 사용신청서
모델 및 코드를 사용하고자 하시면 사용신청서를 작성하여 제출해주세요.


### 4-3. 협약서
모델 및 코드를 사용할 경우 협약서 내용을 준수해주세요.


### 4-4. 라이선스
모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 특히 상업적 활용을을 위해서는 사전협의가 필요합니다.

