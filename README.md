# 특허분야 한국어 AI언어모델 KorPatELECTRA
KorPatELECTRA(Korean Patent ELECTRA)는 [한국특허정보원](https://www.kipi.or.kr)이 연구개발한 AI 언어모델입니다. 
<br>특허분야 한국어 자연어처리 문제 해결 및 특허산업분야의 지능정보화 인프라 마련을 위해 기존 [Google ELECTRA](https://github.com/google-research/electra) 모델의 아키텍쳐를 기반으로 대용량 국내 특허문헌(약 466만 문헌, 5.4억 문장, 445억 토큰, 130GB)을 사전학습(pre-training)하였고, 무료로 제공하고 있습니다.

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
![KorPatELECTRA](./imgs/korpatelectra.png)

&nbsp;
## 2. KorPatELECTRA 개요
### 2-1. 사전학습 환경
#### 개발환경
- Anaconda >=4.6.8
- Python >= 3.6
- MWP Tokenizer(Mecab-ko wordpiece Patent Tokenizer)
- Tensorflow-gpu >= 1.15.0

#### 학습환경
- 특허문헌 130GB 코퍼스의 5억 4천만 문장을 학습
- NVIDIA V100 32GB GPU 16개로 tensorflow의 mirrored strategy 병렬학습
- 128 train vaech size, 100만 step 학습

### 2-2. 코퍼스
- 특허문헌수 : 4,661,158문헌
- 문장 수 : 546,496,725문장
- 토큰 수 : 44,525,763,134건
- 코퍼스 크기 : 약 130GB

### 2-3. 사전 및 토크나이저
언어모델 학습에 사용된 특허문헌을 대상으로 약 666만개의 주요 명사 및 복합명사를 추출하였으며, 이를 한국어 형태소분석기 Mecab-ko의 사용자 사전에 추가 후 WordPiece를 통하여 Subword로 분할하는 방식의 특허 텍스트에 특화된 MWP 토크나이저(Mecab-ko Wordpiece Patent Tokenizer)입니다.
- Mecab-ko 특허 사용자 사전파일명 : pat_all_mecab_dic.csv (6,663,693개 용어)
- WordPiece 사전파일명 : vocab.txt  (35,000개 토큰)
- WordPiece 스페셜 토큰 : [PAD], [UNK], [CLS], [SEP], [MASK]


### 2-4. 평가
- 특허데이터 기반 자연어처리 태스크

|<center>모델</center>|<center>vocab len</center>|<center>Patent NER</br>(F1)</center>|<center>CPC code classification(ACC)</center>|<center>PatQuAD(EM/F1)</center>|
|:--:|:--:|:--:|:--:|:--:|
|Google BERT|11만|87.98|72.33|51.63|81.36|
|KoELECTRA|35000|87.47|72.98|72.45|88.09
|<b>KorPatELECTRA</b>|<b>35000</b>|<b>91.01</b>|<b>73.90</b>|<b>89.85</b>|


&nbsp;
## 3. KorPatELECTRA 사용안내
### 3-1. 요구사항

### 3-2. 토크나이저

	
### 3-3. 파인튜닝
※ [Google electra](https://github.com/google-research/electra) 모델학습 방식과 동일

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

