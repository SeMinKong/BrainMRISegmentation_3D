# 3D 모델 학습 가이드

이 프로젝트는 MU-Glioma-Post MRI 볼륨의 네 채널을 입력으로 사용하는 학습 코드를 제공합니다. 학습된 가중치와 환자 데이터는 포함하지 않습니다. 기본 웹 데모의 합성 MRI·마스크와 실제 모델 추론은 구분됩니다.

## 구현 범위

| 구성 | 현재 구현 |
| --- | --- |
| 3D U-Net | MONAI 인코더·디코더, residual unit, 네 채널 입력, 정수 클래스 출력 |
| Swin UNETR | 동일 manifest와 학습 루프에서 선택 가능한 MONAI Transformer 모델 |
| nnU-Net v2 | raw dataset 변환 CLI, 고정 환자 분할 파일, 설치·학습 완료 모델 폴더 추론 어댑터 |
| 학습 | 환자별 train/val 구분, RAS 방향 정리, 간격 재표본화, 비영 복셀 z-score, 전경 패치 샘플링, 공간 반전, Dice+CE, AdamW |
| 검증 | validation 볼륨 전체 sliding-window 추론, 클래스별 Dice와 평균, 체크포인트·메타데이터·JSON 학습 기록 |
| 추론 | 저장된 전처리/채널/모델 설정 검증, sliding-window, 원래 T1 크기·affine으로 복원, uint8 NIfTI 저장 |

nnU-Net은 단일 아키텍처 이름이 아니라 데이터에 맞춰 전처리와 모델·훈련 구성을 정하는 방법론입니다. 이 MVP의 수동 MONAI U-Net은 nnU-Net과 별개입니다. 구현 참고: [MONAI 네트워크](https://docs.monai.io/en/stable/networks.html), [MONAI sliding-window](https://monai.readthedocs.io/en/1.5.1/inferers.html), [nnU-Net 공식 저장소](https://github.com/MIC-DKFZ/nnUNet).

## 데이터 계약

- 입력: 공동 정합된 3D `.nii`/`.nii.gz` 파일 네 개. 내부 순서는 `t1,t1ce,t2,flair`이며 `t1n,t1c,t2w,t2f` 별칭도 허용합니다.
- 네 MRI와 정답 마스크의 shape·affine이 서로 다르면 거절합니다. 정합, 두개골 제거, DICOM 변환은 구현 범위에 포함하지 않습니다.
- NIfTI spatial unit은 mm를 사용합니다. 단위가 unspecified인 볼륨은 mm로 가정하므로 입력 파일의 실제 단위를 확인합니다. metre/micron으로 명시된 입력은 변환 후 사용해야 합니다.
- 전처리는 RAS 방향의 axis-aligned 격자로 변환하고 지정 mm 간격으로 MRI를 선형 보간합니다. 마스크는 최근접 보간합니다. 추론 마스크를 원래 T1 격자로 최근접 복원합니다. 재표본화에 따른 세부 경계 손실은 발생할 수 있습니다.
- 기본 라벨: `0 background / 1 NETC / 2 SNFH / 3 ET / 4 RC`. MU-Glioma-Post의 치료 후 교종 MRI를 위한 설정입니다. RC는 수술 후 절제강이며 종양 자체와 동일하지 않습니다. 정의와 다운로드는 [MU-Glioma-Post 공식 페이지](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)를 기준으로 합니다.
- 학습 manifest의 라벨은 0..4 범위에서 설정할 수 있지만, **웹 추론은 위 다섯 라벨의 의미까지 일치하는 체크포인트만 허용**합니다. 다른 의미의 마스크는 generic 프리셋으로 열람하며 자동으로 MU-Glioma-Post 라벨로 변환하지 않습니다.

## 설치와 작은 실행 확인

프로젝트 루트, 활성화한 Python 가상환경에서 실행합니다. `-E`는 외부 `PYTHONPATH`에 다른 프로젝트 패키지가 섞이지 않도록 합니다. 먼저 환경에 맞는 [공식 PyTorch 설치 방법](https://pytorch.org/get-started/locally/)으로 PyTorch를 설치한 다음 선택 의존성을 설치합니다.

```powershell
python -E -m pip install -r ml/requirements.txt
python -E -m ml.smoke --output runs/smoke
```

smoke 명령은 두 개의 작은 **합성** 환자 볼륨을 만들고 CPU에서 실제 optimizer step 1회, validation, 저장된 checkpoint의 실제 추론을 수행합니다. 최종 출력의 크기·affine·정수 라벨을 검증합니다. 의미 있는 분할 성능을 검증하는 실험은 아닙니다. `runs/smoke/best.pt`를 실데이터용 모델로 취급하지 마세요. `--fixture-only`는 PyTorch 없이 합성 NIfTI와 manifest만 생성합니다.

## MU-Glioma-Post 데이터 준비와 학습

데이터는 [MU-Glioma-Post 공식 페이지](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)에서 직접 받은 뒤 프로젝트 밖 또는 무시된 `data/`에 둡니다. [다운로드 안내](data-access.md)에 따라 실제 파일·환자·검사 구조를 확인하고 [manifest 예제](../configs/manifest.example.json)에 경로를 기록합니다. 원본 배포 파일명을 아직 확인하지 않았으므로 자동 탐색 호환성을 가정하지 않습니다.

`ml.manifest`를 사용하려면 `_seg` 또는 `-seg` 마스크 및 같은 접두사의 네 시퀀스가 있어야 합니다. 예를 들어 확인한 사례명이 `patient-001_timepoint-01` 형식일 때:

```powershell
python -E -m ml.manifest --data-root D:/datasets/MU-Glioma-Post --patient-regex '^(.+)_timepoint-\d+$' --output data/mu-glioma-post-manifest.json
```

자동 탐색은 **명시적인 `--patient-regex`를 요구**하며 첫 캡처 그룹을 환자 ID로 사용합니다. 배포본의 환자 식별 규칙에 맞는지 생성된 manifest를 직접 확인해야 합니다. 종단 검사도 같은 환자에 묶여야 합니다. 정규식이 맞지 않는 파일명은 자동 추측하지 않고 거절합니다. 검증 분할은 환자 ID에 고정 seed를 적용해 생성하며, train과 val에 같은 patient_id가 있으면 학습이 중단됩니다. 이 검사는 잘못 기입한 환자 ID까지 알아낼 수는 없습니다.

```powershell
python -E -m ml.train --manifest data/mu-glioma-post-manifest.json --model unet3d --output runs/unet3d --epochs 100 --patch-size 64 64 64 --spacing 1 1 1
python -E -m ml.train --manifest data/mu-glioma-post-manifest.json --model swinunetr --output runs/swinunetr --epochs 100 --patch-size 64 64 64
```

3D U-Net 기본 채널은 `16,32,64,128,256`, Swin UNETR의 feature size는 24입니다. Swin 패치는 각 변이 32의 배수이면서 32보다 커야 합니다. 기본 U-Net 패치는 16의 배수이면서 16보다 커야 합니다. GPU 메모리가 부족하면 `--channels 8 16 32 64` 또는 적절한 패치 크기로 U-Net 규모부터 줄입니다. 배치는 1이고 MRI 전체를 한 사례씩 메모리에 로드하므로 CPU RAM도 필요합니다. GPU용 혼합 정밀도, 디스크 캐시, 분산 학습은 후속 작업입니다.

`best.pt`, `last.pt`, `metadata.json`, `metrics.json`, `resolved-manifest.json`이 저장됩니다. 메타데이터에는 실제 optimizer step, 모델 설정, 라벨, 채널 순서, 전처리, seed, 환자 분할, manifest hash, 라이브러리 버전, validation Dice가 들어갑니다. 현재 optimizer 복원을 통한 중단 재개 기능은 없습니다.

검증 Dice는 재표본화된 전체 볼륨의 클래스별 값입니다. 정답과 예측이 둘 다 빈 클래스는 `null`이며 평균에서 제외합니다. 한쪽만 빈 경우는 0입니다. 평균은 유효한 case-class Dice의 산술평균입니다. 병변별 개별 평가는 수행하지 않습니다. 환자 단위 홀드아웃, 하이퍼파라미터 고정, 원래 격자 평가, 경계 거리·병변 단위 평가를 갖춘 실험으로 확장해야 합니다.

## 학습 모델을 웹에서 연결

서버를 실행하는 PowerShell에서 직접 관리하는 로컬 체크포인트만 등록합니다. 모델 선택기는 설정된 파일과 선택 패키지가 있을 때 사용 가능 상태를 표시합니다. 실제 추론 시작 시 체크포인트의 계약을 다시 검사합니다.

```powershell
$env:MRI_UNET_CHECKPOINT = (Resolve-Path runs/unet3d/best.pt).Path
$env:MRI_SWIN_CHECKPOINT = (Resolve-Path runs/swinunetr/best.pt).Path
$env:MRI_DEVICE = "auto"  # auto, cpu, cuda
# 이후 README의 서버 시작 명령 실행
```

웹 API는 사용자 입력으로 임의 체크포인트 경로를 받지 않습니다. PyTorch 체크포인트는 `weights_only=True`로 읽고 strict state_dict 로딩을 적용합니다. 학습된 가중치가 없으면 실제 모델 추론은 오류를 반환하며 합성 분할로 대체하지 않습니다.

## nnU-Net v2 학습 연결

nnU-Net은 기본 의존성이 아닙니다. 먼저 별도 환경에 설치하고 공식 환경 변수·학습 지침을 확인합니다. [데이터 형식](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md), [학습 지침](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).

```powershell
python -E -m pip install nnunetv2
$env:nnUNet_raw = "D:/nnunet/raw"
$env:nnUNet_preprocessed = "D:/nnunet/preprocessed"
$env:nnUNet_results = "D:/nnunet/results"
python -E -m ml.export_nnunet --manifest data/mu-glioma-post-manifest.json --output D:/nnunet/raw/Dataset501_MUGliomaPost
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
Copy-Item D:/nnunet/raw/Dataset501_MUGliomaPost/splits_final.json D:/nnunet/preprocessed/Dataset501_MUGliomaPost/splits_final.json
nnUNetv2_train 501 3d_fullres 0
$env:MRI_NNUNET_MODEL_DIR = "D:/nnunet/results/Dataset501_MUGliomaPost/nnUNetTrainer__nnUNetPlans__3d_fullres"
$env:MRI_NNUNET_FOLDS = "0"
```

학습·검증 사례는 `imagesTr/labelsTr`, test 사례가 있으면 `imagesTs/labelsTs`에 내보냅니다. 별도의 `splits_final.json`을 **전처리 후 학습 전에** 복사해야 원래 환자 분할이 유지됩니다. 기본 자동 교차검증 분할로 재생성하지 않도록 주의합니다. exporter는 연속적인 0..N 라벨만 허용합니다.

추론 어댑터는 `plans.json`, `dataset.json`, `fold_*/checkpoint_final.pth`가 있는 폴더를 읽습니다. 모델의 image reader/writer를 그대로 사용해 nnU-Net의 배열 축·spacing 규칙을 유지합니다. `dataset.json`의 네 channel_names는 t1/t1ce/t2/flair 또는 t1n/t1c/t2w/t2f 별칭이어야 하고 웹 라벨 의미도 일치해야 합니다. cascade 이전 단계와 region-based 다중 라벨 출력은 지원하지 않습니다. nnU-Net 자체 모델 로더는 신뢰하는 학습 결과 폴더에서만 사용합니다. 이 연결 코드는 공식 API 기반의 어댑터이며 실제 MU-Glioma-Post 학습 가중치로 확인하려면 데이터·학습 폴더가 필요합니다.

## 권장 학습 순서

1. smoke와 데이터 단위 테스트로 축, affine, 복셀 간격, 4채널 입력의 의미를 확인합니다.
2. 작은 3D U-Net을 몇 개의 훈련 환자에 과적합시켜 데이터 파이프라인과 손실 변화를 확인합니다. 검증 환자는 별도로 유지합니다.
3. 동일한 환자 분할과 평가 조건으로 U-Net 학습을 진행합니다.
4. nnU-Net의 자동 전처리·패치·네트워크 설정을 수동 U-Net 설정과 비교합니다.
5. Swin UNETR을 추가해 성능뿐 아니라 추론 시간과 자원 사용량을 함께 비교합니다. 아키텍처 이름만으로 성능 우열을 가정하지 않습니다.

자동 다운로드, 사전학습 가중치, 병변별 평가, 5-fold 앙상블, 모델 서빙 성능 보장, 웹에서의 학습 작업 관리 UI는 아직 구현하지 않았습니다. 학습은 CLI에서 실행하고 웹에서는 등록된 모델의 사례별 추론 결과를 비교합니다.
