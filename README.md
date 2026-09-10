# BrainMRISegmentation_3D · neuro/lab

저장소: [SeMinKong/BrainMRISegmentation_3D](https://github.com/SeMinKong/BrainMRISegmentation_3D)

**MU-Glioma-Post MRI 볼륨을 직접 다루고, 3D 종양 분할 결과를 분리·관찰·비교하는 로컬 연구·학습용 워크스페이스입니다.** 치료 후 교종 MRI의 네 시퀀스와 NETC·SNFH·ET·RC 라벨을 기본 데이터 계약으로 사용하고 3D U-Net, nnU-Net v2, Swin UNETR을 연결할 수 있게 구성했습니다.

초기 MVP에는 실행 가능한 웹·API·NIfTI 처리·모델 학습 코드가 들어 있습니다. **MU-Glioma-Post 데이터와 학습된 가중치는 포함하지 않습니다.** 처음 실행하면 수학적으로 생성한 합성 MRI와 마스크가 나타나므로, 데이터 다운로드나 GPU 없이 UI를 먼저 사용할 수 있습니다. 합성 데모의 예측과 점수는 실제 모델 성능이 아닙니다.

**현재 상태 — 2026-09-10:** 실제 다운로드 파일 2,978개의 무결성·입력 형식을 점검했고, 원본 파일명을 그대로 읽는 가져오기와 중복 검사·학습 목록 생성 도구를 추가했습니다. 점검한 데이터에서 571개 검사를 학습 후보로 정리했으며, 실제 데이터로 모델 학습은 아직 실행하지 않았습니다. 자세한 결과는 [검증 기록](docs/verification.md)에 있습니다.

## 빠르게 실행하기

Windows PowerShell에서 이 프로젝트 폴더로 이동한 다음 실행합니다. Python **3.11 이상**과 Node.js **22.12 이상**을 준비하세요. 최초 설치에는 패키지 다운로드를 위한 인터넷 연결이 필요합니다.

```powershell
Set-Location C:\Users\semin\Desktop\01_SeMinKong\git\BrainMRISegmentation_3D
.\scripts\setup.ps1
.\scripts\start.ps1
```

- 웹: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API 문서: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 종료: 서버를 실행한 터미널에서 `Ctrl+C`

`setup.ps1`은 `.venv`에 Python 의존성을 설치하고 프런트엔드를 빌드합니다. `start.ps1`은 API와 빌드된 웹을 하나의 로컬 주소로 제공합니다. 기본 설치에는 PyTorch가 필요하지 않습니다. 첫 서버 시작 시 `.data/`에 합성 사례가 생성됩니다.

스크립트 실행 정책으로 막히면 현재 PowerShell 세션에만 적용한 후 다시 실행할 수 있습니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 웹에서 해볼 수 있는 것

| 화면 | 기능 |
| --- | --- |
| **워크스페이스** | MRI 시퀀스 전환, 축상·관상·시상 단면 탐색, 마스크 오버레이, 3D 뇌·분할 영역 표시 |
| **종양 분리** | `종양만 보기`로 뇌 표면 숨기기, 라벨별 표시·숨김, `영역 펼쳐 보기`로 겹쳐진 내부 영역 관찰 |
| **보기 설정** | 3D 회전·이동·확대·자동 회전·전체 화면, 뇌/마스크 불투명도, 영상 윈도우, 보기 초기화 |
| **결과 비교** | 동일 사례의 마스크 두 개를 나란히 표시, 전체 전경 Dice·HD95·부피 차이 계산 |
| **영역 정보** | 라벨별 복셀 수 기반 부피(mL), 연결 성분 수, 일관된 영역 색상 |
| **추론 모니터링** | 모델 연결 상태, 작업 대기·실행·완료·실패, 진행 단계와 경과 시간, 사례별 추론 기록 |
| **가져오기·저장** | 실제 `.nii`/`.nii.gz` 사례 업로드, 선택한 분할 마스크를 `.nii.gz`로 다운로드 |
| **모델 학습 탭** | 모델별 공부 방향, 논문·공식 자료, 연결 상태와 추론 기록; 학습 실행은 CLI에서 수행 |

처음에는 다음 순서로 조작해 보세요.

1. 기본 **합성 데모**에서 시퀀스를 `T1ce` 또는 `FLAIR`로 선택합니다.
2. 세 단면의 슬라이더나 마우스 휠로 분할 영역을 찾아봅니다.
3. **종양만 보기**를 누르고, **영역 펼쳐 보기**를 켭니다. 하단 눈 아이콘으로 관심 라벨만 남겨 내부 구조를 확인합니다.
4. **결과 비교**에서 `Reference mask`와 `Demo · perturbed reference`를 비교합니다.
5. 왼쪽 **데모 파이프라인 실행**으로 작업 완료와 새 결과 추가 흐름을 확인합니다.
6. 실제 데이터가 준비되면 **NIfTI 가져오기**로 한 사례의 MRI와 선택적인 정답 마스크를 함께 추가합니다.

`영역 펼쳐 보기`는 관찰을 위해 라벨별 3D 표면의 표시 위치를 이동합니다. 원래 마스크·부피·다운로드 파일은 바뀌지 않으며, 토글을 끄면 실제 공간 위치로 돌아갑니다. 연결 성분 수는 각 라벨의 연결된 복셀 집합 수이며 독립적인 종양 개수 판정이 아닙니다.

## MU-Glioma-Post 데이터 가져오기

데이터는 **[MU-Glioma-Post 공식 페이지](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)**에서 받습니다. `Data Access`의 영상·분할 NIfTI 다운로드 항목을 선택하세요. 다운로드와 첫 적용 절차는 [데이터 다운로드 안내](docs/data-access.md)에 정리했습니다.

기본 범위는 **치료 후 교종 MRI와 분할 마스크**입니다. 여기서 다루는 데이터는 **3D NIfTI MRI 볼륨**이며, 스캐너의 k-space raw 신호 처리 도구는 아닙니다.

원본은 `data/MU-Glioma-Post/` 아래에 환자·검사 폴더 구조를 유지해 보관합니다. 이번에 확인한 다운로드 파일은 다음 형식이며, 이름을 바꾸거나 별도 입력용 복사본을 만들 필요가 없습니다.

```text
data/MU-Glioma-Post/PatientID_0003/Timepoint_1/
├── PatientID_0003_Timepoint_1_brain_t1n.nii.gz   # T1
├── PatientID_0003_Timepoint_1_brain_t1c.nii.gz   # 조영증강 T1
├── PatientID_0003_Timepoint_1_brain_t2w.nii.gz   # T2
├── PatientID_0003_Timepoint_1_brain_t2f.nii.gz   # FLAIR
└── PatientID_0003_Timepoint_1_tumorMask.nii.gz   # 정답 마스크
```

**`data/`에 파일을 넣는 것만으로 웹 사례 목록에 자동 등록되지는 않습니다.** 웹의 **NIfTI 가져오기**에서 한 `Timepoint_*` 폴더의 MRI 4개와 정답 마스크가 있으면 함께 선택하고, `MU-Glioma-Post` 프리셋으로 가져옵니다. 여러 환자나 검사 시점의 파일을 한 사례로 섞지 않습니다.

웹은 파일명 마지막 토큰으로 시퀀스를 구분하며 `_`와 `-`를 모두 인식합니다. `tumorMask`, `seg`, `mask`, `segmentation`은 마스크로 인식합니다. MRI 한 개만 가져와도 단면을 볼 수 있지만 실제 모델 추론에는 **네 시퀀스 모두**가 필요합니다. `t1`, `t1ce`, `t2`, `flair` 별칭도 지원합니다. 정답 마스크가 없어도 등록한 모델로 추론할 수 있지만 정답 대비 평가 점수는 계산할 수 없습니다.

| 라벨 프리셋 | 의미 | 용도 |
| --- | --- | --- |
| `mu_glioma_post` | `0` 배경, `1` NETC, `2` SNFH, `3` ET, `4` RC | MU-Glioma-Post 웹 추론 계약 |
| `generic` | `0` 배경, `1..4` Region 1..4 | 의미를 따로 확인할 범용 마스크 열람 |

프리셋은 라벨의 해석을 정하며 파일의 숫자를 자동 변환하지 않습니다. RC는 절제강이므로 화면의 `전체 라벨 부피`에는 종양 자체가 아닌 영역도 포함될 수 있습니다. 자세한 라벨 정의와 학습 계약은 [모델 학습 가이드](docs/model-study.md)에 정리했습니다.

입력 조건은 다음과 같습니다.

- 3차원 `.nii` 또는 `.nii.gz`; 한 사례당 MRI 최소 1개, 마스크 포함 최대 5개.
- 모든 시퀀스와 마스크의 shape·affine이 일치하는 공동 정합된 볼륨.
- NIfTI 공간 단위는 mm. 단위 미지정 입력은 mm로 간주하므로 입력 파일의 실제 단위를 확인.
- 파일당 최대 256 MiB, 사례당 총 768 MiB, 파일당 최대 3,200만 복셀 및 사례 전체 9,600만 복셀.
- 웹은 RAS 방향으로 축을 정리하고 실제 간격을 유지합니다. oblique/shear 격자는 업로드 전에 축 정렬된 격자로 재표본화해야 합니다.
- DICOM·ZIP·4D 시계열, 영상 정합, 두개골 제거는 현재 가져오기 범위에 없습니다.

## 데이터 점검과 학습 목록 준비

프로젝트 루트에서 다음 두 명령을 실행합니다. 첫 단계는 원본 전체를 읽어 검사하고, 두 번째 단계는 검사 결과에 따라 학습 목록을 만듭니다. **원본 MRI는 이동·이름 변경·수정·삭제하지 않습니다.**

```powershell
.\.venv\Scripts\python.exe -E scripts/audit_mu_glioma_post.py --data-root data/MU-Glioma-Post --output data/quality-check --workers 4
.\.venv\Scripts\python.exe -E scripts/prepare_mu_glioma_post.py --audit data/quality-check/audit.json --output-dir data --seed 42 --val-fraction 0.2
```

2026-09-10에 점검한 로컬 자료의 결과는 다음과 같습니다. 배포본이나 파일을 바꾸면 다시 검사해야 하며, 아래 수량은 그때 확인한 자료 기준입니다.

| 항목 | 확인 결과 및 처리 |
| --- | --- |
| 전체 규모 | 환자 ID 203개 · 검사 시점 596개 · NIfTI 2,978개 |
| 파일 무결성·입력 형식 | 전체 gzip CRC·복셀 값·허용 라벨·검사 내 shape/affine 검사 통과 |
| 공간 정보 | 전체 240×240×155, 1 mm, LPS; 웹에서 RAS로 정규화 |
| 마스크 없는 검사 | 2개를 지도 학습에서 제외하고 별도 목록으로 보관 |
| 다른 환자 ID 사이의 동일 MRI | MRI 4종이 같은 11쌍, 양쪽 22개 검사를 학습에서 보류 |
| 시퀀스 의심 | 한 검사에서 T1과 FLAIR가 동일하여 1개 보류 |
| 최종 학습 후보 | 571개 검사 · 환자 ID 200개; 학습 435개 / 검증 136개 |

동일 MRI 11쌍 중 8쌍은 정답 마스크의 복셀 값이 다릅니다. 어느 쪽이 맞는지 임의로 판단하지 않고 두 사례를 모두 검토 대상으로 보관했습니다. 같은 환자의 모든 시점과 동일 파일로 연결된 환자 ID의 남은 검사는 같은 `split_group`에 둡니다. 원래 환자 ID를 합치지는 않습니다. seed 42, 검증 비율 0.2는 **분할 그룹 기준**이며, 최종 환자 ID 분할은 학습 160개 / 검증 40개입니다.

| 생성 파일 | 용도 |
| --- | --- |
| `data/mu-glioma-post-manifest.json` | **기본 학습 입력**; 보류 사례를 제외한 상대 경로 manifest |
| `data/mu-glioma-post-all-labeled.json` | 보류 사례를 포함한 라벨 보유 594개 전체 목록; 검토용 |
| `data/mu-glioma-post-unlabeled.json` | 마스크 없는 2개 검사 목록; 지도 학습 입력이 아님 |
| `data/quality-check/audit.json`, `inventory.csv`, `cases.csv` | 전체 검사 결과, 파일 SHA-256, 검사별 목록 |
| `data/quality-check/curation.json`, `excluded-cases.csv` | 보류 사유, 연결된 환자 ID, 최종 분할 |

위 결과와 데이터는 `data/` 아래에서 로컬로 생성되며 Git에는 포함되지 않습니다. 파일을 추가·교체하면 두 명령을 다시 실행하세요. 이번 작업에서 작성한 로컬 HTML 보고서와 노트북은 별도 스냅샷이며, 위 두 명령으로 자동 생성되거나 갱신되지는 않습니다.

압축 파일 SHA-256 검사는 재압축되거나 일부 수정된 유사 영상을 모두 탐지하지 못합니다. 좌표·라벨 형식이 맞는다고 실제 해부학적 정렬이나 주석 정확도가 검증된 것은 아닙니다. 검사 방법과 범위는 [데이터 안내](docs/data-access.md)와 [검증 기록](docs/verification.md)을 참고하세요.

## 모델 공부와 실제 학습

실제 모델 사용 시 선택 의존성을 추가합니다. CUDA를 사용할 경우 먼저 환경에 맞는 PyTorch 설치를 준비하세요. 상세 명령과 nnU-Net 학습 과정은 [docs/model-study.md](docs/model-study.md)를 참고하세요.

```powershell
.\scripts\setup.ps1 -WithML
.\.venv\Scripts\python.exe -E -m ml.smoke --output runs/smoke
```

CPU 전용 PyTorch를 설치하려면 `setup.ps1 -WithML -CpuOnly`를 사용합니다.

`ml.smoke`는 작은 합성 볼륨 두 개로 **실제 학습 1 step → validation → 체크포인트 저장 → 등록 모델 추론 → 출력 shape·affine·정수 라벨 검증**을 수행합니다. 생성된 `runs/smoke/best.pt`는 연결을 확인하기 위한 가중치이며 실데이터 성능 모델이 아닙니다.

MU-Glioma-Post는 위의 **데이터 점검과 학습 목록 준비** 두 단계를 먼저 실행하고, 생성된 `data/mu-glioma-post-manifest.json`으로 학습합니다.

```powershell
.\.venv\Scripts\python.exe -E -m ml.train --manifest data/mu-glioma-post-manifest.json --model unet3d --output runs/unet3d --epochs 100 --patch-size 64 64 64
```

다른 폴더 구조를 연결할 때는 [수동 manifest 예제](configs/manifest.example.json)를 참고할 수 있습니다. `ml.manifest` 자동 탐색은 명시적인 `--patient-regex`가 필요하고 파일 구성·환자 ID 분할을 검사하지만, 파일 내용 중복에 따른 보류 정책은 적용하지 않습니다. 점검한 MU-Glioma-Post의 학습 목록은 `prepare_mu_glioma_post.py`로 재생성하세요. 환자 ID 자체가 잘못 기록된 경우까지 자동으로 보정하지는 않습니다.

| 모델 | 구현 상태 |
| --- | --- |
| **3D U-Net** | MONAI residual 3D U-Net, 네 채널 입력, 패치 학습, Dice+CE, 전체 볼륨 sliding-window 검증·추론 |
| **Swin UNETR** | 같은 manifest·학습 루프에서 `--model swinunetr`로 선택 가능한 모델 |
| **nnU-Net v2** | 데이터 export, 고정 split 내보내기, 외부에서 학습한 모델 폴더를 연결하는 추론 어댑터 |
| **YOLO** | 기존 별도 2D 프로젝트를 비교 출발점으로 유지; 이 MVP에는 3D YOLO 구현을 추가하지 않음 |

훈련 결과는 `best.pt`, `last.pt`, `metadata.json`, `metrics.json`, `resolved-manifest.json`으로 저장됩니다. 웹에서 사용하려면 서버를 시작하는 PowerShell에서 경로를 등록합니다.

```powershell
$env:MRI_UNET_CHECKPOINT = (Resolve-Path runs/unet3d/best.pt).Path
$env:MRI_DEVICE = "auto"
.\scripts\start.ps1
```

| 환경 변수 | 기본값 / 역할 |
| --- | --- |
| `MRI_DATA_DIR` | 프로젝트 `.data/`; 업로드와 마스크 저장 위치 |
| `MRI_DEVICE` | `auto`; `cpu`, `cuda`도 선택 가능 |
| `MRI_UNET_CHECKPOINT` | 이 프로젝트의 3D U-Net 학습 체크포인트 |
| `MRI_SWIN_CHECKPOINT` | 이 프로젝트의 Swin UNETR 학습 체크포인트 |
| `MRI_NNUNET_MODEL_DIR` | `plans.json`, `dataset.json`, `fold_*/checkpoint_final.pth`를 포함한 nnU-Net 모델 폴더 |
| `MRI_NNUNET_FOLDS` | `0`; 예: `0,1,2,3,4` |

설정 형식은 [.env.example](.env.example)에도 있습니다. 프로젝트 루트에 `.env`로 복사해 값을 넣으면 두 시작 스크립트가 읽습니다. 체크포인트의 모델·네 채널 순서·전처리·라벨 의미가 일치해야 실제 추론이 실행됩니다. 설정하지 않은 모델은 UI에서 미연결로 표시합니다. 데모 파이프라인은 합성 사례에만 사용할 수 있습니다.

## 프로젝트 구조

### GitHub에 포함되는 소스 구조

현재 Git에서 추적하는 **44개 파일 전체**를 표시했습니다. `git ls-files` 목록과 대조했으며, 폴더와 파일 이름은 실제 경로 기준입니다.

```text
BrainMRISegmentation_3D/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py  # FastAPI 라우트·업로드·추론 작업
│   │   ├── store.py  # 사례·마스크 저장과 캐시
│   │   └── volumes.py  # NIfTI 검증·단면·3D mesh·지표
│   ├── tests/
│   │   └── test_api.py  # API·공간 좌표·가져오기 테스트
│   └── __init__.py
├── configs/
│   └── manifest.example.json  # 다른 입력 구조를 위한 수동 manifest 예제
├── docs/
│   ├── architecture.md  # 데이터 흐름과 계산 정의
│   ├── data-access.md  # 다운로드·원본 입력·점검·정리
│   ├── model-study.md  # 모델 공부·학습·nnU-Net 연결
│   └── verification.md  # 검증 범위와 실행 결과
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MeshViewer.tsx  # 3D 표면·종양 분리·회전
│   │   │   └── SliceViewer.tsx  # 축상·관상·시상 단면
│   │   ├── api.ts  # API 타입과 요청
│   │   ├── App.tsx  # 사례 탐색·가져오기·결과 비교 UI
│   │   ├── inferenceJob.ts  # 추론 상태 조회·재시도·요청 취소
│   │   ├── main.tsx  # React 시작점
│   │   └── styles.css  # 화면 스타일
│   ├── tests/
│   │   └── inferenceJob.test.mjs  # 추론 조회 회귀 테스트
│   ├── index.html  # 웹 HTML 진입점
│   ├── package-lock.json  # 고정된 npm 의존성
│   ├── package.json  # 프런트엔드 의존성·실행 명령
│   ├── tsconfig.json  # TypeScript 설정
│   └── vite.config.ts  # 개발 프록시·빌드 설정
├── ml/
│   ├── tests/
│   │   └── test_data_contract.py  # 데이터·모델 계약 테스트
│   ├── __init__.py
│   ├── adapters.py  # 모델 등록·체크포인트 검증·추론
│   ├── data.py  # 전처리·패치·원본 격자 복원
│   ├── export_nnunet.py  # nnU-Net 데이터셋 변환
│   ├── manifest.py  # 파일 탐색·명시적 환자 ID 분할
│   ├── models.py  # 3D U-Net·Swin UNETR 생성
│   ├── requirements.txt  # ML 선택 의존성 안내
│   ├── schema.py  # 채널·라벨·환자 분할 계약
│   ├── smoke.py  # 합성 데이터 학습·추론 연결 검증
│   └── train.py  # 학습·검증·체크포인트 저장
├── scripts/
│   ├── audit_mu_glioma_post.py  # 원본 전체 무결성·격자·라벨·중복 검사
│   ├── prepare_mu_glioma_post.py  # 의심 사례 보류·학습 목록 생성
│   ├── setup.ps1  # 의존성 설치·웹 빌드
│   ├── start-dev.ps1  # API·Vite 개발 서버
│   └── start.ps1  # 로컬 API·빌드된 웹 실행
├── .env.example  # 로컬 설정 예제
├── .gitignore  # 데이터·환경·생성물 제외 규칙
├── pyproject.toml  # Python 패키지·의존성·테스트 설정
└── README.md  # 프로젝트 시작 안내
```

### 로컬 데이터와 실행 결과

다음 경로는 현재 로컬 작업 폴더에 있고 `.gitignore`로 제외됩니다. `data/quality-check/`와 manifest는 현재 목록을 표시하고, MRI는 **실제로 존재하는 환자 1명의 검사 1개만 예시**로 표시했습니다. 다른 환자·시점과 `.data/`·`runs/`의 내부 생성 파일은 생략했습니다.

```text
BrainMRISegmentation_3D/
├── .data/  # 앱이 사용하는 사례 저장소
│   └── demo-brain-001/  # 기본 합성 사례; 내부 파일 생략
├── data/  # 사용자가 받은 원본·학습 목록·점검 결과
│   ├── MU-Glioma-Post/  # 환자 ID 203개; 아래는 실제 한 사례만 표시
│   │   └── PatientID_0003/
│   │       └── Timepoint_1/  # 다른 환자·시점은 생략
│   │           ├── PatientID_0003_Timepoint_1_brain_t1c.nii.gz
│   │           ├── PatientID_0003_Timepoint_1_brain_t1n.nii.gz
│   │           ├── PatientID_0003_Timepoint_1_brain_t2f.nii.gz
│   │           ├── PatientID_0003_Timepoint_1_brain_t2w.nii.gz
│   │           └── PatientID_0003_Timepoint_1_tumorMask.nii.gz
│   ├── quality-check/
│   │   ├── audit.ipynb  # 실행 결과가 저장된 점검 노트북
│   │   ├── audit.json  # 전체 원본 검사 결과
│   │   ├── build_notebook.py  # 로컬 노트북 재생성 도구
│   │   ├── cases.csv  # 검사별 목록
│   │   ├── check_web_import.py  # 실제 파일의 API 입출력 검증
│   │   ├── curation.json  # 보류 기준·연결 그룹·최종 분할
│   │   ├── duplicate-case-pairs.json  # 중복 사례와 마스크 차이 확인
│   │   ├── excluded-cases.csv  # 보류한 라벨 보유 사례
│   │   ├── inventory.csv  # 파일 목록·SHA-256
│   │   ├── preview.png  # 실제 사례의 네 시퀀스 미리보기
│   │   ├── report.html  # 로컬 점검 보고서
│   │   └── web-import-check.json  # 실제 사례 입출력 검증 결과
│   ├── mu-glioma-post-all-labeled.json  # 보류 사례를 포함한 전체 라벨 목록
│   ├── mu-glioma-post-manifest.json  # 정리된 기본 학습 입력
│   ├── mu-glioma-post-unlabeled.json  # 마스크 없는 사례 목록
│   └── README.md  # 이 로컬 자료의 사용 안내
└── runs/  # 테스트·검증 출력; 학습 실행 시 체크포인트도 저장
```

`data/`는 원본 데이터와 학습 준비 결과를 보관하고, `.data/`는 웹에서 가져온 사례와 기본 합성 사례를 보관합니다. 웹 가져오기를 해야 원본이 앱 사례로 등록됩니다. `runs/`에는 현재 테스트·검증 출력이 있으며, 실제 데이터의 학습 가중치는 아직 생성하지 않았습니다.

설치·빌드·테스트 과정에서 생기는 다음 경로도 현재 로컬에 있습니다.

| 경로 | 용도 |
| --- | --- |
| `.venv/` | 프로젝트 Python 가상환경 |
| `frontend/node_modules/` | 설치한 프런트엔드 패키지 |
| `frontend/dist/` | 빌드된 웹 파일 |
| `frontend/tsconfig.tsbuildinfo` | TypeScript 증분 빌드 정보 |
| `.pytest_cache/`, 각 Python 폴더의 `__pycache__/` | 테스트·Python 실행 캐시 |
| `brain_mri_segmentation_3d.egg-info/` | 로컬 Python 패키지 설치 정보 |
| `.git/` | Git이 관리하는 로컬 저장소 메타데이터 |

새로 clone한 폴더에는 GitHub 소스 구조가 먼저 생기며, 데이터 다운로드와 설치·실행 후 로컬 경로들이 추가됩니다.

## 개발과 검증

```powershell
.\scripts\start-dev.ps1
# API: 127.0.0.1:8000, 개발 웹: 127.0.0.1:5173

.\.venv\Scripts\python.exe -E -m pytest backend/tests ml/tests
npm --prefix frontend test
npm --prefix frontend run build
```

직접 서버를 시작하려면 프로젝트 루트에서 아래 명령을 사용합니다. `-E`는 다른 프로젝트의 `PYTHONPATH`가 이 환경에 섞이는 것을 방지합니다.

```powershell
.\.venv\Scripts\python.exe -E -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

단위 테스트는 합성 사례를 사용해 입력 거절, 물리 좌표, 마스크 저장, 실제 계산된 지표 및 추론 작업 흐름을 검사합니다. 선택 ML 설치 후 `ml.smoke`로 실제 optimizer와 체크포인트 추론까지 확인할 수 있습니다. MU-Glioma-Post에서 학습한 가중치의 정확도 검증은 별도의 데이터와 실험이 필요합니다.

2026-09-10 기준 백엔드·데이터 계약 테스트 **43개**가 통과했습니다. 실제 다운로드 사례의 MRI 4개와 `tumorMask`를 가져와 단면·3D mesh 응답을 확인했고, 다운로드한 마스크가 RAS로 정규화한 원본의 복셀 값·affine과 일치하는지 검증했습니다. 상세 실행 범위는 [검증 기록](docs/verification.md)에 있습니다.

## 현재 범위와 다음 단계

현재 3D 화면은 **마스크의 표면 mesh**를 표시합니다. 반투명 뇌 표면은 영상 intensity로 만든 위치 참고용 외피이며 정밀 뇌 조직 분할이 아닙니다. 화면 성능을 위해 표면 계산을 축소하므로 작은 영역·경계는 단면과 원본 해상도 마스크로 함께 확인하세요. 부피와 지표는 화면 mesh가 아닌 마스크 복셀에서 계산합니다.

비교 점수는 선택한 두 마스크의 전체 전경과 라벨별 계산입니다. 정답 없는 두 예측의 일치는 정확도를 뜻하지 않으며, 병변별 개별 평가도 수행하지 않습니다. 웹의 마스크 export는 저장된 RAS 격자를 유지합니다. CLI 추론은 입력 T1의 원래 격자로 복원합니다. 좌표와 지표의 정확한 정의는 [아키텍처 문서](docs/architecture.md)를 참고하세요.

사례와 분할 결과는 디스크에 남고 추론 큐·작업 기록은 서버 재시작 시 초기화됩니다. 개인 로컬 사용을 위한 MVP이며 로그인·공유·배포, DICOM/PACS, 웹 학습 실행·GPU 모니터링, 연결된 단면 crosshair, 수동 마스크 편집, 병변별 개별 선택은 후속 확장 대상입니다. 실제 데이터를 통한 학습·평가 후 환자별 실험 비교와 nnU-Net 기준 성능 확보부터 진행하는 구조입니다.
