# BrainMRISegmentation_3D · neuro/lab

저장소: [SeMinKong/BrainMRISegmentation_3D](https://github.com/SeMinKong/BrainMRISegmentation_3D)

**MU-Glioma-Post MRI 볼륨을 직접 다루고, 3D 종양 분할 결과를 분리·관찰·비교하는 로컬 연구·학습용 워크스페이스입니다.** 치료 후 교종 MRI의 네 시퀀스와 NETC·SNFH·ET·RC 라벨을 기본 데이터 계약으로 사용하고 3D U-Net, nnU-Net v2, Swin UNETR을 연결할 수 있게 구성했습니다.

초기 MVP에는 실행 가능한 웹·API·NIfTI 처리·모델 학습 코드가 들어 있습니다. **MU-Glioma-Post 데이터와 학습된 가중치는 포함하지 않습니다.** 처음 실행하면 수학적으로 생성한 합성 MRI와 마스크가 나타나므로, 데이터 다운로드나 GPU 없이 UI를 먼저 사용할 수 있습니다. 합성 데모의 예측과 점수는 실제 모델 성능이 아닙니다.

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

웹은 파일명 마지막 토큰으로 시퀀스를 구분하며 `_`와 `-`를 모두 인식합니다. 다음은 프로젝트 입력 형식 예제이며 다운로드 원본의 파일명을 보장하지 않습니다. 실제 배포본의 환자·검사 폴더와 파일명을 확인한 뒤 연결합니다.

```text
patient-001_timepoint-01_t1n.nii.gz   # T1
patient-001_timepoint-01_t1c.nii.gz   # 조영증강 T1
patient-001_timepoint-01_t2w.nii.gz   # T2
patient-001_timepoint-01_t2f.nii.gz   # FLAIR
patient-001_timepoint-01_seg.nii.gz   # 정답 마스크, 웹 열람에서는 선택 사항
```

웹에서는 MRI 한 개만 가져와도 단면을 볼 수 있습니다. 실제 모델 추론에는 **네 시퀀스 모두**가 필요합니다. `t1`, `t1ce`, `t2`, `flair` 별칭도 지원합니다. 정답 마스크가 없어도 등록한 모델로 추론할 수 있지만 정답 대비 평가 점수는 계산할 수 없습니다.

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

## 모델 공부와 실제 학습

실제 모델 사용 시 선택 의존성을 추가합니다. CUDA를 사용할 경우 먼저 환경에 맞는 PyTorch 설치를 준비하세요. 상세 명령과 nnU-Net 학습 과정은 [docs/model-study.md](docs/model-study.md)를 참고하세요.

```powershell
.\scripts\setup.ps1 -WithML
.\.venv\Scripts\python.exe -E -m ml.smoke --output runs/smoke
```

CPU 전용 PyTorch를 설치하려면 `setup.ps1 -WithML -CpuOnly`를 사용합니다.

`ml.smoke`는 작은 합성 볼륨 두 개로 **실제 학습 1 step → validation → 체크포인트 저장 → 등록 모델 추론 → 출력 shape·affine·정수 라벨 검증**을 수행합니다. 생성된 `runs/smoke/best.pt`는 연결을 확인하기 위한 가중치이며 실데이터 성능 모델이 아닙니다.

MU-Glioma-Post 다운로드 후 [manifest 예제](configs/manifest.example.json)의 경로와 환자 ID를 실제 파일에 맞춰 `data/mu-glioma-post-manifest.json`에 작성합니다. 같은 환자의 여러 검사를 같은 분할로 묶은 뒤 학습합니다.

```powershell
.\.venv\Scripts\python.exe -E -m ml.train --manifest data/mu-glioma-post-manifest.json --model unet3d --output runs/unet3d --epochs 100 --patch-size 64 64 64
```

`ml.manifest` 자동 탐색을 사용하려면 실제 파일명과 일치하는 `--patient-regex`를 명시해야 합니다. 원본 배포 구조는 다운로드 후 확인하며, 자동 탐색 조건과 다르면 수동 manifest를 사용합니다. 환자 ID가 잘못 입력되면 자동 누출 검사도 이를 알아낼 수 없으므로 학습 전에 분할을 확인하세요.

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

```text
BrainMRISegmentation_3D/
├── README.md                       # 실행·UI 사용·데이터·모델 연결
├── pyproject.toml                  # Python 패키지와 core/dev/ml 의존성
├── .env.example                    # 모델 경로·데이터 위치 설정 예제
├── scripts/
│   ├── setup.ps1                   # 가상환경·의존성 설치·웹 빌드
│   ├── start.ps1                   # 로컬 API + 빌드된 웹 실행
│   └── start-dev.ps1               # API + Vite 개발 서버
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 라우트·업로드·추론 작업 큐
│   │   ├── store.py                # 사례/마스크 파일 저장·메타데이터·캐시
│   │   └── volumes.py              # NIfTI 검증·단면·3D mesh·지표·합성 데모
│   └── tests/test_api.py           # 업로드·방향·지표·작업·오류 경로 검증
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # 사례 탐색·결과 비교·모델 상태·업로드 UI
│   │   ├── api.ts                  # API 타입과 요청 함수
│   │   ├── styles.css              # 반응형 다크 워크스페이스 스타일
│   │   └── components/
│   │       ├── MeshViewer.tsx      # Three.js 3D 표면·종양 분리·회전
│   │       └── SliceViewer.tsx     # 축상·관상·시상 단면 뷰어
│   ├── package.json
│   ├── package-lock.json           # npm 의존성 고정
│   └── vite.config.ts              # 개발 API 프록시와 프로덕션 빌드
├── ml/
│   ├── schema.py                   # 채널·라벨·환자 분할 계약
│   ├── manifest.py                 # NIfTI 사례 탐색·명시적 환자 규칙으로 train/val 생성
│   ├── data.py                     # RAS·spacing·정규화·패치·출력 격자 복원
│   ├── models.py                   # 3D U-Net / Swin UNETR 생성
│   ├── train.py                    # 학습·검증·체크포인트·기록 저장
│   ├── adapters.py                 # 모델 레지스트리·체크포인트 검증·추론
│   ├── export_nnunet.py            # nnU-Net 데이터셋과 고정 split 변환
│   ├── smoke.py                    # 작은 합성 데이터로 학습/추론 연결 확인
│   └── tests/test_data_contract.py # 축·affine·라벨·분할·모델 계약 검증
├── configs/manifest.example.json
├── docs/
│   ├── architecture.md             # 데이터 흐름·API·계산 정의·확장 경계
│   └── model-study.md              # 모델 원리·데이터 준비·학습·nnU-Net 연결
├── .data/                          # 실행 시 생성; 사례 NIfTI와 case.json
├── data/                           # 사용자 데이터/manifest; Git에서 제외
└── runs/                           # 학습 출력; Git에서 제외
```

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

## 현재 범위와 다음 단계

현재 3D 화면은 **마스크의 표면 mesh**를 표시합니다. 반투명 뇌 표면은 영상 intensity로 만든 위치 참고용 외피이며 정밀 뇌 조직 분할이 아닙니다. 화면 성능을 위해 표면 계산을 축소하므로 작은 영역·경계는 단면과 원본 해상도 마스크로 함께 확인하세요. 부피와 지표는 화면 mesh가 아닌 마스크 복셀에서 계산합니다.

비교 점수는 선택한 두 마스크의 전체 전경과 라벨별 계산입니다. 정답 없는 두 예측의 일치는 정확도를 뜻하지 않으며, 병변별 개별 평가도 수행하지 않습니다. 웹의 마스크 export는 저장된 RAS 격자를 유지합니다. CLI 추론은 입력 T1의 원래 격자로 복원합니다. 좌표와 지표의 정확한 정의는 [아키텍처 문서](docs/architecture.md)를 참고하세요.

사례와 분할 결과는 디스크에 남고 추론 큐·작업 기록은 서버 재시작 시 초기화됩니다. 개인 로컬 사용을 위한 MVP이며 로그인·공유·배포, DICOM/PACS, 웹 학습 실행·GPU 모니터링, 연결된 단면 crosshair, 수동 마스크 편집, 병변별 개별 선택은 후속 확장 대상입니다. 실제 데이터를 통한 학습·평가 후 환자별 실험 비교와 nnU-Net 기준 성능 확보부터 진행하는 구조입니다.
