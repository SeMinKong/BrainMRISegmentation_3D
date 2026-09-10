# MU-Glioma-Post 다운로드와 첫 적용

데이터는 **[MU-Glioma-Post 공식 페이지](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)**에서 직접 다운로드합니다.

## 다운로드

공식 페이지의 **Data Access → Images (skull-stripped) and Segmentations → Download (11gb)**를 선택합니다. 2026-09-08 확인 기준 NIfTI 영상·마스크를 제공하며, 다운로드에는 IBM Aspera Connect가 필요하다고 안내합니다. 라이선스는 CC BY 4.0이며 공개 결과에는 배포 페이지의 데이터 인용을 포함합니다.

## 프로젝트 입력

치료 후 교종 MRI의 네 시퀀스 **T1·조영증강 T1·T2·FLAIR**를 사용합니다. 기본 프리셋 `mu_glioma_post`는 `0 배경 / 1 NETC / 2 SNFH / 3 ET / 4 RC`를 의미합니다. RC는 절제강으로 전체 라벨 부피에 포함됩니다. 라벨 정의와 데이터 설명은 위 공식 페이지를 기준으로 합니다.

## 다운로드 후 적용 순서

1. 압축을 풀고 원래 환자·검사 폴더 구조를 유지합니다. 저장 위치는 프로젝트의 `data/` 또는 별도 데이터 폴더를 사용합니다.
2. 한 환자의 한 검사에서 네 MRI와 정답 마스크를 확인합니다. 프로젝트는 한 개의 정수 라벨 마스크를 입력받습니다. 영역별 이진 마스크가 따로 있는 경우에는 실제 파일을 확인한 뒤 통합해야 합니다.
3. 웹 **NIfTI 가져오기**에서 그 검사의 파일 최대 5개를 함께 선택하고 `MU-Glioma-Post` 프리셋으로 엽니다. 압축 ZIP 자체는 업로드하지 않습니다.
4. 정답 마스크의 단면·3D 표시·라벨별 부피를 확인합니다. 실제 모델 예측에는 별도로 학습한 체크포인트가 필요합니다.
5. 학습 전에 파일 경로와 환자 ID를 manifest에 기록합니다. 같은 환자의 여러 검사는 반드시 같은 train/val/test에 묶습니다.

웹은 파일명 마지막 토큰의 `t1n`, `t1c`, `t2w`, `t2f`, `seg`와 시퀀스 별칭을 인식하며 `_`와 `-` 구분자를 지원합니다. 실제 배포본의 파일명이 이 조건에 맞는지는 다운로드 후 확인합니다. 원본 파일은 유지한 채 프로젝트 입력용 복사본이나 명시적 manifest로 연결합니다.

## 학습 manifest

기본 절차는 [수동 manifest 예제](../configs/manifest.example.json)에 실제 파일 경로·환자 ID·분할을 입력해 `data/mu-glioma-post-manifest.json`으로 저장하는 것입니다. 예제의 파일명은 프로젝트 형식을 설명하기 위한 것으로 배포본의 실제 파일명을 뜻하지 않습니다.

자동 탐색 도구는 `_seg.nii`/`-seg.nii` 및 압축형 `.nii.gz` 마스크와 같은 접두사를 가진 MRI를 찾습니다. 환자 ID를 추측하지 않으므로 `--patient-regex`가 필요합니다. 예를 들어 확인한 사례명이 `patient-001_timepoint-01` 형식일 때만 다음 명령을 사용합니다. 데이터 위치와 정규식은 실제 구조에 맞게 지정해야 합니다.

```powershell
.\.venv\Scripts\python.exe -E -m ml.manifest --data-root D:/datasets/MU-Glioma-Post --patient-regex '^(.+)_timepoint-\d+$' --output data/mu-glioma-post-manifest.json
```

다운로드가 끝나면 압축을 푼 폴더 경로를 기준으로 파일·라벨·격자를 검사한 뒤 첫 사례 업로드와 환자 분할을 진행합니다. 자세한 학습 명령은 [모델 학습 가이드](model-study.md)를 참고하세요.
