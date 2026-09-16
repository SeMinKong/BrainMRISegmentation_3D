import type { Case, Label } from "./api";

/** Plain-language names for the MU-Glioma-Post labels, for explaining a scan to a patient. */
export const LABEL_GUIDE: Record<number, { friendly: string; description: string }> = {
  1: { friendly: "비조영 종양", description: "조영제가 들어가지 않는 종양 부분 (NETC)" },
  2: { friendly: "주변 신호 변화", description: "종양 주변의 부종·신호 변화 영역 (SNFH)" },
  3: { friendly: "조영증강 종양", description: "조영제로 밝게 보이는 활성 종양 (ET)" },
  4: { friendly: "절제강", description: "수술로 종양을 제거한 뒤 남은 빈 공간 (RC)" },
};

export const friendlyLabel = (label: Label, data: Case) =>
  data.label_preset === "mu_glioma_post" ? LABEL_GUIDE[label.id]?.friendly ?? label.name : label.name;

export const labelDescription = (label: Label, data: Case) =>
  data.label_preset === "mu_glioma_post" ? LABEL_GUIDE[label.id]?.description ?? "" : `라벨 ${label.id}`;

export const sequenceName: Record<string, string> = {
  t1n: "T1",
  t1c: "T1 조영",
  t2w: "T2",
  t2f: "FLAIR",
};

export const sequenceHint: Record<string, string> = {
  t1n: "기본 해부 구조",
  t1c: "활성 종양이 밝게 보임",
  t2w: "물·부종이 밝게 보임",
  t2f: "부종과 신호 변화 강조",
};

/** Plain-language glossary for people who are not clinicians. Keyed by the term id used in <Help term=...>. */
export const GLOSSARY: Record<string, { title: string; text: string }> = {
  unseen: {
    title: "학습에 쓰지 않은 검사 (검증 검사)",
    text: "모델을 만들 때 한 번도 보여 주지 않은 환자의 검사입니다. 여기서 잘 맞아야 처음 보는 환자에게도 통한다고 볼 수 있습니다. 학습에 쓴 검사는 답을 외웠을 수 있어 참고용입니다.",
  },
  reference: {
    title: "판독 마스크 (정답)",
    text: "전문가가 MRI 위에 종양 영역을 색으로 표시해 둔 것입니다. 모델이 맞혀야 하는 답안지 역할을 합니다.",
  },
  prediction: {
    title: "모델 예측",
    text: "내가 학습시킨 모델이 MRI 네 장만 보고 스스로 그린 종양 영역입니다. 판독 마스크와 비교해 얼마나 비슷한지 봅니다.",
  },
  dice: {
    title: "일치도 (Dice)",
    text: "두 영역이 얼마나 겹치는지를 0~1로 나타냅니다. 1이면 완전히 같고 0이면 전혀 겹치지 않습니다. 대략 0.9 이상 매우 일치, 0.7 이상 대체로 일치, 0.5 아래는 많이 다릅니다. 작은 영역은 몇 복셀만 어긋나도 값이 크게 떨어집니다.",
  },
  volume: {
    title: "부피 (mL)",
    text: "색으로 표시된 영역의 크기입니다. 1 mL는 1 cm³, 각설탕 하나 정도입니다. 영상의 한 점(복셀) 크기에 점 개수를 곱해 계산합니다.",
  },
  sequence: {
    title: "MRI 시퀀스",
    text: "같은 뇌를 다른 방식으로 촬영한 영상들입니다. T1은 기본 구조, T1 조영은 조영제가 들어간 활성 종양이 밝게, T2와 FLAIR는 물·부종이 밝게 보입니다. 모델은 네 장을 동시에 봅니다.",
  },
  planes: {
    title: "축상면 · 관상면 · 시상면",
    text: "3D 영상을 세 방향으로 잘라 본 단면입니다. 축상면은 위에서 내려다본 것, 관상면은 앞에서, 시상면은 옆에서 본 것입니다.",
  },
  netc: { title: "비조영 종양 (NETC)", text: "조영제가 들어가지 않는 종양 부분입니다. 주변과 신호가 비슷해 모델이 가장 어려워하는 영역입니다." },
  snfh: { title: "주변 신호 변화 (SNFH)", text: "종양 주변의 부종이나 신호가 달라진 조직입니다. FLAIR에서 밝게 보이며 대개 범위가 넓습니다." },
  et: { title: "조영증강 종양 (ET)", text: "조영제로 밝게 보이는 활성 종양입니다. T1 조영 영상에서 가장 잘 구별됩니다." },
  rc: { title: "절제강 (RC)", text: "수술로 종양을 제거한 뒤 남은 빈 공간입니다. 종양이 아니지만 치료 후 영상에서 함께 표시합니다." },
  model: {
    title: "내 모델 (3D U-Net)",
    text: "MRI 네 장을 입력받아 각 점이 어느 영역인지 맞히도록 학습한 신경망입니다. 학습에 쓴 435개 검사로 배우고, 쓰지 않은 136개 검사로 성능을 측정했습니다.",
  },
};

export const labelTerm: Record<number, string> = { 1: "netc", 2: "snfh", 3: "et", 4: "rc" };
