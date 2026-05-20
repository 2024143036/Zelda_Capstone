import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import os


def main():
    print("⏳ MobileBERT 감성 분류 모델 학습 프로세스 시작...")

    # 1. 전처리 완료된 데이터셋 불러오기
    csv_file = "zelda_train_final.csv"
    if not os.path.exists(csv_file):
        print(f"❌ '{csv_file}' 파일이 없습니다! data_processing.py를 먼저 실행해주세요.")
        return

    df = pd.read_csv(csv_file)

    # BERT 내부 규칙: 라벨을 0, 1, 2로 매핑 (+1 처리)
    df['label'] = df['Label'] + 1
    df = df.rename(columns={'Review': 'text'})

    # 학습용 데이터(80%)와 검증용 데이터(20%)로 분할
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"📊 데이터 분할 완료 - 학습용: {len(train_df)}건 / 검증용: {len(val_df)}건")

    # 2. 허깅페이스 데이터셋 형태로 변환
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))

    # 3. MobileBERT 토크나이저 및 기본 모델 로드
    print("🌐 구글 오리지널 MobileBERT 구조 및 토크나이저 다운로드 중...")
    model_name = "google/mobilebert-uncased"
    tokenizer = MobileBertTokenizer.from_pretrained(model_name)
    model = MobileBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 4. 텍스트 데이터를 토큰으로 변환하는 함수
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    print("🔤 리뷰 텍스트 토큰화(Tokenization) 진행 중...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # 5. 하이퍼파라미터 및 학습 옵션 설정 (AI 뇌를 확실하게 개조하는 세팅)
    training_args = TrainingArguments(
        output_dir="./results",

        # 💡 [핵심 수정 1] 공부 횟수를 3회에서 20회로 대폭 늘려 패턴을 완전히 외우게 합니다.
        num_train_epochs=20,

        # 💡 [핵심 수정 2] 경량화 데이터에 맞춰 학습 속도(Learning Rate)를 정밀하게 고정합니다.
        learning_rate=5e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,  # 💡 [핵심 수정 3] 빠른 적응을 위해 웜업 단계를 줄입니다.
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,  # 더 자주 로그를 보도록 수정

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_num_workers=0
    )

    # 6. 트레이너(Trainer) 가동
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\n🔥 MobileBERT 엔진 학습 가동 (Fine-tuning)... 잠시만 기다려주세요.")
    trainer.train()
    print("✅ 모델 학습 완료!")

    # 7. 완성된 나만의 젤다 전용 감성 분류 모델 영구 저장
    save_directory = "./my_zelda_mobilebert"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"💾 최적화된 AI 모델이 '{save_directory}' 폴더에 안전하게 저장되었습니다!")


if __name__ == "__main__":
    main()