import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def process_massive_zelda_data():
    input_file = "zelda_metacritic_raw.csv"

    if not os.path.exists(input_file):
        print(f"❌ 원본 파일 '{input_file}'이 폴더에 없습니다!")
        print("💡 조치 방법: 먼저 'zelda_crawler.py'를 실행하여 원본 데이터를 생성해 주세요.")
        return

    print("📊 1. 24,500건 스케일 대용량 젤다 로우 데이터 로드 중...")
    df = pd.read_csv(input_file)
    print(f"➡️ 로드 완료! 수집된 총 데이터 수: {len(df):,}건")

    # 💡 텍스트 데이터 정밀 정제 (중복 리뷰 제거 및 무의미한 단문 필터링)
    df = df.drop_duplicates(subset=['Review'])
    df = df[df['Review'].astype(str).str.len() > 15]
    print(f"✨ 의미 있는 데이터 최종 정제 후: {len(df):,}건 남음")

    # 💡 평점(Score) 기반 감성 라벨링 (8점 이상: 긍정, 4점 이상: 중립, 그 이하: 부정)
    def get_sentiment_label(score):
        if score >= 8:
            return 1  # 긍정
        elif score >= 4:
            return 0  # 중립
        else:
            return -1  # 부정

    df['Label'] = df['Score'].apply(get_sentiment_label)

    # 🎯 [핵심] 야생의 숨결(BOTW)과 왕국의 눈물(TOTK) 데이터셋 분리
    botw_df = df[df['Title'].astype(str).str.contains('breath|botw', case=False, na=False)].copy()
    totk_df = df[df['Title'].astype(str).str.contains('tears|totk', case=False, na=False)].copy()

    print("\n============================================================")
    print(f"🎮 야생의 숨결 (BOTW) 유효 데이터: {len(botw_df):,}건")
    botw_counts = botw_df['Label'].value_counts()
    print(
        f"   - 긍정(1): {botw_counts.get(1, 0):,}건 | 중립(0): {botw_counts.get(0, 0):,}건 | 부정(-1): {botw_counts.get(-1, 0):,}건")

    print(f"🎮 왕국의 눈물 (TOTK) 유효 데이터: {len(totk_df):,}건")
    totk_counts = totk_df['Label'].value_counts()
    print(
        f"   - 긍정(1): {totk_counts.get(1, 0):,}건 | 중립(0): {totk_counts.get(0, 0):,}건 | 부정(-1): {totk_counts.get(-1, 0):,}건")
    print("============================================================\n")

    # 차트 작성을 위한 그룹명 정규화
    botw_df['Game_Group'] = 'Breath of the Wild'
    totk_df['Game_Group'] = 'Tears of the Kingdom'
    compare_df = pd.concat([botw_df, totk_df])

    print("⚖️ 2. 데이터 불균형 해소를 위한 각 타이틀별 층화 하향 샘플링(Stratified Down-sampling) 개시...")

    # 야숨 최소 그룹 기준 밸런싱
    botw_min = botw_counts.min() if len(botw_counts) > 0 else 10
    botw_balanced = botw_df.groupby('Label').sample(n=botw_min, random_state=42)

    # 왕눈 최소 그룹 기준 밸런싱
    totk_min = totk_counts.min() if len(totk_counts) > 0 else 10
    totk_balanced = totk_df.groupby('Label').sample(n=totk_min, random_state=42)

    # 최종 MobileBERT 학습셋 조립
    final_train_df = pd.concat([botw_balanced, totk_balanced])
    final_train_df.to_csv("zelda_train_final.csv", index=False, encoding='utf-8-sig')

    print(f"🏁 MobileBERT 비교 학습용 최종 데이터 'zelda_train_final.csv' 생성 완료! (총 {len(final_train_df):,}건)")

    print("\n📈 3. [가산점 치트키] 보고서 첨부용 야숨 vs 왕눈 감성 비교 시각화 차트 생성 중...")

    # 폰트 깨짐 및 마이너스 기호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    # Seaborn을 활용한 깔끔한 감성별 막대그래프 구현
    sns.countplot(x='Game_Group', hue='Label', data=compare_df, palette='Set2')
    plt.title('Zelda BOTW vs TOTK User Sentiment Comparison (Massive Pool)')
    plt.xlabel('Game Title')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment', labels=['Negative (-1)', 'Neutral (0)', 'Positive (1)'])

    # 이미지 파일 저장
    plt.savefig('zelda_title_comparison.png', dpi=300)
    print("📊 보고서의 하이라이트가 될 'zelda_title_comparison.png' 비교 차트 저장 성공!")


if __name__ == "__main__":
    process_massive_zelda_data()