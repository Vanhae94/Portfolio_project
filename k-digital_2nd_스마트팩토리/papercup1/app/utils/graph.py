import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib.patches as mpatches

logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL)

# 그래프 저장 경로 설정
STATIC_PATH = os.path.join("app", "static", "images")

# 로깅 설정
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "Malgun Gothic"

# 📌 파이 차트 생성
def create_pie_chart(total_standard, total_break, total_normal):
    try:
        total_standard = 0 if pd.isna(total_standard) else total_standard
        total_break = 0 if pd.isna(total_break) else total_break
        total_normal = 0 if pd.isna(total_normal) else total_normal

        if total_standard == 0 and total_break == 0 and total_normal == 0:
            total_standard, total_break, total_normal = 1, 1, 1

        plt.figure(figsize=(6, 6))
        plt.pie(
            [total_standard, total_break, total_normal], 
            labels=["규격 불량", "파손 불량", "정상"], 
            autopct="%1.1f%%", 
            colors=["red", "blue", "green"], 
            textprops={'fontsize': 10}  # 폰트 크기 조정
        )
        pie_path = os.path.join(STATIC_PATH, "pie_chart.png")
        plt.savefig(pie_path)
        plt.close()

        logger.info(f"✅ 파이 차트 저장 완료: {pie_path}")
        return pie_path

    except Exception as e:
        logger.error(f"🚨 파이 차트 생성 오류: {str(e)}")
        return None

# 📌 막대 차트 생성
def create_bar_chart(total_standard, total_break, total_normal):
    try:
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(
            x=["파손 불량", "규격 불량", "정상"], 
            y=[total_break, total_standard, total_normal], 
            palette=["blue", "red", "green"]
        )

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height:,.0f}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center' if height > 10 else 'bottom', 
                fontsize=9, color='black', xytext=(0, 5), textcoords='offset points'
            )
        bar_path = os.path.join(STATIC_PATH, "bar_chart.png")
        plt.savefig(bar_path)
        plt.close()

        logger.info(f"✅ 막대 차트 저장 완료: {bar_path}")
        return bar_path

    except Exception as e:
        logger.error(f"🚨 막대 차트 생성 오류: {str(e)}")
        return None

# 📌 기간별 비교 막대 차트 (막대 내부 투명, 테두리 색상 설정, 숫자 표시)
def create_daily_bar_chart(df):
    try:
        plt.figure(figsize=(12, 10))
        df_melted = df.melt(id_vars="date", value_vars=["standard", "break", "normal"], 
                            var_name="status", value_name="count")

        # 색상 매핑
        color_map = {"standard": "red", "break": "blue", "normal": "green"}

        # hue 설정을 명확하게 지정
        ax = sns.barplot(
            x="date", y="count", hue="status", data=df_melted, dodge=True, palette=color_map
        )

        # 막대 내부를 투명하게 설정 & 개별 테두리 색상 적용
        for bar, (_, row) in zip(ax.patches, df_melted.iterrows()):
            bar.set_facecolor("none")  # 내부 투명
            bar.set_edgecolor(color_map[row["status"]])  # 개별 테두리 색상

            # 값이 0보다 클 때만 표시
            if bar.get_height() > 0:
                ax.annotate(
                    f'{bar.get_height():,.0f}',  # 천 단위 콤마 포함
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),  
                    ha='center', va='bottom', fontsize=9, color=color_map[row["status"]],
                    xytext=(0, 5), textcoords='offset points'  # 위로 살짝 이동
                )

        # 범례 커스텀 (내부 투명 처리)
        handles = [
            mpatches.Patch(facecolor="none", edgecolor="red", label="규격 불량"),
            mpatches.Patch(facecolor="none", edgecolor="blue", label="파손 불량"),
            mpatches.Patch(facecolor="none", edgecolor="green", label="정상")
        ]
        plt.legend(handles=handles, title="상태")
        plt.xticks(rotation=45)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # 이미지 저장
        daily_bar_path = os.path.join(STATIC_PATH, "daily_bar_chart.png")
        plt.savefig(daily_bar_path)
        plt.close()

        logger.info(f"✅ 기간별 비교 막대 차트 저장 완료: {daily_bar_path}")
        return daily_bar_path

    except Exception as e:
        logger.error(f"🚨 기간별 비교 막대 차트 생성 오류: {str(e)}")
        return None

# 📌 기간별 추이 선 차트
def create_daily_line_chart(df):
    try:
        plt.figure(figsize=(12, 10))
        sns.lineplot(x="date", y="standard", data=df, color="red", label="규격 불량", linewidth=2, alpha=0.8)
        sns.lineplot(x="date", y="break", data=df, color="blue", label="파손 불량", linewidth=2, alpha=0.8)
        sns.lineplot(x="date", y="normal", data=df, color="green", label="정상", linewidth=2, alpha=0.8)

        step = max(1, len(df) // 12)  
        for i in range(0, len(df), step):  
            x, y_standard = df["date"].iloc[i], df["standard"].iloc[i]
            x, y_break = df["date"].iloc[i], df["break"].iloc[i]
            x, y_normal = df["date"].iloc[i], df["normal"].iloc[i]

            plt.text(x, y_standard, f'{y_standard:,.0f}', ha='center', va='bottom', fontsize=8, color='red')
            plt.text(x, y_break, f'{y_break:,.0f}', ha='center', va='bottom', fontsize=8, color='blue')
            plt.text(x, y_normal, f'{y_normal:,.0f}', ha='center', va='bottom', fontsize=8, color='green')

        plt.xticks(rotation=45)
        plt.legend()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        daily_line_path = os.path.join(STATIC_PATH, "daily_line_chart.png")
        plt.savefig(daily_line_path)
        plt.close()

        logger.info(f"✅ 기간별 추이 선 차트 저장 완료: {daily_line_path}")
        return daily_line_path

    except Exception as e:
        logger.error(f"🚨 기간별 추이 선 차트 생성 오류: {str(e)}")
        return None
