import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import ndimage, stats
import statsmodels.formula.api as smf

st.set_page_config(page_title="Brain Structure Explorer", page_icon="🧠", layout="wide")

URL = "https://s3.amazonaws.com/openneuro.org/ds003826/participants.tsv"
NA_VALUES = ["n/a", "N/A", "NA", ""]


# ---------------------------------------------------------------- 데이터 준비
@st.cache_data(show_spinner="OpenNeuro에서 데이터를 불러오는 중...")
def load_from_url():
    return pd.read_csv(URL, sep="\t", na_values=NA_VALUES)


def prepare(df):
    df = df.copy()
    cth_cols = [c for c in df.columns if "CTh" in c]
    if cth_cols:
        df = df.rename(columns={cth_cols[0]: "CTh"})
    for c in ["age", "PSQI", "ESS", "CTh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "group" in df.columns:
        df["group"] = df["group"].astype(str)
    return df


def p_text(p):
    if p < 0.05:
        return "통계적으로 유의한 관계가 관찰되었습니다 (p < 0.05)."
    return "이 데이터에서는 통계적으로 뚜렷한 관계가 관찰되지 않았습니다 (p ≥ 0.05)."


# ---------------------------------------------------------------- 사이드바
st.sidebar.title("🧠 Brain Structure Explorer")
st.sidebar.warning(
    "이 앱의 결과는 개인의 뇌 상태나 질병을 진단하지 않으며, "
    "인과관계를 증명하지 않습니다. 학습용 탐색 분석입니다."
)
uploaded = st.sidebar.file_uploader("participants.tsv 직접 업로드 (선택)", type=["tsv", "csv"])

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded, sep=None, engine="python", na_values=NA_VALUES)
    else:
        raw = load_from_url()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 왼쪽에서 participants.tsv를 직접 업로드해 주세요.")
    st.caption(f"오류 내용: {e}")
    st.stop()

df = prepare(raw)
needed = {"PSQI", "CTh"}
if not needed.issubset(df.columns):
    st.error(f"필요한 열이 없습니다: {needed - set(df.columns)}. 열 이름: {list(df.columns)}")
    st.stop()

data = df.dropna(subset=["PSQI", "CTh"]).copy()

page = st.sidebar.radio(
    "메뉴",
    ["1. 데이터셋 소개", "2. 참가자 특성", "3. 수면 점수와 피질 두께",
     "4. 집단 비교", "5. 나이 보정 분석", "6. VBM 학습", "7. 논문 한계 퀴즈"],
)
st.sidebar.caption(f"전체 {len(df)}명 중 분석 대상 {len(data)}명 (PSQI·피질 두께 모두 있음)")

# ---------------------------------------------------------------- 1. 데이터셋 소개
if page.startswith("1"):
    st.title("수면과 뇌 구조 데이터 탐구 앱")
    st.write(
        "OpenNeuro **ds003826** (18~35세 건강한 성인의 T1 구조 MRI와 수면 설문 자료)를 "
        "이용해 수면의 질과 뇌 구조 지표의 관계를 탐색합니다."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 참가자 수", len(df))
    c2.metric("분석 대상 수", len(data))
    if "age" in df.columns:
        c3.metric("나이 범위", f"{df['age'].min():.0f} ~ {df['age'].max():.0f}세")

    st.subheader("변수 설명")
    st.table(pd.DataFrame({
        "변수": ["age", "sex", "group", "PSQI", "ESS", "CTh (L_Ent/FFG_CTh)"],
        "의미": [
            "나이", "성별", "아침형·저녁형 집단",
            "Pittsburgh Sleep Quality Index. 높을수록 수면의 질이 좋지 않음",
            "Epworth Sleepiness Scale. 높을수록 낮 졸림이 심함",
            "왼쪽 내후각·방추상회 클러스터의 평균 피질 두께 (전체 뇌 부피가 아님)",
        ],
    }))
    st.info("이 데이터에는 과로 집단이 없으므로 논문(Overwork and changes in brain structure)을 "
            "재현하는 것이 아니라, 수면이라는 보완 요인을 탐색하는 용도입니다.")
    st.subheader("데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------------------------------------------- 2. 참가자 특성
elif page.startswith("2"):
    st.title("참가자 특성 시각화")
    c1, c2 = st.columns(2)
    if "age" in data.columns:
        c1.plotly_chart(px.histogram(data, x="age", nbins=12, title="나이 분포"), use_container_width=True)
    c2.plotly_chart(px.histogram(data, x="PSQI", nbins=12, title="PSQI 분포 (높을수록 수면의 질 나쁨)"),
                    use_container_width=True)
    c3, c4 = st.columns(2)
    if "ESS" in data.columns:
        c3.plotly_chart(px.histogram(data.dropna(subset=["ESS"]), x="ESS", nbins=12,
                                     title="ESS 분포 (높을수록 낮 졸림 심함)"), use_container_width=True)
    if "group" in data.columns:
        gc = data["group"].value_counts().reset_index()
        gc.columns = ["group", "인원"]
        c4.plotly_chart(px.bar(gc, x="group", y="인원", title="아침형/저녁형 집단 인원"),
                        use_container_width=True)

# ---------------------------------------------------------------- 3. 수면 점수와 피질 두께
elif page.startswith("3"):
    st.title("수면 점수와 피질 두께")
    color = "group" if "group" in data.columns else None
    show_line = st.checkbox("추세선 표시", value=True)
    fig = px.scatter(data, x="PSQI", y="CTh", color=color,
                     trendline="ols" if show_line and color is None else None,
                     labels={"PSQI": "PSQI (높을수록 수면의 질 나쁨)", "CTh": "평균 피질 두께 (mm)"})
    st.plotly_chart(fig, use_container_width=True)
    if show_line and color is not None:
        st.caption("집단별로 색을 나눈 경우 추세선은 표시하지 않습니다. 전체 추세선은 '집단 색 구분'을 끈 아래 그래프를 보세요.")
        st.plotly_chart(px.scatter(data, x="PSQI", y="CTh", trendline="ols"), use_container_width=True)

    r, p = stats.pearsonr(data["PSQI"], data["CTh"])
    rho, ps = stats.spearmanr(data["PSQI"], data["CTh"])
    c1, c2 = st.columns(2)
    c1.metric("Pearson r", f"{r:.3f}", f"p = {p:.4f}", delta_color="off")
    c2.metric("Spearman ρ", f"{rho:.3f}", f"p = {ps:.4f}", delta_color="off")
    st.write(p_text(p))
    st.info("상관관계는 두 변수가 함께 변하는 경향일 뿐, 수면이 피질 두께를 바꾼다는 뜻이 아닙니다.")

# ---------------------------------------------------------------- 4. 집단 비교
elif page.startswith("4"):
    st.title("아침형 vs 저녁형 집단 비교")
    if "group" not in data.columns:
        st.warning("group 열이 없습니다.")
        st.stop()
    var = st.selectbox("비교할 변수", ["PSQI", "CTh"], format_func=lambda v: {"PSQI": "PSQI (수면의 질)", "CTh": "피질 두께"}[v])
    st.plotly_chart(px.box(data, x="group", y=var, points="all"), use_container_width=True)
    st.dataframe(data.groupby("group")[var].agg(["count", "mean", "std"]).round(3), use_container_width=True)

    groups = data["group"].unique()
    if len(groups) == 2:
        a = data.loc[data["group"] == groups[0], var]
        b = data.loc[data["group"] == groups[1], var]
        t, pt = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b)
        st.write(f"Welch t-test p = **{pt:.4f}**, Mann-Whitney U p = **{pu:.4f}**")
        st.write("두 집단의 평균 차이가 우연이라고 보기 어렵습니다." if pt < 0.05
                 else "두 집단의 평균 차이가 뚜렷하지 않습니다.")
    else:
        st.info(f"집단이 2개가 아니어서 검정은 생략합니다. (집단: {list(groups)})")

# ---------------------------------------------------------------- 5. 나이 보정 분석
elif page.startswith("5"):
    st.title("나이 보정 회귀분석")
    if "age" not in data.columns:
        st.warning("age 열이 없습니다.")
        st.stop()
    reg = data.dropna(subset=["age"])
    options = {"PSQI만": "CTh ~ PSQI", "PSQI + 나이": "CTh ~ PSQI + age"}
    if "sex" in reg.columns and reg["sex"].notna().any():
        options["PSQI + 나이 + 성별"] = "CTh ~ PSQI + age + C(sex)"

    rows, models = [], {}
    for name, formula in options.items():
        m = smf.ols(formula, data=reg.dropna(subset=["sex"]) if "sex" in formula else reg).fit()
        models[name] = m
        rows.append({"모델": name, "PSQI 계수": m.params["PSQI"], "p값": m.pvalues["PSQI"],
                     "R²": m.rsquared, "n": int(m.nobs)})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)

    sel = st.selectbox("자세히 볼 모델", list(models.keys()), index=min(1, len(models) - 1))
    m = models[sel]
    st.write(pd.DataFrame({"계수": m.params, "p값": m.pvalues}).round(4))

    # 나이 영향을 제거한 부분상관
    res_y = smf.ols("CTh ~ age", data=reg).fit().resid
    res_x = smf.ols("PSQI ~ age", data=reg).fit().resid
    r_p, p_p = stats.pearsonr(res_x, res_y)
    fig = px.scatter(x=res_x, y=res_y, trendline="ols",
                     labels={"x": "PSQI (나이 영향 제거)", "y": "피질 두께 (나이 영향 제거)"},
                     title=f"나이 보정 후 관계 (부분상관 r = {r_p:.2f}, p = {p_p:.3f})")
    st.plotly_chart(fig, use_container_width=True)
    st.write(p_text(models["PSQI + 나이"].pvalues["PSQI"]))
    st.info("나이를 보정해도 수면, 스트레스, 운동 같은 다른 교란변수는 남아 있어 인과관계로 해석할 수 없습니다.")

# ---------------------------------------------------------------- 6. VBM 학습
elif page.startswith("6"):
    st.title("VBM (복셀 기반 형태 분석) 학습")
    tabs = st.tabs(["① 복셀", "② 조직 분할", "③ 표준화", "④ 스무딩", "⑤ 통계 지도"])

    with tabs[0]:
        st.write("**복셀(voxel)** = volume + pixel. 3차원 MRI를 이루는 작은 정육면체이며, "
                 "각 복셀은 해당 위치의 신호 값을 가집니다. 복셀 하나가 신경세포 하나를 뜻하지는 않습니다.")
        n = st.slider("격자 해상도 (한 변의 복셀 수)", 4, 40, 12)
        yy, xx = np.mgrid[0:n, 0:n]
        c = (n - 1) / 2
        img = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * (n / 4) ** 2)))
        fig = px.imshow(img, color_continuous_scale="gray", title=f"{n}×{n} 복셀로 나눈 2D 단면 (예시)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("복셀 수가 많을수록 세밀하지만 계산량이 늘어납니다.")

    with tabs[1]:
        st.write("T1 강조 MRI에서 각 복셀을 **회백질 · 백질 · 뇌척수액**일 확률로 나눕니다. "
                 "VBM은 주로 회백질의 분포와 양을 비교합니다.")
        st.table(pd.DataFrame({
            "조직": ["회백질", "백질", "뇌척수액"],
            "특징": ["신경세포 세포체가 많음", "신경섬유(축삭)가 많음", "뇌와 척수를 채우는 액체"],
            "T1 영상 밝기": ["중간(회색)", "밝음", "어두움"],
        }))

    with tabs[2]:
        st.write("사람마다 뇌의 크기와 모양이 다르므로 모든 MRI를 **MNI152 표준 뇌 템플릿**에 맞춥니다 (공간 정규화).")
        st.warning("정규화가 부정확하면 실제 조직 차이가 아니라 영상을 맞추는 과정의 오류가 차이처럼 나타날 수 있습니다.")

    with tabs[3]:
        st.write("주변 복셀과 평균을 내어 잡음을 줄이는 과정입니다. 논문에서는 8mm 가우시안 스무딩을 사용했습니다.")
        fwhm = st.slider("스무딩 강도 (FWHM, 복셀 단위)", 0.0, 10.0, 0.0, 0.5)
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:64, 0:64]
        signal = 1.5 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 5 ** 2)))
        noisy = signal + rng.normal(0, 1.0, (64, 64))
        out = ndimage.gaussian_filter(noisy, sigma=fwhm / 2.355) if fwhm > 0 else noisy
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.imshow(noisy, color_continuous_scale="gray", title="원본 (신호 + 잡음)"), use_container_width=True)
        c2.plotly_chart(px.imshow(out, color_continuous_scale="gray", title=f"스무딩 후 (FWHM={fwhm})"), use_container_width=True)
        st.caption("약하면 잡음이 남고, 너무 강하면 차이가 있는 위치가 흐려집니다.")

    with tabs[4]:
        st.write("모든 복셀에서 두 집단을 비교해 **t값 지도**를 만들고, 기준을 넘는 복셀을 색으로 표시합니다. "
                 "아래는 가상의 데이터로 만든 예시입니다.")
        thr = st.slider("t값 기준 (|t| 이상만 표시)", 0.0, 6.0, 2.0, 0.5)
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:40, 0:40]
        blob = 0.9 * np.exp(-(((xx - 25) ** 2 + (yy - 15) ** 2) / (2 * 4 ** 2)))
        ga = np.array([ndimage.gaussian_filter(rng.normal(0, 1, (40, 40)), 1.5) * 2.5 for _ in range(20)])
        gb = np.array([ndimage.gaussian_filter(rng.normal(0, 1, (40, 40)), 1.5) * 2.5 + blob for _ in range(20)])
        t_map, _ = stats.ttest_ind(gb, ga, axis=0)
        shown = np.where(np.abs(t_map) >= thr, t_map, np.nan)
        st.plotly_chart(px.imshow(shown, color_continuous_scale="RdBu_r", zmin=-6, zmax=6,
                                  title=f"가상 t값 지도 (|t| ≥ {thr})"), use_container_width=True)
        st.info("복셀이 수만~수십만 개이므로 다중비교 보정이 필요합니다. 기준이 낮으면 우연으로 생긴 영역도 나타납니다.")

# ---------------------------------------------------------------- 7. 퀴즈
else:
    st.title("논문 한계 퀴즈")
    quiz = [
        {"q": "논문은 한 시점에서 과로 집단과 비과로 집단의 MRI를 비교했다. 이 연구 설계는?",
         "o": ["종단 연구", "횡단면 연구", "무작위 대조 실험", "사례 보고"], "a": 1,
         "e": "한 시점에서 집단을 비교하는 횡단면 연구라 원인과 결과의 순서를 확정하기 어렵습니다."},
        {"q": "과로 집단에서 중간 전두회 값이 더 크게 나왔다. 올바른 해석은?",
         "o": ["과로가 뇌를 변화시켰다", "과로 집단과 비과로 집단 사이에 구조적 차이가 관찰되었다",
               "과로하면 뇌 기능이 좋아진다", "과로하면 뇌 기능이 나빠진다"], "a": 1,
         "e": "관찰된 것은 '차이'이며, 원인이나 기능의 좋고 나쁨까지 단정할 수 없습니다."},
        {"q": "근무시간과 뇌 구조 모두에 영향을 줄 수 있는 변수(수면, 스트레스 등)를 무엇이라 하는가?",
         "o": ["교란변수", "종속변수", "상수", "표준편차"], "a": 0,
         "e": "교란변수를 통제하지 않으면 두 변수의 관계가 왜곡될 수 있습니다."},
        {"q": "PSQI 점수가 높다는 것은?",
         "o": ["수면의 질이 좋다", "수면의 질이 좋지 않다", "낮에 졸리다", "아침형 인간이다"], "a": 1,
         "e": "PSQI는 높을수록 수면의 질이 좋지 않음을 뜻합니다. 낮 졸림은 ESS입니다."},
        {"q": "스무딩을 너무 강하게 하면 생기는 문제는?",
         "o": ["잡음이 늘어난다", "차이가 발생한 위치가 흐려진다", "복셀 수가 늘어난다", "MRI 촬영이 다시 필요하다"], "a": 1,
         "e": "잡음은 줄지만 작은 영역의 차이와 정확한 위치 정보가 흐려집니다."},
    ]
    with st.form("quiz"):
        answers = [st.radio(f"Q{i+1}. {q['q']}", q["o"], index=None, key=f"q{i}") for i, q in enumerate(quiz)]
        submitted = st.form_submit_button("채점하기")
    if submitted:
        score = 0
        for i, (q, ans) in enumerate(zip(quiz, answers)):
            ok = ans == q["o"][q["a"]]
            score += ok
            (st.success if ok else st.error)(f"Q{i+1}. {'정답' if ok else '오답'} — {q['e']}")
        st.metric("점수", f"{score} / {len(quiz)}")
