"""Asteria 额外审核工作台（演示版）。

运行：在仓库根目录执行  streamlit run review_app.py
只读取 outputs/ 里已经生成的结果，不重新训练模型，也不改动任何文件。
"""

import math
from pathlib import Path

import pandas as pd
import streamlit as st

# 数据文件夹：和本文件放在一起的 outputs/（这样无论从哪里启动、部署到哪里都能找到）
OUT = Path(__file__).parent / "outputs"

# ---------- 1. 页面设置 ----------
st.set_page_config(page_title="Asteria Review Console", page_icon="🗂️", layout="wide")

# ---------- 2. 风险原因的"人话"对照表 ----------
# 成员 D 的建议：给审核员看的原因只能是可核实的财务信息（外部评分、负债、还款记录、贷款结构）。
# 所以把 1,210 个特征归成 7 个主题，每个主题配一句说明和一个审核员该核实的问题。
THEMES = {
    "External credit scores": {
        "icon": "📉",
        "why": "External credit scores are low compared with other applicants.",
        "check": "Is the low external score consistent with the applicant's own repayment record?",
    },
    "Employment stability": {
        "icon": "💼",
        "why": "Short time in the current job.",
        "check": "Is the income stable and verifiable (payslips, employment contract)?",
    },
    "Loan size and repayments": {
        "icon": "💷",
        "why": "The loan or its repayments are high relative to income or to the goods bought.",
        "check": "Is the repayment affordable on verified income? What does credit above the goods price pay for?",
    },
    "Debts with other lenders": {
        "icon": "🏦",
        "why": "A large share of credit from other lenders is still owed, or was overdue.",
        "check": "What are the applicant's current external commitments?",
    },
    "Repayment record with Asteria": {
        "icon": "🧾",
        "why": "Earlier instalments were paid late or not in full.",
        "check": "Has any past instalment gone unpaid by its due date, and why?",
    },
    "Previous applications": {
        "icon": "📂",
        "why": "Pattern of earlier applications (refusals, amounts granted below amounts requested).",
        "check": "Why were earlier applications refused or reduced? A refusal alone is not a credit defect.",
    },
    "Card and store-credit use": {
        "icon": "💳",
        "why": "Heavy use of credit cards or point-of-sale credit.",
        "check": "Is card utilisation high or rising? Are minimum payments being met?",
    },
}
LOAN_FEATURES = {
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL", "CREDIT_TO_GOODS",
    "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "NAME_CONTRACT_TYPE",
}


def theme_of(feature: str):
    """把一个特征名归到一个主题；不属于可核实财务信息的返回 None（不展示给审核员）。"""
    if feature.startswith("EXT_SOURCE"):
        return "External credit scores"
    if feature in ("DAYS_EMPLOYED", "EMPLOYMENT_YEARS", "EMPLOYMENT_SENTINEL"):
        return "Employment stability"
    if feature in LOAN_FEATURES:
        return "Loan size and repayments"
    if feature.startswith("BUREAU") or feature == "EXTERNAL_DEBT_TO_INCOME":
        return "Debts with other lenders"
    if feature.startswith("INST"):
        return "Repayment record with Asteria"
    if feature.startswith("PREV"):
        return "Previous applications"
    if feature.startswith(("CARD", "POS")):
        return "Card and store-credit use"
    return None


def show_value(feature: str, value) -> str:
    """把特征值写成容易读的样子。"""
    if pd.isna(value):
        return "not available"
    if feature == "DAYS_EMPLOYED":
        return f"{-value / 365.25:.1f} years in current job"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


# ---------- 3. 读取数据 ----------
@st.cache_data
def load():
    scores = pd.read_csv(OUT / "holdout_catboost_full.csv").set_index("SK_ID_CURR")
    shap = pd.read_parquet(OUT / "shap_values.parquet")  # 500 个留出集申请人的逐人解释
    features = pd.read_parquet(OUT / "development_features.parquet").loc[shap.index]
    batch = scores.loc[shap.index].copy()  # 演示批次 = 有逐人解释的这 500 人
    batch["loan_amount"] = features["AMT_CREDIT"]
    batch["exposure"] = batch["calibrated_probability"] * batch["loan_amount"]
    # 每个人、每个主题的 SHAP 贡献之和（正数 = 这个主题把风险推高）
    themes = pd.Series({c: theme_of(c) for c in shap.columns}).dropna()
    theme_shap = shap[themes.index].T.groupby(themes).sum().T
    batch["main_reason"] = theme_shap.idxmax(axis=1).where(theme_shap.max(axis=1) > 0, "No strong factor")
    metrics = pd.read_csv(OUT / "metrics.csv")
    strategies = pd.read_csv(OUT / "review_strategies.csv")
    descriptions = pd.read_csv(OUT / "feature_dictionary.csv").set_index("feature")["description"]
    return batch, shap, features, themes, theme_shap, metrics, strategies, descriptions


batch, shap, features, themes, theme_shap, metrics, strategies, descriptions = load()
final = metrics[(metrics["model"] == "catboost_full") & (metrics["phase"] == "holdout_calibrated_f2")].iloc[0]
LABEL_THRESHOLD = float(final["threshold"])  # predictions.csv 用的固定阈值（F2 最优，0.0858）
BASE_RATE = final["positives"] / final["n"]  # 平均还款困难率 8.07%

# ---------- 4. 记住审核员做过的决定 ----------
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

# ---------- 5. 侧边栏：审核员、审核能力、排序方式 ----------
st.sidebar.title("🗂️ Review Console")
# 审核员：队列里的案例按名次轮流分给 3 位审核员，每人只看到自己的案例和进度；组长能看到全部。
REVIEWERS = ["Underwriter 1", "Underwriter 2", "Underwriter 3"]
TEAM_LEAD = "Team lead (all cases)"
reviewer = st.sidebar.selectbox("Signed in as", REVIEWERS + [TEAM_LEAD])
capacity = st.sidebar.radio("Extra-review capacity", [0.1, 0.2, 0.3], index=1, format_func=lambda c: f"Top {c:.0%}")
ranking = st.sidebar.radio(
    "Rank applicants by",
    ["probability", "exposure"],
    format_func=lambda r: "Risk of difficulty" if r == "probability" else "Risk x loan amount",
)
if st.sidebar.button("Start a new session"):
    st.session_state.decisions = {}
    st.rerun()
st.sidebar.caption(
    "Demo on 500 labelled holdout applicants. The score ranks applicants for extra review. "
    "It does not approve or decline anyone; every applicant still goes through standard underwriting."
)

# 队列按"名次"定义：取这一批里分数最高的 ceil(n x capacity) 个人（策略文档第 3 节）
score_column = "calibrated_probability" if ranking == "probability" else "exposure"
batch = batch.sort_values(score_column, ascending=False)
n_queue = math.ceil(len(batch) * capacity)
GROUPS = ["Extra-review queue", "Lighter checks", "Standard underwriting"]
batch["group"] = GROUPS[2]
batch.loc[batch["calibrated_probability"] >= LABEL_THRESHOLD, "group"] = GROUPS[1]
batch.iloc[:n_queue, batch.columns.get_loc("group")] = GROUPS[0]
queue = batch.iloc[:n_queue].copy()
queue["status"] = [st.session_state.decisions.get(i, {}).get("outcome", "Pending") for i in queue.index]
# 分案：第 1 名给审核员 1，第 2 名给审核员 2，第 3 名给审核员 3，第 4 名又回到审核员 1……
#   这样每个人拿到的案例数量和风险高低都差不多。
queue["assigned_to"] = [REVIEWERS[k % len(REVIEWERS)] for k in range(len(queue))]
team_queue = queue  # 整个团队的队列（组长视图和团队进度用）
if reviewer != TEAM_LEAD:
    queue = queue[queue["assigned_to"] == reviewer]  # 当前审核员自己的案例
n_mine = len(queue)
n_done = int((queue["status"] != "Pending").sum())
strategy = strategies[
    (strategies["model"] == "catboost_full")
    & (strategies["strategy"] == ranking)
    & (strategies["capacity"].round(2) == capacity)
].iloc[0]

st.title("Asteria Finance · Extra-Review Console")
tab_today, tab_case, tab_record = st.tabs(["① Today's workload", "② Review a case", "③ Record & outcomes"])

# ---------- 6. 今日工作量：三个处理组 + 这个队列值不值得审 ----------
with tab_today:
    counts = batch["group"].value_counts()
    a, b, c, d = st.columns(4)
    a.metric("Applications in this batch", len(batch))
    b.metric("① Extra-review queue", int(counts.get(GROUPS[0], 0)), help="Top of the ranking: full extra review")
    c.metric("② Lighter checks", int(counts.get(GROUPS[1], 0)), help=f"Above the label threshold {LABEL_THRESHOLD:.4f}, outside the queue")
    d.metric("③ Standard underwriting", int(counts.get(GROUPS[2], 0)), help="Below the label threshold")
    who = "the team" if reviewer == TEAM_LEAD else reviewer
    st.progress(n_done / n_mine, text=f"{who}: {n_done} of {n_mine} assigned applications reviewed")
    if reviewer == TEAM_LEAD:
        progress = team_queue.groupby("assigned_to")["status"].agg(
            assigned="count", reviewed=lambda s: int((s != "Pending").sum())
        )
        st.dataframe(progress, width="stretch")
    else:
        team_done = int((team_queue["status"] != "Pending").sum())
        st.caption(f"Whole team: {team_done} of {n_queue} queued applications reviewed.")

    st.info(
        f"**Why this queue is worth the effort.** On the 5,400-applicant holdout, extra review of the top "
        f"{capacity:.0%} finds **{int(strategy['cases_captured'])} of {int(final['positives'])}** applicants who later "
        f"had repayment difficulty ({strategy['recall']:.1%}). About 1 in {1 / strategy['precision']:.0f} queued "
        f"applicants is a real case, **{strategy['lift']:.2f}x** what a random review would find."
    )
    with st.expander("Model evidence behind the ranking"):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("PR-AUC (primary)", f"{final['pr_auc_ap']:.3f}", help=f"A random ranking scores about {BASE_RATE:.3f}")
        e2.metric("ROC-AUC", f"{final['roc_auc']:.3f}")
        e3.metric("Brier score", f"{final['brier']:.4f}", help="Lower = more reliable probabilities")
        e4.metric("Difficulty-case credit covered", f"{strategy['difficulty_credit_coverage']:.1%}")

    st.subheader("Extra-review queue: whole team" if reviewer == TEAM_LEAD else f"My cases: {reviewer}")
    table = queue.reset_index()[["SK_ID_CURR", "calibrated_probability", "loan_amount", "main_reason", "assigned_to", "status"]]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "SK_ID_CURR": st.column_config.NumberColumn("Applicant", format="%d"),
            "calibrated_probability": st.column_config.ProgressColumn(
                "Risk of difficulty", format="percent", min_value=0, max_value=float(batch["calibrated_probability"].max())
            ),
            "loan_amount": st.column_config.NumberColumn("Loan amount", format="%,.0f"),
            "main_reason": "Main risk factor",
            "assigned_to": "Assigned to",
            "status": "Status",
        },
    )

# ---------- 7. 审核一个申请人 ----------
with tab_case:
    pending = list(queue.index[queue["status"] == "Pending"])
    if reviewer == TEAM_LEAD:
        st.info("The team lead monitors progress and records. Sign in as an underwriter to review cases.")
    elif not pending:
        st.success("Every application assigned to you has been reviewed. See the record in tab ③.")
    else:
        pick = st.selectbox("Application under review (next pending is selected automatically)", pending)
        person = batch.loc[pick]
        facts = features.loc[pick]
        probability = person["calibrated_probability"]

        left, right = st.columns([1, 1.4])
        with left:
            st.metric("Risk of repayment difficulty", f"{probability:.1%}", f"{probability / BASE_RATE:.1f}x the average applicant", delta_color="inverse")
            st.progress(min(probability / float(batch["calibrated_probability"].max()), 1.0))
            st.markdown("**Application facts**")
            st.markdown(
                f"- Loan amount: **{facts['AMT_CREDIT']:,.0f}**\n"
                f"- Annual income: **{facts['AMT_INCOME_TOTAL']:,.0f}**\n"
                f"- Repayment per period: **{facts['AMT_ANNUITY']:,.0f}** "
                f"({facts['ANNUITY_TO_INCOME']:.0%} of income)\n"
                f"- Loan is **{facts['CREDIT_TO_INCOME']:.1f}x** annual income"
            )
        with right:
            st.markdown("**Why the model flagged this application** (strongest factors first)")
            reasons = theme_shap.loc[pick].sort_values(ascending=False)
            reasons = reasons[reasons > 0].head(3)
            if reasons.empty:
                st.write("No single factor stands out.")
            for theme in reasons.index:
                info = THEMES[theme]
                with st.container(border=True):
                    st.markdown(f"{info['icon']} **{theme}**: {info['why']}")
                    st.caption(f"Reviewer check: {info['check']}")
                    evidence = shap.loc[pick, themes.index[themes == theme]].sort_values(ascending=False).head(2)
                    with st.expander("Evidence from the data"):
                        for feature in evidence.index:
                            st.write(f"`{feature}` = {show_value(feature, facts[feature])}")
                            st.caption(str(descriptions.get(feature, "")))
            st.caption("Factors describe associations in the model, not causes. The reviewer decides.")

        with st.form("decision_form", clear_on_submit=True):
            st.markdown("**Extra review**")
            checks = st.multiselect(
                "Checks completed",
                ["Income verified", "External commitments reviewed", "Affordability reassessed", "Repayment history discussed"],
            )
            outcome = st.radio(
                "Outcome",
                ["Checks passed: continue standard underwriting", "Request more documents", "Refer to senior underwriter"],
                horizontal=True,
            )
            note = st.text_input("Reason (required for the audit trail)")
            if st.form_submit_button("Save and open next application", type="primary"):
                if not note.strip():
                    st.error("Please record a reason before saving.")
                else:
                    st.session_state.decisions[pick] = {
                        "SK_ID_CURR": pick,
                        "reviewer": reviewer,
                        "risk": round(float(probability), 4),
                        "main_reason": person["main_reason"],
                        "checks": ", ".join(checks),
                        "outcome": outcome,
                        "note": note.strip(),
                    }
                    st.rerun()

# ---------- 8. 记录与结果 ----------
with tab_record:
    st.subheader("Decision record")
    if not st.session_state.decisions:
        st.info("No decisions yet. Review a case in tab ②.")
    else:
        log = pd.DataFrame(st.session_state.decisions.values())
        if reviewer != TEAM_LEAD:
            log = log[log["reviewer"] == reviewer]  # 审核员只看自己的记录，组长看全部
        st.caption(f"Showing {len(log)} record(s) for {'the whole team' if reviewer == TEAM_LEAD else reviewer}.")
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("Download the record (CSV)", log.to_csv(index=False), "decision_log.csv")
        st.caption("Each record keeps the reviewer, the checks made and the reason: the audit trail for monitoring.")

    st.subheader("What happened afterwards (demo only)")
    st.caption("Holdout applicants have known outcomes, so the demo can show how each group turned out.")
    if st.checkbox("Reveal actual outcomes for this batch"):
        outcome_table = batch.groupby("group")["TARGET"].agg(applicants="count", difficulty_cases="sum", difficulty_rate="mean")
        outcome_table = outcome_table.reindex(GROUPS).fillna(0)
        cols = st.columns(3)
        for col, name in zip(cols, GROUPS):
            row = outcome_table.loc[name]
            col.metric(name, f"{row['difficulty_rate']:.1%} had difficulty", f"{int(row['difficulty_cases'])} of {int(row['applicants'])} applicants", delta_color="off")
        found = int(outcome_table.loc[GROUPS[0], "difficulty_cases"])
        total = int(batch["TARGET"].sum())
        st.success(
            f"In this batch, the extra-review queue ({n_queue} of {len(batch)} applications) contained "
            f"**{found} of the {total}** applicants who later had repayment difficulty. "
            f"Applicants outside the queue still had some difficulty cases, which is why everyone keeps standard underwriting."
        )
