import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Page Setup ---
st.set_page_config(page_title="Bulk Amazon Review Analyzer", layout="wide")
st.title("📊 Bulk Review Analyzer + Performance Report")

# --- Load Model ---
@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    batch_size=32)

classifier = load_model()
st.success("✅ 5-Star AI Model is loaded!")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 How to use")
    st.markdown("""
    1. Upload a **CSV**.
    2. Needs a **`text`** column (review text).
    3. Optional: a **`label`** column for performance metrics.
    4. Click **"🚀 Analyze All"**.
    """)

# --- Upload ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Smart column detection ---
    text_col = None
    for col in ['text', 'review_text', 'reviewText', 'review']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        st.error("❌ No review text column found. Please name it 'text' or 'review_text'.")
        st.stop()

    label_col = None
    for col in ['label', 'sentiment', 'target']:
        if col in df.columns:
            label_col = col
            break

    # --- Preview ---
    st.subheader("📄 Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Total rows: {len(df)}")
    if label_col:
        st.caption(f"✅ Found ground-truth labels in column '{label_col}' – performance report enabled!")
    else:
        st.caption("ℹ️ No label column found – performance report will be skipped.")

    # --- Analyze Button ---
    if st.button("🚀 Analyze All Reviews", type="primary"):
        with st.spinner(f"🧠 AI is analyzing {len(df)} reviews..."):
            texts = df[text_col].astype(str).tolist()
            results = classifier(texts)

            df['predicted_stars'] = [int(r['label'].split()[0]) for r in results]
            df['confidence_score'] = [r['score'] for r in results]

            def categorize(stars):
                if stars >= 4: return "🟢 Positive"
                elif stars == 3: return "🟡 Neutral"
                else: return "🔴 Negative"

            df['sentiment_category'] = df['predicted_stars'].apply(categorize)

        st.success("✅ Analysis Complete!")

        # --- Dashboard ---
        st.subheader("📈 Sentiment Dashboard")
        col1, col2, col3 = st.columns(3)
        col1.metric("⭐ Average Rating", f"{df['predicted_stars'].mean():.2f} / 5.0")
        col2.metric("📝 Total Reviews", len(df))
        col3.metric("👍 Positive Rate", f"{(df['sentiment_category'] == '🟢 Positive').mean() * 100:.1f}%")

        # Chart 1: Star Distribution
        st.subheader("Distribution of Star Ratings")
        fig_stars = px.histogram(df, x="predicted_stars", nbins=5,
                                 color_discrete_sequence=["#FFA500"])
        st.plotly_chart(fig_stars, use_container_width=True)

        # Chart 2: Sentiment Pie
        st.subheader("Overall Sentiment")
        sentiment_counts = df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment',
                         color='Sentiment',
                         color_discrete_map={"🟢 Positive": "#00CC96",
                                             "🟡 Neutral": "#EFC14E",
                                             "🔴 Negative": "#EF553B"},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

        # ============================================================
        # ============ LEVEL 6: PERFORMANCE REPORT ===================
        # ============================================================
        if label_col:
            st.divider()
            st.header("🎯 Model Performance Report")
            st.markdown("Comparing our **5-star AI predictions** against the **true labels**.")

            # --- Show what the labels look like ---
            st.write("🔍 Sample of true labels found in the dataset:")
            st.write(df[label_col].unique()[:5])

            # --- Convert labels to binary (handles 0/1 AND "POSITIVE"/"NEGATIVE") ---
            def convert_to_binary(value):
                """Handles both numeric (0/1) and text (POSITIVE/NEGATIVE) labels"""
                if pd.isna(value):
                    return None
                try:
                    return int(value)  # Already 0 or 1
                except (ValueError, TypeError):
                    pass
                val_str = str(value).upper().strip()
                if val_str in ["POSITIVE", "POS", "1", "TRUE"]:
                    return 1
                elif val_str in ["NEGATIVE", "NEG", "0", "FALSE"]:
                    return 0
                return None

            df['true_binary'] = df[label_col].apply(convert_to_binary)

            # --- Convert stars -> binary prediction (1-2=Neg, 3=skip, 4-5=Pos) ---
            df['binary_prediction'] = df['predicted_stars'].apply(
                lambda s: 1 if s >= 4 else (0 if s <= 2 else None)
            )

            # --- Filter valid rows ---
            valid_df = df.dropna(subset=['binary_prediction', 'true_binary'])
            excluded = len(df) - len(valid_df)

            if len(valid_df) == 0:
                st.warning("⚠️ No valid rows to evaluate. Check your label column format.")
            else:
                st.info(f"🔢 Evaluated **{len(valid_df)}** reviews "
                        f"(excluded **{excluded}** ambiguous rows).")

                y_true = valid_df['true_binary'].astype(int)
                y_pred = valid_df['binary_prediction'].astype(int)

                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                # KPI metrics
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
                kpi2.metric("⚖️ F1-Score", f"{f1 * 100:.2f}%")
                kpi3.metric("✅ Correct", f"{(y_true == y_pred).sum()} / {len(valid_df)}")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=['Predicted Neg', 'Predicted Pos'],
                    y=['True Neg', 'True Pos'],
                    colorscale='Blues',
                    showscale=True
                )
                fig_cm.update_layout(title="How well did the AI classify each category?")
                st.plotly_chart(fig_cm, use_container_width=True)

                # Breakdown
                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                ### 📖 What do these numbers mean?

                | Term | Count | Meaning |
                |------|-------|---------|
                | ✅ **True Positives** | {tp} | Correctly identified positive reviews |
                | ✅ **True Negatives** | {tn} | Correctly identified negative reviews |
                | ❌ **False Positives** | {fp} | Said positive, but was actually negative |
                | ❌ **False Negatives** | {fn} | Said negative, but was actually positive |

                **Business Interpretation:**
                - **Accuracy** = % of all predictions that were correct
                - **F1-Score** = Balance between precision and recall (great when classes are imbalanced)
                """)

                # Interpretation hint
                if accuracy >= 0.85:
                    st.success("🌟 Excellent performance! This AI is production-ready.")
                elif accuracy >= 0.75:
                    st.info("👍 Good performance. Could be improved with fine-tuning.")
                else:
                    st.warning("⚠️ Below 75% accuracy. Consider using a fine-tuned model.")
        # ============================================================
        # ================ END OF LEVEL 6 SECTION ====================
        # ============================================================

        # Full table
        st.subheader("📋 Enriched Data")
        st.dataframe(df, use_container_width=True)

        # Download
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV",
                           data=csv_download,
                           file_name='analyzed_reviews.csv',
                           mime='text/csv',
                           use_container_width=True)

else:
    st.info("👈 Upload a CSV file to begin.")