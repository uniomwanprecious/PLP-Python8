# CORD-19 Research Data Analysis Report

## I. Executive Summary

This project analyzed 34,332 CORD-19 research papers. The findings show an explosive research response to the pandemic, dominated by a few major journals, with the primary focus being on clinical outcomes and treatment.

## II. Summary of Findings

### A. Temporal Trend
* **Explosive Growth:** Publications increased near-vertically starting in 2020.
* **Peak Output:** Research peaked in 2021 (over 8,000 papers in the sample).

### B. Journal Skew
* **Dominant Journal:** PLoS One leads significantly with nearly 1,400 papers.
* **Concentration:** PLoS One published approximately 64% more than the second-ranked journal.

### C. Research Focus (Top 3 Words)
* The most frequent words are "patients," "health," and "between," confirming a strong clinical focus.

## III. Reflection on Challenges and Learning

### A. Technical Challenges
* **NLTK Failure:** The system could not download the NLTK 'stopwords' data due to a network error (WinError 10060).
* **Solution:** I successfully adapted the code to use a manual list of stop words, ensuring the analysis was completed despite the dependency failure.
* **Streamlit Fix:** The app required the command `python -m streamlit run streamlit_app.py` due to path issues.

### B. Key Takeaways
* **Graceful Degradation:** Learned to use fallback methods (manual stopwords) when dependencies fail.
* **Code Integrity:** Confirmed the importance of using `@st.cache_data` for efficient Streamlit application loading.