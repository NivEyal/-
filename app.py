# app.py
import streamlit as st
import openai
import pandas as pd
import numpy as np
import pymupdf as fitz  # PyMuPDF
import re
import traceback
import io # For handling byte streams from uploaded files
import logging # For bank parsers
import unicodedata # For bank parsers
from datetime import datetime # For bank parsers
import plotly.express as px # For bank parsers
import plotly.graph_objects as go # For bank parsers
import pdfplumber # For Discount & Leumi parsers

# --- OpenAI API Key Setup ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("מפתח OpenAI API לא נמצא. אנא הגדר אותו בקובץ secrets.toml")
    st.stop()
except Exception as e:
    st.error(f"שגיאה בטעינת מפתח OpenAI API: {e}")
    st.stop()

# --- Logging Setup for Bank Parsers ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Streamlit handles logging differently, direct print/st.write might be more visible for now during dev
# For production, proper logging configuration for Streamlit apps can be set.

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$ START: CREDIT REPORT PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COLUMN_HEADER_WORDS = {
    "שם", "מקור", "מידע", "מדווח", "מזהה", "עסקה", "מספר", "עסקאות",
    "גובה", "מסגרת", "מסגרות", "סכום", "הלוואות", "מקורי", "יתרת", "חוב",
    "יתרה", "שלא", "שולמה", "במועד"
}
BANK_KEYWORDS = {"בנק", "בע\"מ", "אגוד", "דיסקונט", "לאומי", "הפועלים", "מזרחי",
                 "טפחות", "הבינלאומי", "מרכנתיל", "אוצר", "החייל", "ירושלים",
                 "איגוד", "מימון", "ישיר", "כרטיסי", "אשראי", "מקס", "פיננסים",
                 "כאל", "ישראכרט"}

def process_entry_final(entry_data, section, all_rows_list):
    if not entry_data or not entry_data.get('bank') or len(entry_data.get('numbers', [])) < 2: return
    bank_name_raw = entry_data['bank']
    bank_name_cleaned = re.sub(r'\s*XX-[\w\d\-]+.*', '', bank_name_raw).strip()
    bank_name_cleaned = re.sub(r'\s+\d{1,3}(?:,\d{3})*$', '', bank_name_cleaned).strip()
    bank_name_cleaned = re.sub(r'\s+בע\"מ$', '', bank_name_cleaned).strip()
    bank_name_final = bank_name_cleaned if bank_name_cleaned else bank_name_raw
    is_likely_bank = any(kw in bank_name_final for kw in ["בנק", "לאומי", "הפועלים", "דיסקונט", "מזרחי", "הבינלאומי", "מרכנתיל", "ירושלים", "איגוד"])
    is_non_bank_entity = any(kw in bank_name_final for kw in ["מימון ישיר", "מקס איט", "כרטיסי אשראי", "כאל", "ישראכרט"])
    if is_likely_bank and not bank_name_final.endswith("בע\"מ"):
        bank_name_final += " בע\"מ"
    elif is_non_bank_entity and not bank_name_final.endswith("בע\"מ"):
         if any(kw in bank_name_final for kw in ["מקס איט פיננסים", "מימון ישיר נדל\"ן ומשכנתאות"]):
              bank_name_final += " בע\"מ"
    numbers = entry_data['numbers']
    num_count = len(numbers)
    limit_col, original_col, outstanding_col, unpaid_col = np.nan, np.nan, np.nan, np.nan
    if num_count >= 2:
        val1 = numbers[0]; val2 = numbers[1]; val3 = numbers[2] if num_count >= 3 else 0.0
        if section in ["עו\"ש", "מסגרת אשראי"]:
            limit_col = val1; outstanding_col = val2; unpaid_col = val3
        elif section in ["הלוואה", "משכנתה"]:
            if num_count >= 3:
                 if val1 < 50 and val1 == int(val1) and num_count >= 4:
                      original_col = numbers[1]; outstanding_col = numbers[2]; unpaid_col = numbers[3]
                 else:
                     original_col = val1; outstanding_col = val2; unpaid_col = val3
            elif num_count == 2:
                 original_col = val1; outstanding_col = val2; unpaid_col = 0.0
        else:
            original_col = val1; outstanding_col = val2; unpaid_col = val3
        all_rows_list.append({
            "סוג עסקה": section, "שם בנק/מקור": bank_name_final, "גובה מסגרת": limit_col,
            "סכום מקורי": original_col, "יתרת חוב": outstanding_col, "יתרה שלא שולמה": unpaid_col
        })

def extract_credit_data_final_v13(pdf_content_bytes, filename_for_logging="UploadedFile"): # Changed pdf_path to pdf_content_bytes
    extracted_rows = []
    st.write(f"\n--- מתחיל עיבוד דוח אשראי: {filename_for_logging} ---")
    try:
        doc = fitz.open(stream=pdf_content_bytes, filetype="pdf") # Use stream for bytes
        current_section = None; current_entry = None; last_line_was_id = False; potential_bank_continuation_candidate = False
        section_patterns = {
            "חשבון עובר ושב": "עו\"ש", "הלוואה": "הלוואה", "משכנתה": "משכנתה",
            "מסגרת אשראי מתחדשת": "מסגרת אשראי",
        }
        number_line_pattern = re.compile(r"^\s*(-?\d{1,3}(?:,\d{3})*\.?\d*)\s*$")
        for page_num, page in enumerate(doc):
            text = page.get_text("text"); lines = text.splitlines()
            for i, line in enumerate(lines):
                original_line = line; line = line.strip()
                if not line: potential_bank_continuation_candidate = False; continue
                is_section_header = False
                for header_keyword, section_name in section_patterns.items():
                    if header_keyword in line and len(line) < len(header_keyword) + 20:
                        if line.count(' ') < 5 :
                            if current_entry and not current_entry.get('processed', False):
                                process_entry_final(current_entry, current_section, extracted_rows)
                            current_section = section_name; current_entry = None
                            last_line_was_id = False; potential_bank_continuation_candidate = False
                            is_section_header = True; break
                if is_section_header: continue
                is_total_line = line.startswith("סה\"כ")
                if is_total_line:
                    if current_entry and not current_entry.get('processed', False):
                        process_entry_final(current_entry, current_section, extracted_rows)
                    current_entry = None; last_line_was_id = False; potential_bank_continuation_candidate = False
                    continue
                if current_section:
                    number_match = number_line_pattern.match(line)
                    is_id_line = line.startswith("XX-") and len(line) > 5
                    is_header_word = any(word == line for word in COLUMN_HEADER_WORDS)
                    is_noise_line = is_header_word or line in [':', '.'] or (len(line)<3 and not line.isdigit())
                    if number_match:
                        if current_entry:
                            try:
                                number = float(number_match.group(1).replace(",", ""))
                                num_list = current_entry.get('numbers', [])
                                if last_line_was_id and len(num_list) >= 2:
                                    if not current_entry.get('processed', False):
                                         process_entry_final(current_entry, current_section, extracted_rows)
                                    new_entry = {'bank': current_entry['bank'], 'numbers': [number], 'processed': False}
                                    current_entry = new_entry
                                else:
                                    if len(num_list) < 4: current_entry['numbers'].append(number)
                            except ValueError: pass
                        last_line_was_id = False; potential_bank_continuation_candidate = False; continue
                    elif is_id_line:
                        last_line_was_id = True; potential_bank_continuation_candidate = False; continue
                    elif is_noise_line:
                        last_line_was_id = False; potential_bank_continuation_candidate = False; continue
                    else:
                         cleaned_line_for_kw_check = re.sub(r'\s*XX-[\w\d\-]+.*', '', line).strip()
                         cleaned_line_for_kw_check = re.sub(r'\d+$', '', cleaned_line_for_kw_check).strip()
                         contains_keyword = any(kw in cleaned_line_for_kw_check for kw in BANK_KEYWORDS)
                         is_potential_bank = contains_keyword or len(cleaned_line_for_kw_check) > 6
                         common_continuations = ["לישראל", "בע\"מ", "ומשכנתאות", "נדל\"ן", "דיסקונט", "הראשון", "פיננסים"]
                         is_continuation = (potential_bank_continuation_candidate and current_entry and
                                            not current_entry.get('numbers') and
                                            any(cleaned_line_for_kw_check.startswith(cont) for cont in common_continuations))
                         if is_continuation:
                             appendix = cleaned_line_for_kw_check
                             if appendix:
                                 current_entry['bank'] += " " + appendix
                                 current_entry['bank'] = current_entry['bank'].replace(" בע\"מ בע\"מ", " בע\"מ")
                             potential_bank_continuation_candidate = True
                         elif is_potential_bank:
                             if current_entry and not current_entry.get('processed', False):
                                 process_entry_final(current_entry, current_section, extracted_rows)
                             current_entry = {'bank': line, 'numbers': [], 'processed': False}
                             potential_bank_continuation_candidate = True
                         else:
                             potential_bank_continuation_candidate = False
                         last_line_was_id = False
        if current_entry and not current_entry.get('processed', False):
            process_entry_final(current_entry, current_section, extracted_rows)
        doc.close() # Added doc.close()
    except Exception as e: st.error(f"שגיאה קריטית בעיבוד דוח אשראי: {e}"); traceback.print_exc(); return pd.DataFrame()
    if not extracted_rows: st.warning("--- לא חולצו שורות מדוח האשראי. ---"); return pd.DataFrame()
    df = pd.DataFrame(extracted_rows)
    final_cols = ["סוג עסקה", "שם בנק/מקור", "גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]
    for col in final_cols:
        if col not in df.columns: df[col] = np.nan
    df = df[final_cols]
    st.write("--- DataFrame סופי נוצר בהצלחה (דוח אשראי) ---")
    return df

def calculate_summary_credit_report(df_extracted, estimated_monthly_income=12000): # Renamed
    if df_extracted.empty: st.warning("לא ניתן לחשב סיכום: DataFrame ריק."); return pd.DataFrame(columns=["פרמטר", "ערך"])
    st.write("--- מחשב סיכום פיננסי (דוח אשראי)... ---")
    for col in ["גובה מסגרת", "סכום מקורי", "יתרת חוב", "יתרה שלא שולמה"]:
         if col in df_extracted.columns:
              df_extracted[col] = pd.to_numeric(df_extracted[col], errors='coerce')
    loan_mortgage_original = df_extracted.loc[df_extracted['סוג עסקה'].isin(['הלוואה', 'משכנתה']), 'סכום מקורי'].sum(skipna=True)
    total_limit = df_extracted["גובה מסגרת"].sum(skipna=True)
    total_outstanding = df_extracted["יתרת חוב"].sum(skipna=True)
    total_unpaid = df_extracted["יתרה שלא שולמה"].sum(skipna=True)
    debt_to_income_ratio = (total_outstanding / (estimated_monthly_income * 12)) if estimated_monthly_income > 0 else 0
    summary_dict = {
        "סך סכום מקורי (הלוואות/משכנתאות)": loan_mortgage_original,
        "סך גובה מסגרות (עו\"ש/אשראי)": total_limit,
        "סך יתרות חוב נוכחיות (כל הסוגים)": total_outstanding,
        "סה\"כ חוב שלא שולם בזמן": total_unpaid,
        "הכנסה חודשית (משוערת)": estimated_monthly_income,
        "יחס חוב להכנסה שנתית (משוערת)": f"{debt_to_income_ratio:.2%}" if estimated_monthly_income > 0 else "N/A"
    }
    st.write("--- חישוב הסיכום הושלם (דוח אשראי). ---")
    return pd.DataFrame(list(summary_dict.items()), columns=["פרמטר", "ערך"])
# $$$$$$$$$$$$$$$$$$$$$$$$ END: CREDIT REPORT PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#------------------------------------------------------------------------------------
#-------------------------- BANK STATEMENT HELPER FUNCTIONS -------------------------
#------------------------------------------------------------------------------------
# These functions are shared by Hapoalim, Discount, Leumi parsers
def clean_number_bank(text): # Renamed to avoid conflict if other clean_number exists
    if text is None: return None
    text = str(text).strip()
    text = re.sub(r'[₪,]', '', text)
    if text.startswith('(') and text.endswith(')'): text = '-' + text[1:-1]
    if text.endswith('-'): text = '-' + text[:-1]
    try: return float(text)
    except ValueError: st.warning(f"Could not convert '{text}' to float for bank statement."); return None

def parse_date_bank(date_str): # Renamed
    if date_str is None: return None
    date_str_cleaned = str(date_str).strip() # Ensure it's a string before stripping
    try: return datetime.strptime(date_str_cleaned, '%d/%m/%Y')
    except ValueError:
        try: return datetime.strptime(date_str_cleaned, '%d/%m/%y')
        except ValueError: st.warning(f"Could not parse date for bank statement: {date_str_cleaned}"); return None

def normalize_text_bank(text): # Renamed
    if text is None: return None
    return unicodedata.normalize('NFC', str(text)) # Ensure text is string

# --- KPI Calculation Functions (Shared by all bank parsers) ---
def calculate_kpis_bank(df):
    kpis = {}
    if df.empty or 'Description' not in df.columns or 'Balance' not in df.columns:
        st.warning("DataFrame for KPI calculation is empty or missing required columns.")
        # Return default/empty KPIs
        keys = ['Average_Negative_Balance', 'Loan_Repayments', 'Delinquency_Payments', 'Days_Negative_Balance',
                'Returned_Authorizations', 'Credit_vs_Income', 'Irregular_Deposits', 'Savings_During_Negative',
                'Transaction_Frequency', 'Average_Debit_Transaction_Size', 'Fee_Incidence', 'Income_Stability', 'Overdraft_Frequency']
        return {key: 0 for key in keys}

    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    # Determine Amount column (could be 'Amount', or 'Debit'/'Credit')
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        debit_col_for_calc = df['Amount'].apply(lambda x: x if x < 0 else 0)
        credit_col_for_calc = df['Amount'].apply(lambda x: x if x > 0 else 0)
    elif 'Debit' in df.columns and 'Credit' in df.columns:
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce')
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce')
        debit_col_for_calc = -df['Debit'].fillna(0) # Debit is usually positive, treat as negative flow
        credit_col_for_calc = df['Credit'].fillna(0)
    else:
        st.warning("Missing 'Amount' or 'Debit'/'Credit' columns for KPI calculation.")
        return {key: 0 for key in ['Average_Negative_Balance', 'Loan_Repayments', 'Delinquency_Payments', 'Days_Negative_Balance',
                'Returned_Authorizations', 'Credit_vs_Income', 'Irregular_Deposits', 'Savings_During_Negative',
                'Transaction_Frequency', 'Average_Debit_Transaction_Size', 'Fee_Incidence', 'Income_Stability', 'Overdraft_Frequency']}


    avg_balance = df['Balance'].mean() if not df['Balance'].isna().all() else 0
    kpis['Average_Negative_Balance'] = avg_balance if avg_balance < 0 else 0
    loan_keywords = ['הלוואה', 'קרן', 'ריבית']
    kpis['Loan_Repayments'] = df['Description'].astype(str).str.contains('|'.join(loan_keywords), case=False, na=False).sum()
    delinquency_keywords = ['פיגור', 'סילוק פיגורים']
    kpis['Delinquency_Payments'] = df['Description'].astype(str).str.contains('|'.join(delinquency_keywords), case=False, na=False).sum()
    kpis['Days_Negative_Balance'] = (df['Balance'] < 0).sum() if not df['Balance'].isna().all() else 0
    return_keywords = ['חזרת הוראה', 'דחיית תשלום']
    kpis['Returned_Authorizations'] = df['Description'].astype(str).str.contains('|'.join(return_keywords), case=False, na=False).sum()
    
    credit_card_keywords = ['מאסטרקארד', 'ויזה', 'אמריקן אקספרס', 'ישראכרט', 'כרטיס אשראי'] # Added 'כרטיס אשראי'
    # Sum of negative amounts that match credit card keywords
    credit_charges = abs(debit_col_for_calc[df['Description'].astype(str).str.contains('|'.join(credit_card_keywords), case=False, na=False)].sum())

    income_keywords = ['משכורת', 'הכנסה', 'שכר'] # Added 'שכר'
    # Sum of positive amounts that match income keywords
    income = credit_col_for_calc[df['Description'].astype(str).str.contains('|'.join(income_keywords), case=False, na=False)].sum()
    
    kpis['Credit_vs_Income'] = credit_charges / income if income > 0 else float('inf') if credit_charges > 0 else 0

    large_deposit_threshold = 10000
    kpis['Irregular_Deposits'] = df[(credit_col_for_calc > large_deposit_threshold) &
                                    (~df['Description'].astype(str).str.contains('משכורת', case=False, na=False))].shape[0]
    savings_keywords = ['פיקדון', 'חסכון']
    kpis['Savings_During_Negative'] = df[(df['Balance'] < 0) &
                                         (df['Description'].astype(str).str.contains('|'.join(savings_keywords), case=False, na=False))].shape[0]
    if not df['Date'].isna().all() and pd.api.types.is_datetime64_any_dtype(df['Date']):
        date_min = df['Date'].min()
        date_max = df['Date'].max()
        if pd.notna(date_min) and pd.notna(date_max) and date_max > date_min:
             date_range = (date_max - date_min).days
        else:
             date_range = 0 # or 1 if preferred to avoid division by zero later

        transactions_count = len(df)
        months = max(date_range / 30, 1)
        kpis['Transaction_Frequency'] = transactions_count / months
    else:
        kpis['Transaction_Frequency'] = 0

    # Use absolute values of debit_col_for_calc for average size
    debit_values = abs(debit_col_for_calc[debit_col_for_calc < 0])
    kpis['Average_Debit_Transaction_Size'] = debit_values.mean() if not debit_values.empty else 0
    
    fee_keywords = ['עמלה', 'דמי']
    kpis['Fee_Incidence'] = df['Description'].astype(str).str.contains('|'.join(fee_keywords), case=False, na=False).sum()
    
    income_transactions = df[df['Description'].astype(str).str.contains('|'.join(income_keywords), case=False, na=False)]
    if not income_transactions.empty and not df['Date'].isna().all() and pd.api.types.is_datetime64_any_dtype(df['Date']):
        income_dates = income_transactions['Date'].dt.to_pydatetime()
        if len(income_dates) > 1:
            intervals = [(income_dates[i+1] - income_dates[i]).days for i in range(len(income_dates)-1)]
            kpis['Income_Stability'] = 1 / (pd.Series(intervals).std() + 1)
        else: kpis['Income_Stability'] = 0
    else: kpis['Income_Stability'] = 0
    
    overdraft_threshold = -1000
    kpis['Overdraft_Frequency'] = (df['Balance'] < overdraft_threshold).sum() if not df['Balance'].isna().all() else 0
    return kpis

def calculate_risk_score_bank(kpis):
    score = 0
    if kpis.get('Days_Negative_Balance', 0) > 30: score += 15
    elif kpis.get('Days_Negative_Balance', 0) > 15: score += 10
    elif kpis.get('Days_Negative_Balance', 0) > 5: score += 5
    credit_income_ratio = kpis.get('Credit_vs_Income', 0)
    if credit_income_ratio == float('inf') or credit_income_ratio > 0.6 : score += 15 # Handle inf
    elif credit_income_ratio > 0.4: score += 10
    elif credit_income_ratio > 0.2: score += 5
    if kpis.get('Delinquency_Payments', 0) > 3: score += 15
    elif kpis.get('Delinquency_Payments', 0) > 1: score += 10
    elif kpis.get('Delinquency_Payments', 0) > 0: score += 5
    if kpis.get('Irregular_Deposits', 0) > 5: score += 10
    elif kpis.get('Irregular_Deposits', 0) > 2: score += 7
    elif kpis.get('Irregular_Deposits', 0) > 0: score += 3
    if kpis.get('Savings_During_Negative', 0) > 0: score += 7
    if kpis.get('Average_Negative_Balance', 0) < -10000: score += 10
    if kpis.get('Loan_Repayments', 0) > 3: score += 7
    if kpis.get('Returned_Authorizations', 0) > 2: score += 7
    if kpis.get('Transaction_Frequency', 0) > 50: score += 10
    elif kpis.get('Transaction_Frequency', 0) > 20: score += 5
    if kpis.get('Average_Debit_Transaction_Size', 0) > 5000: score += 10
    elif kpis.get('Average_Debit_Transaction_Size', 0) > 2000: score += 5
    if kpis.get('Fee_Incidence', 0) > 5: score += 10
    elif kpis.get('Fee_Incidence', 0) > 2: score += 5
    if kpis.get('Income_Stability', 0) < 0.05: score += 10
    elif kpis.get('Income_Stability', 0) < 0.1: score += 5
    if kpis.get('Overdraft_Frequency', 0) > 20: score += 15
    elif kpis.get('Overdraft_Frequency', 0) > 10: score += 10
    elif kpis.get('Overdraft_Frequency', 0) > 0: score += 5
    return min(score, 100)

def generate_recommendations_bank(risk_score):
    recommendations = []
    if risk_score >= 80: recommendations.append("⚠️ מצב כלכלי קריטי: מומלץ לפנות לייעוץ פיננסי מיידי ולבחון איחוד הלוואות.")
    elif risk_score >= 60: recommendations.append("⚠️ סיכון גבוה: צמצם הוצאות אשראי ובדוק מקורות הכנסה יציבים.")
    elif risk_score >= 30: recommendations.append("⚠️ סיכון בינוני: שפר ניהול תזרים על ידי תקציב חודשי.")
    else: recommendations.append("✅ מצב יציב: המשך לנהל תקציב באחריות.")
    return recommendations
#------------------------------------------------------------------------------------
#-------------------------- END BANK STATEMENT HELPER FUNCTIONS ---------------------
#------------------------------------------------------------------------------------


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$ START: BANK HAPOALIM PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def extract_transactions_from_pdf_hapoalim(pdf_content_bytes, filename_for_logging):
    transactions = []
    try:
        doc = fitz.open(stream=pdf_content_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"שגיאה בפתיחת קובץ PDF של בנק הפועלים {filename_for_logging}: {e}", icon="🚨")
        return transactions

    # More robust patterns
    date_pattern = re.compile(r"(\d{2}/\d{2}/\d{2,4})") # DD/MM/YY or DD/MM/YYYY
    # Amount pattern: matches numbers with commas and decimal points, possibly with minus or shekel sign
    # It tries to capture two numbers for debit/credit and then a balance.
    # This is a common pattern: Date Description Debit Credit Balance
    # Or: Date Description Amount Balance (where amount is negative for debit)
    # Let's try to capture individual numbers first, then assign based on context/columns
    number_pattern = re.compile(r"₪?(-?(?:\d{1,3},)*\d{1,3}\.\d{2}-?|-?\d+\.\d{2}-?)") # handles 1,234.56 or 1234.56 or -1234.56 or 1234.56-

    for page_num, page in enumerate(doc, 1):
        lines = page.get_text("text", sort=True).splitlines() # sort=True can help with table order
        st.write(f"מעבד עמוד {page_num} מתוך {doc.page_count} בקובץ הפועלים: {filename_for_logging} ({len(lines)} שורות)")
        
        # Try to identify header to skip it or infer column order (complex, for now simple extraction)
        # Heuristic: Look for lines that are mostly text and then numbers
        for line_idx, line_text in enumerate(lines):
            original_line = line_text
            line_normalized = normalize_text_bank(line_text.strip())
            if not line_normalized or len(line_normalized) < 10: # Skip very short lines
                continue

            # Find all dates and numbers in the line
            dates_found = date_pattern.findall(original_line)
            numbers_found_raw = number_pattern.findall(original_line)
            numbers_found = [clean_number_bank(n) for n in numbers_found_raw if clean_number_bank(n) is not None]

            parsed_date = None
            description = line_normalized
            amount = None
            balance = None
            
            if not dates_found: # If no date, unlikely a transaction line for Hapoalim standard format
                st.text(f" מדלג על שורה ללא תאריך: {original_line[:60]}...")
                continue

            # Assume the first valid date is the transaction date
            parsed_date = parse_date_bank(dates_found[0])
            if not parsed_date:
                st.text(f" מדלג על שורה, תאריך לא תקין: {original_line[:60]}...")
                continue
            
            # Attempt to extract description by removing numbers and dates
            desc_temp = original_line
            for d in dates_found: desc_temp = desc_temp.replace(d, '')
            for n_raw in numbers_found_raw: desc_temp = desc_temp.replace(n_raw, '')
            description = normalize_text_bank(desc_temp.replace("₪", "").strip())
            
            # Logic for amounts and balance (Hapoalim often has Date, Desc, Debit, Credit, Balance)
            # Or Date, Desc, Amount, Balance
            if len(numbers_found) >= 2: # Expecting at least Amount and Balance
                # Common: last number is balance, second to last is amount (or credit if debit is empty)
                balance = numbers_found[-1]
                # If 3 numbers: Debit, Credit, Balance. The "Amount" would be Debit or Credit
                # If 2 numbers: Amount, Balance
                if len(numbers_found) == 3: # Potentially Debit, Credit, Balance
                    # Hapoalim usually lists debit then credit. If one is 0 or missing, it's effectively the other.
                    # This is tricky as one could be 0.
                    # Let's assume the sum of the first two (if one is negative) is the "net amount"
                    # Or, one is debit (negative flow), one is credit (positive flow)
                    # For now, let's take the non-zero one before balance as 'amount'
                    if numbers_found[-2] != 0: # Credit
                        amount = numbers_found[-2]
                    elif numbers_found[-3] != 0: # Debit
                        amount = -abs(numbers_found[-3]) # Assuming debit is positive in their column
                    else: # Both are zero, take one as amount
                         amount = numbers_found[-2]

                elif len(numbers_found) == 2: # Amount, Balance
                    amount = numbers_found[0]
                # Heuristic: if amount is positive and balance decreased, it's a debit.
                # This requires previous balance, which is complex for unsorted lines.
                # For now, rely on the sign of the amount if available, or if it's clearly debit/credit columns.
            
            elif len(numbers_found) == 1 : # Only one number, could be balance or amount
                 # If it's close to previous balance, it's probably balance. Otherwise, amount.
                 # For now, assume it's an amount if description looks valid. This is weak.
                 amount = numbers_found[0] # Or could be balance

            transaction = {
                'Date': parsed_date,
                'Description': description,
                'Amount': amount, # Negative for debit, positive for credit
                'Balance': balance,
                'SourceFile': filename_for_logging,
                '_OriginalLine': original_line[:100] # For debugging
            }
            transactions.append(transaction)
            # st.text(f"HAPOALIM Extracted: D={parsed_date}, A={amount}, B={balance}, Desc={description[:30]}")

    doc.close()
    st.write(f"חולצו {len(transactions)} עסקאות מקובץ הפועלים: {filename_for_logging}")
    return transactions
# $$$$$$$$$$$$$$$$$$$$$$$ END: BANK HAPOALIM PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$ START: BANK DISCOUNT PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def reverse_hebrew_text(text): # Specific to Discount parser potentially
    if not text: return text
    # Check if text contains Hebrew characters
    if not any('\u0590' <= char <= '\u05EA' for char in text):
        return text # Not Hebrew, return as is
    
    words = text.split()
    # Reverse individual words only if they are primarily Hebrew letters
    # This avoids reversing numbers or mixed alphanumeric words unnecessarily
    reversed_words = []
    for word in words:
        if len(word) > 1 and sum(1 for char in word if '\u0590' <= char <= '\u05EA') / len(word) > 0.5: # More than 50% Hebrew
             reversed_words.append(word[::-1])
        else:
             reversed_words.append(word)
    return ' '.join(reversed_words[::-1]) # Reverse order of words

def parse_discont_transaction_line(line_text):
    line = normalize_text_bank(line_text.strip())
    if not line: return None

    # Discount format: Date ValueDate Description Ref Debit Credit Balance (order varies slightly)
    # Regex to find dates and numbers
    date_pattern_strict = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})") # Two DD/MM/YYYY
    date_pattern_flexible = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")
    number_pattern = re.compile(r"₪?(-?(?:\d{1,3},)*\d{1,3}\.\d{2}-?|-?\d+\.\d{2}-?)")

    dates_found_strict = date_pattern_strict.search(line)
    parsed_date, parsed_value_date = None, None

    if dates_found_strict:
        parsed_date = parse_date_bank(dates_found_strict.group(1))
        parsed_value_date = parse_date_bank(dates_found_strict.group(2))
        # Remove dates from line to isolate other parts
        line_minus_dates = date_pattern_strict.sub('', line).strip()
    else:
        dates_found_flex = date_pattern_flexible.findall(line)
        if len(dates_found_flex) >= 1:
            parsed_date = parse_date_bank(dates_found_flex[0]) # Assuming first date is primary
            if len(dates_found_flex) >= 2:
                 parsed_value_date = parse_date_bank(dates_found_flex[1]) # Assuming second is value date
            line_minus_dates = line
            for d_str in dates_found_flex: line_minus_dates = line_minus_dates.replace(d_str, '')
            line_minus_dates = line_minus_dates.strip()
        else: # No date found, skip line
            return None
    
    if not parsed_date: return None # Must have at least one valid date

    numbers_raw = number_pattern.findall(line_minus_dates)
    numbers = [clean_number_bank(n) for n in numbers_raw if clean_number_bank(n) is not None]

    description = line_minus_dates
    for n_raw in numbers_raw: description = description.replace(n_raw, '')
    description = description.replace("₪", "").strip()
    description = reverse_hebrew_text(description) # Discount often needs this
    
    # Typical order: Ref? Debit Credit Balance (3 or 4 numbers usually at the end)
    debit, credit, balance, ref = None, None, None, None # Initialize

    if len(numbers) >= 1: balance = numbers[-1] # Last number is usually balance
    if len(numbers) >= 2: # If two numbers, it's usually Amount, Balance or Credit, Balance
        # If previous was Debit, this is Credit
        # This is tricky. If amount column exists, it's simpler.
        # Let's assume second to last is "amount" if only two.
        # It could be credit if positive, debit if negative.
        # This needs to be resolved with actual statement examples.
        # For now, let's assume that if three numbers: Debit, Credit, Balance
        amount_candidate = numbers[-2] if len(numbers) >=2 else None
        if amount_candidate is not None:
            if balance is not None and parsed_date is not None: # If we have a balance and date
                # Heuristic: if balance decreases, it's a debit, increases it's credit
                # This requires previous balance.
                # For now, if there are 3 numbers like D, C, B:
                if len(numbers) >= 3:
                    credit_val = numbers[-2]
                    debit_val = numbers[-3]
                    if credit_val > 0: credit = credit_val
                    if debit_val > 0 : debit = debit_val # Debit is positive in its column
                elif len(numbers) == 2: # Amount, Balance. How to tell if debit/credit?
                    # If numbers[-2] (the amount) is positive, assume credit. If negative, debit.
                    # But statements often show debit as positive in its own column.
                    # This is the hardest part without clear column separation.
                    # A common Discount pattern is: Date ValueDate Description Ref Debit Credit Balance
                    # If only 2 numbers, it might be a summary line or simpler transaction list.
                    # Let's assume if only two numbers found, numbers[0] is Amount, numbers[1] is Balance.
                    # If amount_candidate > 0, credit = amount_candidate. Else debit = abs(amount_candidate).
                    # This is a guess.
                    if numbers[0] > 0: credit = numbers[0]
                    else: debit = abs(numbers[0])


    # Reference is often an alphanumeric string before numbers/description
    # This is very heuristic
    potential_ref_match = re.match(r"(\d+[-\d\w]*/?\d*)", description)
    if potential_ref_match:
        ref = potential_ref_match.group(1)
        description = description.replace(ref, "").strip()

    transaction = {
        'Date': parsed_date, 'ValueDate': parsed_value_date, 'Description': description,
        'Reference': ref, 'Debit': debit, 'Credit': credit, 'Balance': balance,
        '_OriginalLine': line_text[:100]
    }
    # st.text(f"DISCOUNT Extracted: D={parsed_date}, Ref={ref} Dr={debit}, Cr={credit}, B={balance}, Desc={description[:30]}")
    return transaction

def extract_and_parse_discont_pdf(pdf_content_bytes, filename_for_logging):
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes))as pdf:
            st.write(f"מעבד קובץ דיסקונט: {filename_for_logging} ({len(pdf.pages)} עמודים)")
            for page_number, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=1, y_tolerance=1) # Tighter tolerance might help
                    if text:
                        lines = text.splitlines()
                        for line_idx, line_text in enumerate(lines):
                            # Skip common headers/footers for Discount Bank
                            if "תאריך ערך" in line_text and "אסמכתא" in line_text and "פרטים" in line_text: continue
                            if "סך חיובים" in line_text or "סך זכויים" in line_text or "יתרת פתיחה" in line_text or "יתרת סגירה" in line_text : continue
                            if line_text.strip().startswith("עמוד") or line_text.strip().startswith("Page"): continue
                            
                            parsed_transaction = parse_discont_transaction_line(line_text)
                            if parsed_transaction:
                                transactions.append(parsed_transaction)
                except Exception as e:
                    st.warning(f"שגיאה בעיבוד עמוד {page_number + 1} בקובץ דיסקונט {filename_for_logging}: {e}")
    except Exception as e:
        st.error(f"שגיאה בפתיחת או עיבוד קובץ PDF של בנק דיסקונט {filename_for_logging}: {e}", icon="🚨")
        return []
    st.write(f"חולצו {len(transactions)} עסקאות מקובץ דיסקונט: {filename_for_logging}")
    return transactions
# $$$$$$$$$$$$$$$$$$$$$$$ END: BANK DISCOUNT PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$ START: BANK LEUMI PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def clean_transaction_amount_leumi(text):
    if text is None or pd.isna(text) or text == '': return None
    text = str(text).strip().replace('₪', '').replace(',', '')
    if '.' not in text: return None # Expect decimal for amounts
    text = text.lstrip('\u200b') # Remove zero-width space
    try:
        # Handle cases like "1.234.56" which should be "1234.56"
        if text.count('.') > 1:
            parts = text.split('.')
            text = parts[0] + ''.join(parts[1:-1]) + '.' + parts[-1] if len(parts) > 2 else parts[0] + '.' + parts[1]

        val = float(text)
        # Heuristic: very large single transaction amounts might be parsing errors or balances
        if abs(val) > 1_000_000: return None # Adjust threshold as needed
        return val
    except ValueError: return None

def parse_leumi_transaction_line_extracted_order_v2(line_text, previous_balance_val):
    line = unicodedata.normalize('NFC', line_text.strip())
    if not line: return None

    # Leumi specific patterns. Order can be: Date ValueDate Description Ref +/-Amount Balance
    # Or: Date ValueDate Description Ref Debit Credit Balance
    # Regex to find all dates and numbers
    date_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")
    # Number pattern: allows for positive/negative, commas, and shekel sign.
    # \u200b is a zero-width space that sometimes appears.
    number_pattern = re.compile(r"[\u200b₪]?\s*(-?(?:\d{1,3},)*\d{1,3}\.\d{2}-?|-?\d+\.\d{2}-?)")

    dates_found = date_pattern.findall(line)
    numbers_raw = number_pattern.findall(line) # Find all potential numbers
    numbers = [clean_number_bank(n) for n in numbers_raw if clean_number_bank(n) is not None]

    parsed_date, parsed_value_date = None, None
    description, reference = "", None
    debit, credit, current_balance = None, None, None

    if not dates_found or len(dates_found) < 1 : return None # Must have at least one date
    
    parsed_date = parse_date_bank(dates_found[0])
    if len(dates_found) > 1: parsed_value_date = parse_date_bank(dates_found[1])
    if not parsed_date: return None

    # Attempt to isolate description by removing dates and numbers
    desc_temp = line
    for d_str in dates_found: desc_temp = desc_temp.replace(d_str, '', 1) # Replace only first occurrence
    for num_raw in numbers_raw: desc_temp = desc_temp.replace(num_raw, '', 1)
    desc_temp = desc_temp.replace("₪","").strip()
    
    # Leumi description is often reversed
    description = normalize_text_bank(desc_temp) # normalize_text_bank already reverses Hebrew

    # Logic for Debit, Credit, Balance (often 3 numbers at the end for Leumi)
    # Or Amount, Balance (2 numbers)
    if len(numbers) >= 1: current_balance = numbers[-1] # Balance is usually the last number
    
    amount_determined = False
    if len(numbers) >= 3: # Potentially Debit, Credit, Balance
        # Assume numbers[-3] is Debit, numbers[-2] is Credit if they are positive
        potential_debit = numbers[-3]
        potential_credit = numbers[-2]
        if potential_debit > 0: debit = potential_debit
        if potential_credit > 0: credit = potential_credit
        amount_determined = True if debit is not None or credit is not None else False

    if not amount_determined and len(numbers) >= 2: # Potentially Amount, Balance
        # This is the "amount" that affected the balance
        transaction_amount_raw = numbers[-2] 
        transaction_amount_cleaned = clean_transaction_amount_leumi(str(transaction_amount_raw)) # Use stricter cleaning for amount

        if transaction_amount_cleaned is not None:
            if previous_balance_val is not None and current_balance is not None:
                balance_diff = current_balance - previous_balance_val
                # Check if transaction_amount_cleaned matches balance_diff (with tolerance)
                if abs(balance_diff - transaction_amount_cleaned) < 0.02: # Credit
                    credit = transaction_amount_cleaned
                elif abs(balance_diff + transaction_amount_cleaned) < 0.02: # Debit
                    debit = transaction_amount_cleaned
                # If signs are mixed, e.g. amount is positive but balance decreased
                elif transaction_amount_cleaned > 0 and balance_diff < -0.01:
                    debit = transaction_amount_cleaned
                elif transaction_amount_cleaned < 0 and balance_diff > 0.01: # Amount is negative, balance increased (unusual)
                    credit = abs(transaction_amount_cleaned) # Store credit as positive
                elif transaction_amount_cleaned != 0 : # Fallback: if amount is non-zero
                    if transaction_amount_cleaned > 0: credit = transaction_amount_cleaned
                    else: debit = abs(transaction_amount_cleaned)
            elif transaction_amount_cleaned !=0: # No previous balance, infer from sign
                 if transaction_amount_cleaned > 0: credit = transaction_amount_cleaned
                 else: debit = abs(transaction_amount_cleaned)
            amount_determined = True

    # Reference number is often a series of digits, sometimes with slashes or hyphens
    # It might be part of the description or a separate column.
    # This is very heuristic. Try to find it in the remaining description.
    ref_match = re.search(r"(\d{4,}[-/]?\d*)", description) # Longer sequence of digits
    if ref_match:
        reference = ref_match.group(1)
        description = description.replace(reference, "").strip()

    if debit is None and credit is None and previous_balance_val is not None and current_balance is not None:
        # If no debit/credit identified explicitly, calculate from balance change
        balance_diff = current_balance - previous_balance_val
        if abs(balance_diff) > 0.01: # Only if there's a significant change
            if balance_diff > 0: credit = balance_diff
            else: debit = abs(balance_diff)

    # Ensure debit/credit are not NaN if they are 0
    debit = 0.0 if debit is None and credit is not None else debit
    credit = 0.0 if credit is None and debit is not None else credit
    
    # Skip if no financial impact
    if debit is None and credit is None: return None
    if debit == 0 and credit == 0: return None


    transaction = {
        'Date': parsed_date, 'ValueDate': parsed_value_date, 'Description': description,
        'Reference': reference, 'Debit': debit, 'Credit': credit, 'Balance': current_balance,
        '_OriginalLine': line_text[:100]
    }
    # st.text(f"LEUMI Extracted: D={parsed_date}, Dr={debit}, Cr={credit}, B={current_balance}, Desc={description[:30]}")
    return transaction

def extract_leumi_transactions_line_by_line(pdf_content_bytes, filename_for_logging):
    transactions = []
    st.write(f"מעבד קובץ לאומי: {filename_for_logging} באמצעות Regex שורה אחר שורה")
    try:
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            previous_balance_val = None
            first_balance_found_on_page = False

            for page_number, page in enumerate(pdf.pages):
                st.write(f"קורא טקסט מעמוד {page_number + 1} בקובץ לאומי")
                # Layout=True helps preserve original reading order which can be crucial for Leumi
                text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True) 
                if not text: continue
                lines = text.splitlines()
                st.write(f"חולצו {len(lines)} שורות מעמוד {page_number + 1}")
                page_transactions_added = 0
                
                # Try to find an initial balance on the page if previous_balance_val is None
                if previous_balance_val is None:
                    for line_text_init_balance in lines:
                        # Look for "יתרת פתיחה" or similar and a number
                        if "יתרת פתיחה" in line_text_init_balance or "יתרה קודמת" in line_text_init_balance:
                            match_balance = re.search(r"(-?(?:\d{1,3},)*\d{1,3}\.\d{2}-?|-?\d+\.\d{2}-?)", line_text_init_balance)
                            if match_balance:
                                previous_balance_val = clean_number_bank(match_balance.group(1))
                                st.write(f"יתרת פתיחה זוהתה: {previous_balance_val}")
                                break
                
                for line_num, line_text in enumerate(lines):
                    cleaned_line = line_text.strip()
                    if not cleaned_line or len(cleaned_line) < 10: continue # Skip empty or very short lines
                    # Skip known header/footer lines for Leumi
                    if "מספר חשבון" in cleaned_line or "תאריך הפקה" in cleaned_line or "בנק לאומי לישראל בע\"מ" in cleaned_line : continue
                    if cleaned_line.startswith("עמוד ") or cleaned_line.lower().startswith("page "): continue
                    if "סיכום ביניים" in cleaned_line or "יתרת פתיחה" in cleaned_line or "יתרה קודמת" in cleaned_line or "יתרת סגירה" in cleaned_line : continue


                    parsed_transaction_data = parse_leumi_transaction_line_extracted_order_v2(cleaned_line, previous_balance_val)

                    if parsed_transaction_data:
                        current_balance_from_line = parsed_transaction_data['Balance']
                        
                        # If this is the first transaction being processed and we don't have a previous balance,
                        # use the balance from this line as the starting point for the *next* transaction's calculation.
                        # Don't add this transaction itself unless it has debit/credit.
                        if previous_balance_val is None and current_balance_from_line is not None:
                            previous_balance_val = current_balance_from_line
                            # st.text(f"   Leumi Line {line_num+1}: Set initial prev_balance to {previous_balance_val}. Desc: {parsed_transaction_data['Description'][:20]}")
                            # Only add if it has financial impact determined by parser already
                            if parsed_transaction_data['Debit'] is not None or parsed_transaction_data['Credit'] is not None:
                                transactions.append(parsed_transaction_data)
                                page_transactions_added +=1
                            continue # Move to next line
                        
                        # Add transaction if it has financial impact (debit/credit)
                        if parsed_transaction_data['Debit'] is not None or parsed_transaction_data['Credit'] is not None:
                            transactions.append(parsed_transaction_data)
                            if current_balance_from_line is not None: # Update previous_balance only if current line has a balance
                                previous_balance_val = current_balance_from_line
                            page_transactions_added += 1
                            # st.text(f"   Leumi Line {line_num+1}: Transaction added. PrevBal updated to {previous_balance_val}. Desc: {parsed_transaction_data['Description'][:20]}")
                        elif current_balance_from_line is not None: # No debit/credit, but has balance, update previous_balance
                            previous_balance_val = current_balance_from_line
                            # st.text(f"   Leumi Line {line_num+1}: No D/C. PrevBal updated to {previous_balance_val}. Desc: {parsed_transaction_data['Description'][:20]}")
                st.write(f"נוספו {page_transactions_added} עסקאות מעמוד {page_number + 1} (לאומי)")
    except Exception as e:
        st.error(f"שגיאה בעיבוד קובץ PDF של בנק לאומי {filename_for_logging}: {e}", icon="🚨")
        traceback.print_exc()
        return []
    st.write(f"עיבוד קובץ לאומי {filename_for_logging} הסתיים. סה\"כ עסקאות: {len(transactions)}")
    return transactions

# $$$$$$$$$$$$$$$$$$$$$$$ END: BANK LEUMI PARSER CODE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# --- Function to get response from GPT ---
def get_gpt_response(prompt_text, context_data_str=""):
    if not openai.api_key:
        st.error("מפתח OpenAI API אינו מוגדר.")
        return "שגיאה: מפתח API חסר."
    
    full_prompt = "אתה עוזר וירטואלי מומחה לכלכלת המשפחה בישראל, ואתה מספק תשובות בעברית.\n"
    if context_data_str:
        full_prompt += f"להלן נתונים רלוונטיים מהמשתמש:\n{context_data_str}\n\n"
    full_prompt += f"שאלה מהמשתמש: {prompt_text}\n\nאנא השב בעברית."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # או "gpt-4" אם יש לך גישה
            messages=[
                {"role": "system", "content": "אתה עוזר וירטואלי מומחה לכלכלת המשפחה בישראל. ענה תמיד בעברית."},
                {"role": "user", "content": full_prompt} # Pass the combined prompt with context
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"שגיאה בפנייה ל-OpenAI: {e}")
        return "מצטער, התרחשה שגיאה בעת יצירת התשובה."

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(page_title="עוזר פיננסי למשפחה", layout="wide", initial_sidebar_state="expanded")
    st.title("עוזר פיננסי אישי מבוסס AI 💰")
    st.markdown("העלה קבצים, קבל ניתוח ושאל שאלות על כלכלת המשפחה שלך.")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "שלום! איך אני יכול לעזור לך היום עם כלכלת המשפחה?"}]
    if "processed_data_string" not in st.session_state:
        st.session_state.processed_data_string = ""
    if "credit_report_df" not in st.session_state:
        st.session_state.credit_report_df = None
    if "credit_summary_df" not in st.session_state:
        st.session_state.credit_summary_df = None
    if "bank_statement_df" not in st.session_state:
        st.session_state.bank_statement_df = None
    if "bank_kpis" not in st.session_state:
        st.session_state.bank_kpis = None
    if "bank_risk_score" not in st.session_state:
        st.session_state.bank_risk_score = None
    if "bank_recommendations" not in st.session_state:
        st.session_state.bank_recommendations = None


    with st.sidebar:
        st.header("העלאת ועיבוד קבצים")
        uploaded_file = st.file_uploader("העלה קובץ PDF:", type="pdf", key="file_uploader")
        
        file_type_options = {
            'ללא קובץ (שיחה כללית)': 'general',
            'דוח נתוני אשראי': 'credit_report',
            'תדפיס חשבון בנק הפועלים': 'hapoalim',
            'תדפיס חשבון בנק דיסקונט': 'discount',
            'תדפיס חשבון בנק לאומי': 'leumi'
        }
        selected_file_type_display = st.selectbox(
            "בחר את סוג הקובץ:",
            options=list(file_type_options.keys()),
            key="file_type_selector"
        )
        file_type = file_type_options[selected_file_type_display]

        process_button = st.button("עבד קובץ", key="process_button", disabled=(uploaded_file is None or file_type == 'general'))

        estimated_income = 12000 # Default
        if file_type == 'credit_report':
            estimated_income = st.number_input(
                "הכנסה חודשית משוערת (לחישוב יחס חוב להכנסה בדוח אשראי):", 
                min_value=0, value=12000, step=500, key="income_input_credit"
            )

    if process_button and uploaded_file is not None:
        st.session_state.processed_data_string = "" # Reset context
        # Clear previous bank/credit specific data
        st.session_state.credit_report_df = None
        st.session_state.credit_summary_df = None
        st.session_state.bank_statement_df = None
        st.session_state.bank_kpis = None
        st.session_state.bank_risk_score = None
        st.session_state.bank_recommendations = None

        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        
        with st.spinner(f"מעבד את הקובץ '{filename}' כסוג '{selected_file_type_display}'..."):
            if file_type == 'credit_report':
                df = extract_credit_data_final_v13(file_bytes, filename)
                st.session_state.credit_report_df = df
                if not df.empty:
                    summary_df = calculate_summary_credit_report(df.copy(), estimated_income)
                    st.session_state.credit_summary_df = summary_df
                    st.session_state.processed_data_string = f"סיכום דוח אשראי:\n{summary_df.to_string()}\n\nפרטי דוח אשראי:\n{df.to_string()}"
                    st.success(f"דוח אשראי '{filename}' עובד בהצלחה!")
                else:
                    st.warning("לא חולצו נתונים מדוח האשראי.")

            elif file_type in ['hapoalim', 'discount', 'leumi']:
                bank_df = pd.DataFrame()
                if file_type == 'hapoalim':
                    transactions_list = extract_transactions_from_pdf_hapoalim(file_bytes, filename)
                    if transactions_list: bank_df = pd.DataFrame(transactions_list)
                elif file_type == 'discount':
                    transactions_list = extract_and_parse_discont_pdf(file_bytes, filename)
                    if transactions_list: bank_df = pd.DataFrame(transactions_list)
                elif file_type == 'leumi':
                    transactions_list = extract_leumi_transactions_line_by_line(file_bytes, filename)
                    if transactions_list: bank_df = pd.DataFrame(transactions_list)

                if not bank_df.empty:
                    bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')
                    # Ensure numeric types for relevant columns before KPIs
                    for col in ['Amount', 'Debit', 'Credit', 'Balance']:
                        if col in bank_df.columns:
                            bank_df[col] = pd.to_numeric(bank_df[col], errors='coerce')
                    
                    bank_df = bank_df.sort_values(by='Date').reset_index(drop=True)
                    st.session_state.bank_statement_df = bank_df
                    
                    kpis = calculate_kpis_bank(bank_df.copy())
                    risk_score = calculate_risk_score_bank(kpis)
                    recommendations = generate_recommendations_bank(risk_score)
                    
                    st.session_state.bank_kpis = kpis
                    st.session_state.bank_risk_score = risk_score
                    st.session_state.bank_recommendations = recommendations
                    
                    # Prepare context string from bank data
                    kpis_str = "\n".join([f"{key.replace('_', ' ').capitalize()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in kpis.items()])
                    recs_str = "\n".join(recommendations)
                    st.session_state.processed_data_string = (
                        f"סיכום תדפיס בנק ({selected_file_type_display}):\n"
                        f"מדדי ביצוע עיקריים (KPIs):\n{kpis_str}\n\n"
                        f"ציון סיכון: {risk_score}\n\n"
                        f"המלצות:\n{recs_str}\n\n"
                        f"קטע מתדפיס הבנק (5 שורות ראשונות לדוגמה):\n{bank_df.head().to_string()}"
                    )
                    st.success(f"תדפיס בנק '{filename}' עובד בהצלחה!")
                else:
                    st.warning(f"לא חולצו נתונים מתדפיס הבנק ({selected_file_type_display}).")
            else:
                st.info("סוג קובץ לא נתמך לעיבוד או שלא נבחר קובץ.")

    # Display processed data if available
    if st.session_state.credit_report_df is not None and not st.session_state.credit_report_df.empty:
        st.subheader("טבלת חובות שחולצה (דוח אשראי)")
        st.dataframe(st.session_state.credit_report_df.style.format({
            "גובה מסגרת": '{:,.0f}', "סכום מקורי": '{:,.0f}',
            "יתרת חוב": '{:,.0f}', "יתרה שלא שולמה": '{:,.0f}'
        }))
    if st.session_state.credit_summary_df is not None and not st.session_state.credit_summary_df.empty:
        st.subheader("סיכום פיננסי (דוח אשראי)")
        st.table(st.session_state.credit_summary_df.style.format({"ערך": lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and x > 1000 and not isinstance(x, str) else x}))

    if st.session_state.bank_statement_df is not None and not st.session_state.bank_statement_df.empty:
        st.subheader(f"תנועות בחשבון ({selected_file_type_display if 'selected_file_type_display' in locals() else ''})")
        st.dataframe(st.session_state.bank_statement_df.head()) # Show a sample

        if st.session_state.bank_kpis:
            st.subheader("מדדי ביצוע מרכזיים (KPIs)")
            kpis_df = pd.DataFrame(list(st.session_state.bank_kpis.items()), columns=['מדד', 'ערך'])
            st.table(kpis_df)
        
        if st.session_state.bank_risk_score is not None:
            st.subheader("ציון סיכון פיננסי")
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number", value=st.session_state.bank_risk_score,
                title={'text': "ציון סיכון פיננסי", 'font': {'size': 20}},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 30], 'color': "lightgreen"},
                                 {'range': [30, 60], 'color': "yellow"},
                                 {'range': [60, 100], 'color': "red"}]}))
            st.plotly_chart(fig_risk, use_container_width=True)

        if st.session_state.bank_recommendations:
            st.subheader("המלצות")
            for rec in st.session_state.bank_recommendations:
                st.markdown(f"- {rec}")

        # Plot Balance Trend for Bank statements
        df_plot = st.session_state.bank_statement_df.dropna(subset=['Date', 'Balance']).copy()
        if not df_plot.empty and pd.api.types.is_datetime64_any_dtype(df_plot['Date']) and pd.api.types.is_numeric_dtype(df_plot['Balance']):
            df_plot = df_plot.sort_values(by='Date')
            st.subheader("מגמת יתרת חשבון")
            fig_balance = px.line(df_plot, x='Date', y='Balance', title='יתרה לאורך זמן', markers=True)
            fig_balance.update_layout(xaxis_title='תאריך', yaxis_title='יתרה (ש"ח)')
            st.plotly_chart(fig_balance, use_container_width=True)


    # Chat interface
    st.subheader("צ'אט לייעוץ פיננסי")
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("שאל שאלה... (לחץ Enter לשליחה)"):
        if not openai.api_key: # Redundant check, but good practice
            st.error("מפתח OpenAI API אינו מוגדר בצד השרת.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("חושב..."):
            response = get_gpt_response(prompt, st.session_state.processed_data_string)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()