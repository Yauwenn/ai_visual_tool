import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import numpy as np
from sklearn.metrics import r2_score
from cycler import cycler
from scipy.stats import pearsonr

# --- Language and Theme Management ---
# Define dictionaries for English and Traditional Mandarin
translations = {
    "en": {
        "title": "Interactive MOF Data Dashboard",
        "upload_prompt": "Upload a CSV file to begin.",
        "upload_button": "Choose a CSV file",
        "filter_header": "Filter Data",
        "filter_prompt_1": "Select a column to filter by",
        "filter_multiselect": "Select values to include",
        "plot_header": "Plot & Statistics",
        "select_view": "Select View",
        "overall_analysis_title": "Overall Data Analysis",
        "overall_analysis_intro": "This section provides a comprehensive data analysis report based on your research goals.",
        "analysis_report_header": "AI Analysis Report",
        "desorption_analysis_header": "1. Desorption Energy Analysis",
        "adsorption_analysis_header": "2. Adsorption Capacity Analysis",
        "relationship_analysis_header": "3. Relationship Analysis",
        "get_ai_analysis": "Get AI Analysis",
        "scatter_plot_title": "Scatter Plot",
        "scatter_plot_intro": "Select two variables and a coloring category to create a scatter plot.",
        "scatter_x_axis": "Select a variable for the X-Axis",
        "scatter_y_axis": "Select a variable for the Y-Axis",
        "scatter_color_by": "Color by",
        "scatter_trend_line": "Add Trend Line",
        "scatter_trend_none": "None",
        "scatter_trend_linear": "Linear (Regression Line)",
        "scatter_trend_poly": "Polynomial Trend Line",
        "bar_chart_title": "Bar Chart",
        "bar_chart_intro": "Select a numerical and a categorical variable to create a bar chart.",
        "bar_numerical_var": "Select a Numerical Variable",
        "bar_categorical_var": "Select a Categorical Variable",
        "line_graph_title": "Line Graph",
        "line_graph_intro": "Select two variables to compare over a common X-axis.",
        "line_x_axis": "Select a common X-Axis",
        "line_y1_axis": "Select the 1st Y-Axis variable",
        "line_y2_axis": "Select the 2nd Y-Axis variable",
        "stats_title": "Summary Statistics",
        "stats_intro": "Select a variable to view its summary statistics.",
        "stats_var": "Select a Numerical Variable",
        "error_csv": "An error occurred while reading the file. Please ensure it is a valid CSV file.",
        "no_file_uploaded": "Please upload a CSV file to begin the analysis.",
        "no_filter_selected": "Please select at least one value to display data."
    },
    "zh": {
        "title": "MOF數據互動式儀表板",
        "upload_prompt": "請上傳一個 CSV 文件以開始。",
        "upload_button": "選擇一個 CSV 文件",
        "filter_header": "數據篩選",
        "filter_prompt_1": "選擇一個要篩選的列",
        "filter_multiselect": "選擇要包含的值",
        "plot_header": "圖表與統計",
        "select_view": "選擇視圖",
        "overall_analysis_title": "整體數據分析",
        "overall_analysis_intro": "本節提供基於您研究目標的綜合數據分析報告。",
        "analysis_report_header": "AI 分析報告",
        "desorption_analysis_header": "1. 脫附能量分析",
        "adsorption_analysis_header": "2. 吸附容量分析",
        "relationship_analysis_header": "3. 關係分析",
        "get_ai_analysis": "獲取 AI 分析",
        "scatter_plot_title": "散點圖",
        "scatter_plot_intro": "選擇兩個變量和一個分類來創建散點圖。",
        "scatter_x_axis": "選擇 X 軸變量",
        "scatter_y_axis": "選擇 Y 軸變量",
        "scatter_color_by": "按顏色分類",
        "scatter_trend_line": "添加趨勢線",
        "scatter_trend_none": "無",
        "scatter_trend_linear": "線性（回歸線）",
        "scatter_trend_poly": "多項式趨勢線",
        "bar_chart_title": "柱狀圖",
        "bar_chart_intro": "選擇一個數值和一個分類變量來創建柱狀圖。",
        "bar_numerical_var": "選擇一個數值變量",
        "bar_categorical_var": "選擇一個分類變量",
        "line_graph_title": "折線圖",
        "line_graph_intro": "選擇兩個變量以在共同的 X 軸上進行比較。",
        "line_x_axis": "選擇一個共同的 X 軸",
        "line_y1_axis": "選擇第一個 Y 軸變量",
        "line_y2_axis": "選擇第二個 Y 軸變量",
        "stats_title": "摘要統計",
        "stats_intro": "選擇一個變量來查看其摘要統計。",
        "stats_var": "選擇一個數值變量",
        "error_csv": "讀取文件時發生錯誤。請確保它是有效的 CSV 文件。",
        "no_file_uploaded": "請上傳 CSV 文件以開始分析。",
        "no_filter_selected": "請至少選擇一個值以顯示數據。"
    }
}

# --- Session State Management for all states ---
# Language and Theme states
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "is_dark_mode" not in st.session_state:
    st.session_state.is_dark_mode = False

# Analysis and Plotting states
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = ""
if "plot_type" not in st.session_state:
    st.session_state.plot_type = 'Overall Data Analysis'
    
# Plot-specific variable selections
if "scatter_x_axis" not in st.session_state:
    st.session_state.scatter_x_axis = None
if "scatter_y_axis" not in st.session_state:
    st.session_state.scatter_y_axis = None
if "scatter_color_by" not in st.session_state:
    st.session_state.scatter_color_by = 'None'
if "scatter_trend_line" not in st.session_state:
    st.session_state.scatter_trend_line = 'None'

if "bar_numerical_var" not in st.session_state:
    st.session_state.bar_numerical_var = None
if "bar_categorical_var" not in st.session_state:
    st.session_state.bar_categorical_var = None

if "line_x_axis" not in st.session_state:
    st.session_state.line_x_axis = None
if "line_y1_axis" not in st.session_state:
    st.session_state.line_y1_axis = None
if "line_y2_axis" not in st.session_state:
    st.session_state.line_y2_axis = None

if "stats_var" not in st.session_state:
    st.session_state.stats_var = None
if "filter_column" not in st.session_state:
    st.session_state.filter_column = 'None'
if "selected_values" not in st.session_state:
    st.session_state.selected_values = []


# Function to toggle language
def toggle_language():
    st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"

# Function to toggle dark mode
def toggle_dark_mode():
    st.session_state.is_dark_mode = not st.session_state.is_dark_mode
    st.config.set_option("theme.base", "dark" if st.session_state.is_dark_mode else "light")

# Helper function to find the correct index for a selectbox
def get_index(options, value):
    try:
        if value is None:
            return 0
        return options.index(value)
    except ValueError:
        return 0

# Get the current translations
t = translations[st.session_state.lang]

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- NEW: Function to translate text using Gemini ---
def translate_text(text, target_lang):
    try:
        if target_lang == "zh":
            prompt = f"Translate the following English text to Traditional Mandarin. Provide only the translated text, do not add any extra commentary or greetings:\n\n{text}"
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        else:
            return text
    except Exception as e:
        return f"⚠️ Translation failed: {e}"

# --- AI Analysis Functions (Unchanged from previous versions) ---
def analyze_overall_data(df):
    try:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        desorption_energy_by_mof = df.groupby('mof_type')['desorption_energy_kj_mol_exp'].mean().sort_values(ascending=True)
        adsorption_capacity_by_mof = df.groupby('mof_type')['specific_adsorption_mmol_g'].mean().sort_values(ascending=False)
        
        prompt = f"""
        You are an expert data analyst specializing in materials science and Direct Air Capture (DAC) technology. You have been provided with a dataset on novel Metal-Organic Frameworks (MOFs).
        The dataset has the following columns:
        Numerical columns: {', '.join(numerical_cols)}
        Categorical columns: {', '.join(categorical_cols)}
        Your primary research goals are:
        1. **Minimize Desorption Energy:** Find a MOF with a lower desorption energy (measured in kJ/mol-CO₂, column 'desorption_energy_kj_mol_exp') than traditional adsorbents. A lower value indicates better energy efficiency.
        2. **Maximize Adsorption Capacity:** Identify MOFs with a high CO₂ adsorption capacity (measured in mmol-CO₂/g, column 'specific_adsorption_mmol_g').
        Here is a summary of the key findings from the dataset:
        **Average Desorption Energy by MOF Type:**
        {desorption_energy_by_mof.to_string()}
        **Average Adsorption Capacity by MOF Type:**
        {adsorption_capacity_by_mof.to_string()}
        Based on the provided information, please provide a concise and direct data analysis report. You should:
        1. **Summarize the Overall Performance:** Directly state which MOF types perform best and worst for each of the two research goals. Use the numbers provided above.
        2. **Explain the Trade-offs:** Describe any observed relationship or trade-off between desorption energy and adsorption capacity. Which MOFs show the best balance?
        3. **Interesting Observations & Future Work:** Point out any interesting or unexpected observations from the data. Suggest what patterns should be investigated further or what new experiments could be run.
        4. **Provide a Conclusion:** Conclude with a clear recommendation of the most promising MOF candidates for future research based on both metrics.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not analyze this chart automatically. Error: {e}"

def analyze_scatter_plot(df, x_col, y_col):
    try:
        x_data = df[x_col].dropna()
        y_data = df[y_col].dropna()
        if len(x_data) < 2 or len(y_data) < 2:
            return "Not enough data points to perform a meaningful analysis."
        correlation, p_value = pearsonr(x_data, y_data)
        top_y_performer = df.loc[df[y_col].idxmax()]
        bottom_y_performer = df.loc[df[y_col].idxmin()]
        prompt = f"""
        You are an expert data analyst. A scatter plot has been generated with the following data:
        - X-axis: {x_col}
        - Y-axis: {y_col}
        - The Pearson correlation coefficient (r) between these two variables is: {correlation:.2f} (p-value: {p_value:.2f})
        - The top data point by Y-axis value is:
            - X value: {top_y_performer[x_col]}
            - Y value: {top_y_performer[y_col]}
        - The bottom data point by Y-axis value is:
            - X value: {bottom_y_performer[x_col]}
            - Y value: {bottom_y_performer[y_col]}
        Please provide a concise analysis of this scatter plot. Your analysis should cover:
        1. **General Relationship:** Describe the relationship between the two variables based on the correlation coefficient.
        2. **Key Findings:** Highlight the top and bottom performers. Are there any notable outliers or clusters?
        3. **Relevance:** Explain how this relationship is relevant to the main DAC objectives of minimizing desorption energy and maximizing adsorption capacity. Is the relationship between these two variables what we would expect?
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not analyze this scatter plot. Error: {e}"

def analyze_bar_chart(df, numerical_var, categorical_var):
    try:
        grouped_data = df.groupby(categorical_var)[numerical_var].agg(['mean', 'std']).sort_values(by='mean', ascending=False)
        prompt = f"""
        You are an expert data analyst. A bar chart has been generated with the following data:
        - X-axis: {categorical_var}
        - Y-axis: Mean of {numerical_var}
        Here is a summary of the grouped data:
        {grouped_data.to_string()}
        Please provide a concise analysis of this bar chart. Your analysis should cover:
        1. **Overall Trends:** Which categories have the highest and lowest mean values?
        2. **Variability:** Comment on the standard deviation. Does a large standard deviation mean the results for a particular category are inconsistent?
        3. **Relevance:** How is this analysis relevant to the main DAC objectives of minimizing desorption energy and maximizing adsorption capacity?
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not analyze this bar chart. Error: {e}"

def analyze_line_graph(df, x_col, y1_col, y2_col):
    try:
        y1_trend = "increasing" if df[y1_col].iloc[-1] > df[y1_col].iloc[0] else "decreasing"
        y2_trend = "increasing" if df[y2_col].iloc[-1] > df[y2_col].iloc[0] else "decreasing"
        prompt = f"""
        You are an expert data analyst. A line graph has been generated comparing two variables.
        - X-axis: {x_col}
        - Line 1: {y1_col} (shows an overall {y1_trend} trend)
        - Line 2: {y2_col} (shows an overall {y2_trend} trend)
        Please provide a concise analysis of this line graph. Your analysis should cover:
        1. **Comparison:** Describe the relationship between the two lines. Do they move in the same direction? Do they intersect?
        2. **Key Insights:** What is the most important observation from this graph?
        3. **Relevance:** How is this comparison relevant to the DAC research objectives?
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not analyze this line graph. Error: {e}"

def analyze_summary_stats(df, col):
    try:
        stats = df[col].describe().to_string()
        prompt = f"""
        You are an expert data analyst. You have been provided with the summary statistics for the variable '{col}'.
        Summary Statistics:
        {stats}
        Please provide a concise analysis of these statistics. Your analysis should cover:
        1. **Distribution:** Describe the distribution of the data based on the mean, median (50%), and standard deviation.
        2. **Outliers:** Are there any potential outliers based on the min/max values and quartiles?
        3. **Relevance:** How do these statistics relate to the overall DAC research objectives?
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not analyze these statistics. Error: {e}"

# --- Main App Logic ---
st.title(t["title"])

# Add language and theme buttons to the sidebar
with st.sidebar:
    st.button("🇹🇼 繁體中文" if st.session_state.lang == "en" else "🇺🇸 English", on_click=toggle_language)
    st.toggle("🌙 Dark Mode", value=st.session_state.is_dark_mode, on_change=toggle_dark_mode)

st.write(t["upload_prompt"])

# --- File Uploader Widget ---
uploaded_file = st.file_uploader(t["upload_button"], type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # --- Filtering Options ---
        st.sidebar.header(t["filter_header"])

        # Create a list of all filterable columns. Always include has_additive.
        filter_options = ['None'] + categorical_columns
        if 'has_additive' in df.columns and 'has_additive' not in filter_options:
            filter_options.append('has_additive')
        
        st.session_state.filter_column = st.sidebar.selectbox(
            t["filter_prompt_1"], 
            filter_options,
            index=get_index(filter_options, st.session_state.filter_column)
        )
        
        if st.session_state.filter_column != 'None':
            unique_values = sorted(df[st.session_state.filter_column].unique())
            st.session_state.selected_values = st.sidebar.multiselect(
                t["filter_multiselect"],
                unique_values,
                default=st.session_state.selected_values if st.session_state.selected_values else unique_values
            )
            if st.session_state.selected_values:
                df = df[df[st.session_state.filter_column].isin(st.session_state.selected_values)]
            else:
                st.warning(t["no_filter_selected"])
                st.stop()
        
        # Update the plot type based on user selection, and save to session state
        st.sidebar.header(t["plot_header"])
        st.session_state.plot_type = st.sidebar.radio(
            t["select_view"],
            ('Overall Data Analysis', 'Scatter Plot', 'Bar Chart', 'Line Graph', 'Summary Statistics'),
            index=get_index(['Overall Data Analysis', 'Scatter Plot', 'Bar Chart', 'Line Graph', 'Summary Statistics'], st.session_state.plot_type)
        )

        # --- Overall Data Analysis ---
        if st.session_state.plot_type == 'Overall Data Analysis':
            st.write(t["overall_analysis_intro"])
            
            # --- Perform and display key analyses ---
            st.subheader(t["desorption_analysis_header"])
            desorption_energy_by_mof = df.groupby('mof_type')['desorption_energy_kj_mol_exp'].mean().sort_values(ascending=True).reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(desorption_energy_by_mof['mof_type'], desorption_energy_by_mof['desorption_energy_kj_mol_exp'], color='skyblue')
            ax.set_title('Average Desorption Energy by MOF Type')
            ax.set_xlabel('MOF Type')
            ax.set_ylabel(r'Average Desorption Energy ($kJ/mol-CO_2$)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader(t["adsorption_analysis_header"])
            adsorption_capacity_by_mof = df.groupby('mof_type')['specific_adsorption_mmol_g'].mean().sort_values(ascending=False).reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(adsorption_capacity_by_mof['mof_type'], adsorption_capacity_by_mof['specific_adsorption_mmol_g'], color='lightgreen')
            ax.set_title('Average Adsorption Capacity by MOF Type')
            ax.set_xlabel('MOF Type')
            ax.set_ylabel(r'Average Adsorption Capacity ($mmol-CO_2/g$)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader(t["relationship_analysis_header"])
            fig, ax = plt.subplots(figsize=(10, 6))
            for mof_type in df['mof_type'].unique():
                subset = df[df['mof_type'] == mof_type]
                ax.scatter(subset['specific_adsorption_mmol_g'], subset['desorption_energy_kj_mol_exp'], label=mof_type)
            ax.set_title('Adsorption Capacity vs. Desorption Energy')
            ax.set_xlabel(r'Adsorption Capacity ($mmol-CO_2/g$)')
            ax.set_ylabel(r'Desorption Energy ($kJ/mol-CO_2$)')
            ax.legend(title='MOF Type', loc='best')
            plt.tight_layout()
            st.pyplot(fig)
            
            if st.button(t["get_ai_analysis"], key='overall_button'):
                st.session_state.analysis_text = analyze_overall_data(df)
                st.session_state.run_analysis = True
            
            if st.session_state.run_analysis:
                st.subheader(t["analysis_report_header"])
                final_text = translate_text(st.session_state.analysis_text, st.session_state.lang)
                st.write(final_text)

        # --- Scatter Plot ---
        elif st.session_state.plot_type == 'Scatter Plot':
            st.write(t["scatter_plot_intro"])
            
            st.session_state.scatter_x_axis = st.sidebar.selectbox(
                t["scatter_x_axis"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.scatter_x_axis)
            )
            st.session_state.scatter_y_axis = st.sidebar.selectbox(
                t["scatter_y_axis"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.scatter_y_axis)
            )
            
            # Create a list of all coloring options. Always include has_additive.
            color_by_options = ['None'] + categorical_columns
            if 'has_additive' in df.columns and 'has_additive' not in color_by_options:
                color_by_options.append('has_additive')
            
            st.session_state.scatter_color_by = st.sidebar.selectbox(
                t["scatter_color_by"],
                color_by_options,
                index=get_index(color_by_options, st.session_state.scatter_color_by)
            )

            st.session_state.scatter_trend_line = st.sidebar.radio(
                t["scatter_trend_line"],
                (t["scatter_trend_none"], t["scatter_trend_linear"], t["scatter_trend_poly"]),
                index=get_index((t["scatter_trend_none"], t["scatter_trend_linear"], t["scatter_trend_poly"]), st.session_state.scatter_trend_line)
            )
            
            if st.session_state.scatter_x_axis and st.session_state.scatter_y_axis:
                fig, ax = plt.subplots()
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']
                
                if st.session_state.scatter_color_by == 'None':
                    ax.scatter(df[st.session_state.scatter_x_axis], df[st.session_state.scatter_y_axis])
                else:
                    unique_categories = df[st.session_state.scatter_color_by].unique()
                    for i, category in enumerate(unique_categories):
                        subset = df[df[st.session_state.scatter_color_by] == category]
                        ax.scatter(subset[st.session_state.scatter_x_axis], subset[st.session_state.scatter_y_axis], color=colors[i % len(colors)])
                
                if st.session_state.scatter_trend_line != t["scatter_trend_none"]:
                    x = df[st.session_state.scatter_x_axis].dropna()
                    y = df[st.session_state.scatter_y_axis].dropna()
                    if not x.empty and not y.empty:
                        deg = 1 if st.session_state.scatter_trend_line == t["scatter_trend_linear"] else 2
                        z = np.polyfit(x, y, deg)
                        p = np.poly1d(z)
                        r2 = r2_score(y, p(x))
                        x_new = np.linspace(min(x), max(x), 100)
                        y_new = p(x_new)
                        ax.plot(x_new, y_new, "r--", label=f"R² = {r2:.2f}")
                        ax.legend()
                    
                ax.set_xlabel(st.session_state.scatter_x_axis)
                ax.set_ylabel(st.session_state.scatter_y_axis)
                ax.set_title(f'Scatter Plot of {st.session_state.scatter_y_axis} vs. {st.session_state.scatter_x_axis}')
                plt.tight_layout()
                st.pyplot(fig)
                
                if st.button(t["get_ai_analysis"], key='scatter_button'):
                    st.session_state.analysis_text = analyze_scatter_plot(df, st.session_state.scatter_x_axis, st.session_state.scatter_y_axis)
                    st.session_state.run_analysis = True
                
                if st.session_state.run_analysis:
                    st.subheader(t["analysis_report_header"])
                    final_text = translate_text(st.session_state.analysis_text, st.session_state.lang)
                    st.write(final_text)
        
        # --- Bar Chart ---
        elif st.session_state.plot_type == 'Bar Chart':
            st.write(t["bar_chart_intro"])
            
            st.session_state.bar_numerical_var = st.sidebar.selectbox(
                t["bar_numerical_var"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.bar_numerical_var)
            )
            st.session_state.bar_categorical_var = st.sidebar.selectbox(
                t["bar_categorical_var"],
                categorical_columns,
                index=get_index(categorical_columns, st.session_state.bar_categorical_var)
            )

            if st.session_state.bar_numerical_var and st.session_state.bar_categorical_var:
                grouped_data = df.groupby(st.session_state.bar_categorical_var)[st.session_state.bar_numerical_var].mean().reset_index()
                fig, ax = plt.subplots()
                ax.bar(grouped_data[st.session_state.bar_categorical_var], grouped_data[st.session_state.bar_numerical_var])
                ax.set_xlabel(st.session_state.bar_categorical_var)
                ax.set_ylabel(f'Mean of {st.session_state.bar_numerical_var}')
                ax.set_title(f'Mean of {st.session_state.bar_numerical_var} by {st.session_state.bar_categorical_var}')
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                
                if st.button(t["get_ai_analysis"], key='bar_button'):
                    st.session_state.analysis_text = analyze_bar_chart(df, st.session_state.bar_numerical_var, st.session_state.bar_categorical_var)
                    st.session_state.run_analysis = True

                if st.session_state.run_analysis:
                    st.subheader(t["analysis_report_header"])
                    final_text = translate_text(st.session_state.analysis_text, st.session_state.lang)
                    st.write(final_text)

        # --- Line Graph ---
        elif st.session_state.plot_type == 'Line Graph':
            st.write(t["line_graph_intro"])
            
            st.session_state.line_x_axis = st.sidebar.selectbox(
                t["line_x_axis"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.line_x_axis)
            )
            st.session_state.line_y1_axis = st.sidebar.selectbox(
                t["line_y1_axis"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.line_y1_axis)
            )
            st.session_state.line_y2_axis = st.sidebar.selectbox(
                t["line_y2_axis"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.line_y2_axis)
            )
            
            if st.session_state.line_x_axis and st.session_state.line_y1_axis and st.session_state.line_y2_axis:
                fig, ax = plt.subplots()
                ax.plot(df[st.session_state.line_x_axis], df[st.session_state.line_y1_axis], marker='o', label=st.session_state.line_y1_axis)
                ax.plot(df[st.session_state.line_x_axis], df[st.session_state.line_y2_axis], marker='x', linestyle='--', label=st.session_state.line_y2_axis)
                ax.set_xlabel(st.session_state.line_x_axis)
                ax.set_ylabel('Values')
                ax.set_title(f'Comparison of {st.session_state.line_y1_axis} and {st.session_state.line_y2_axis}')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                if st.button(t["get_ai_analysis"], key='line_button'):
                    st.session_state.analysis_text = analyze_line_graph(df, st.session_state.line_x_axis, st.session_state.line_y1_axis, st.session_state.line_y2_axis)
                    st.session_state.run_analysis = True
                
                if st.session_state.run_analysis:
                    st.subheader(t["analysis_report_header"])
                    final_text = translate_text(st.session_state.analysis_text, st.session_state.lang)
                    st.write(final_text)
        
        # --- Summary Statistics ---
        elif st.session_state.plot_type == 'Summary Statistics':
            st.write(t["stats_intro"])
            
            st.session_state.stats_var = st.sidebar.selectbox(
                t["stats_var"],
                numerical_columns,
                index=get_index(numerical_columns, st.session_state.stats_var)
            )
            
            if st.session_state.stats_var:
                summary_df = df[st.session_state.stats_var].describe().to_frame()
                st.table(summary_df)

                if st.button(t["get_ai_analysis"], key='stats_button'):
                    st.session_state.analysis_text = analyze_summary_stats(df, st.session_state.stats_var)
                    st.session_state.run_analysis = True

                if st.session_state.run_analysis:
                    st.subheader(t["analysis_report_header"])
                    final_text = translate_text(st.session_state.analysis_text, st.session_state.lang)
                    st.write(final_text)

    except Exception as e:
        st.error(f"{t['error_csv']} Error: {e}")
else:
    st.info(t["no_file_uploaded"])