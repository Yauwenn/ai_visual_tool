import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Function to analyze a graph with Gemini
def analyze_graph(chart_type, x_col, y_col, df):
    try:
        # Generate summary statistics (to avoid sending full dataset)
        stats = df[[x_col, y_col]].describe().to_dict()

        prompt = f"""
        You are an expert data analyst. A {chart_type} has been generated.

        X-axis: {x_col}
        Y-axis: {y_col}

        Summary statistics: {stats}

        Please provide:
        1. A short description of the graph.
        2. Explanation of the X and Y axes.
        3. Key insights (trends, best/worst performers, outliers).
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        return f"⚠️ Could not analyze this chart automatically. Error: {e}"


# Set the title of the app
st.title('Interactive MOF Data Dashboard')
st.write('Upload a CSV file to begin.')

# --- File Uploader Widget ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# We will only proceed if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the uploaded file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # --- Get the list of numerical and categorical columns ---
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

        # --- Filtering Options ---
        st.sidebar.header('Filter Data')
        filter_column = st.sidebar.selectbox('Select a column to filter by', ['None'] + categorical_columns)
        
        if filter_column != 'None':
            unique_values = sorted(df[filter_column].unique())
            selected_values = st.sidebar.multiselect(
                'Select values to include',
                unique_values,
                default=unique_values
            )
            if selected_values:
                df = df[df[filter_column].isin(selected_values)]
            else:
                st.warning("Please select at least one value to display data.")
                st.stop()
        
        # --- Create a plot type selection in the sidebar ---
        st.sidebar.header('Plot & Statistics')
        plot_type = st.sidebar.radio(
            "Select View",
            ('Scatter Plot', 'Bar Chart', 'Line Graph', 'Summary Statistics')
        )

        # --- Scatter Plot ---
        if plot_type == 'Scatter Plot':
            st.write('Select two variables and a coloring category to create a scatter plot.')
            x_axis = st.sidebar.selectbox('Select a variable for the X-Axis', numerical_columns)
            y_axis = st.sidebar.selectbox('Select a variable for the Y-Axis', numerical_columns)
            color_by = st.sidebar.selectbox('Color by', ['None'] + categorical_columns)
            
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                if color_by == 'None':
                    ax.scatter(df[x_axis], df[y_axis])
                else:
                    for category in df[color_by].unique():
                        subset = df[df[color_by] == category]
                        ax.scatter(subset[x_axis], subset[y_axis], label=category)
                    ax.legend()
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f'Scatter Plot of {y_axis} vs. {x_axis}')
                st.pyplot(fig)

                # AI analysis
                analysis = analyze_graph("Scatter Plot", x_axis, y_axis, df)
                st.subheader("AI Analysis")
                st.write(analysis)
        
        # --- Bar Chart ---
        elif plot_type == 'Bar Chart':
            st.write('Select a numerical and a categorical variable to create a bar chart.')
            numerical_var = st.sidebar.selectbox('Select a Numerical Variable', numerical_columns)
            categorical_var = st.sidebar.selectbox('Select a Categorical Variable', categorical_columns)
            
            if numerical_var and categorical_var:
                grouped_data = df.groupby(categorical_var)[numerical_var].mean().reset_index()
                fig, ax = plt.subplots()
                ax.bar(grouped_data[categorical_var], grouped_data[numerical_var])
                ax.set_xlabel(categorical_var)
                ax.set_ylabel(f'Mean of {numerical_var}')
                ax.set_title(f'Mean of {numerical_var} by {categorical_var}')
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

                # AI analysis
                analysis = analyze_graph("Bar Chart", categorical_var, numerical_var, grouped_data)
                st.subheader("AI Analysis")
                st.write(analysis)

        # --- Line Graph ---
        elif plot_type == 'Line Graph':
            st.write('Select two variables to compare over a common X-axis.')
            x_axis = st.sidebar.selectbox('Select a common X-Axis', numerical_columns)
            y_axis_1 = st.sidebar.selectbox('Select the 1st Y-Axis variable', numerical_columns)
            y_axis_2 = st.sidebar.selectbox('Select the 2nd Y-Axis variable', numerical_columns)
            
            if x_axis and y_axis_1 and y_axis_2:
                fig, ax = plt.subplots()
                ax.plot(df[x_axis], df[y_axis_1], marker='o', label=y_axis_1)
                ax.plot(df[x_axis], df[y_axis_2], marker='x', linestyle='--', label=y_axis_2)
                ax.set_xlabel(x_axis)
                ax.set_ylabel('Values')
                ax.set_title(f'Comparison of {y_axis_1} and {y_axis_2}')
                ax.legend()
                st.pyplot(fig)

                # AI analysis
                analysis = analyze_graph("Line Graph", x_axis, f"{y_axis_1} and {y_axis_2}", df)
                st.subheader("AI Analysis")
                st.write(analysis)

        # --- Summary Statistics ---
        elif plot_type == 'Summary Statistics':
            st.write('Select a variable to view its summary statistics.')
            stat_var = st.sidebar.selectbox('Select a Numerical Variable', numerical_columns)
            if stat_var:
                summary_df = df[stat_var].describe().to_frame()
                st.table(summary_df)

                # Optional AI analysis for stats
                analysis = analyze_graph("Summary Statistics", stat_var, stat_var, df)
                st.subheader("AI Analysis")
                st.write(analysis)

    except Exception as e:
        st.error(f"An error occurred while reading the file. Please ensure it is a valid CSV file. Error: {e}")
else:
    st.info("Please upload a CSV file to begin the analysis.")
