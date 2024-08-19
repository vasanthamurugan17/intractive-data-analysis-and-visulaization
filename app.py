import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load and display the DataFrame
def load_dataframe():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if st.checkbox("Do you want to display the DataFrame?"):
                st.write(df)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        st.info("Please upload a CSV file.")
        return None

# Function to plot the graph based on user input
def plot_graph(df, df_name="Original DataFrame"):
    if df is not None:
        x_col = st.number_input("Enter the column number for the x-axis:", min_value=0, max_value=len(df.columns)-1)
        y_col = st.number_input("Enter the column number for the y-axis:", min_value=0, max_value=len(df.columns)-1)
        if st.button(f"Plot Graph ({df_name})"):
            if 0 <= x_col < len(df.columns) and 0 <= y_col < len(df.columns):
                plt.figure(figsize=(10, 6))
                plt.plot(df.iloc[:, x_col], df.iloc[:, y_col])
                plt.xlabel(df.columns[x_col])
                plt.ylabel(df.columns[y_col])
                plt.title(f"Graph - {df_name}")
                st.pyplot(plt)
            else:
                st.error("Invalid column numbers. Please enter valid column numbers within the range.")

# Function to select the row range and plot a graph with different options
def plot_graph_with_range(df, df_name="Original DataFrame"):
    if df is not None:
        start_row = st.number_input("Enter the starting row (from):", min_value=0, max_value=len(df)-1)
        end_row = st.number_input("Enter the ending row (to):", min_value=0, max_value=len(df)-1)
        graph_type = st.selectbox("Select the graph type:", ['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'])
        x_col = st.number_input("Enter the column number for the x-axis:", min_value=0, max_value=len(df.columns)-1)
        y_col = st.number_input("Enter the column number for the y-axis:", min_value=0, max_value=len(df.columns)-1)

        if st.button("Plot Graph with Range"):
            new_df = df.iloc[start_row:end_row+1]
            st.write(f"New DataFrame based on the specified range ({df_name}):")
            st.write(new_df)

            if 0 <= x_col < len(new_df.columns) and 0 <= y_col < len(new_df.columns):
                plt.figure(figsize=(10, 6))
                if graph_type == 'Scatter Plot':
                    plt.scatter(new_df.iloc[:, x_col], new_df.iloc[:, y_col])
                elif graph_type == 'Line Plot':
                    plt.plot(new_df.iloc[:, x_col], new_df.iloc[:, y_col])
                elif graph_type == 'Box Plot':
                    plt.boxplot([new_df.iloc[:, x_col], new_df.iloc[:, y_col]])
                elif graph_type == 'Bar Plot':
                    plt.bar(new_df.iloc[:, x_col], new_df.iloc[:, y_col])
                plt.xlabel(new_df.columns[x_col])
                plt.ylabel(new_df.columns[y_col])
                plt.title(f"{graph_type} - {df_name}")
                st.pyplot(plt)
            else:
                st.error("Invalid column numbers. Please enter valid column numbers within the range.")

def normalize_and_plot(df):
    if df is not None:
        start_row = st.number_input("Enter the starting row (from):", min_value=0, max_value=len(df)-1)
        end_row = st.number_input("Enter the ending row (to):", min_value=0, max_value=len(df)-1)
        graph_type = st.selectbox("Select the graph type:", ['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'])
        x_col = st.number_input("Enter the column number for the x-axis:", min_value=0, max_value=len(df.columns)-1)
        y_col = st.number_input("Enter the column number for the y-axis:", min_value=0, max_value=len(df.columns)-1)

        if st.button("Normalize and Plot"):
            new_df = df.iloc[start_row:end_row+1]
            max_values = new_df.iloc[:, 1:7].max()  # Adjust column range as necessary

            new_df1 = new_df.copy()
            for column in new_df.columns[1:7]:
                new_df1[column] = new_df1[column] / max_values[column]

            st.write("New DataFrame with normalized values:")
            st.write(new_df1)

            if 0 <= x_col < len(new_df1.columns) and 0 <= y_col < len(new_df1.columns):
                plt.figure(figsize=(10, 6))
                if graph_type == 'Scatter Plot':
                    plt.scatter(new_df1.iloc[:, x_col], new_df1.iloc[:, y_col])
                elif graph_type == 'Line Plot':
                    plt.plot(new_df1.iloc[:, x_col], new_df1.iloc[:, y_col])
                elif graph_type == 'Box Plot':
                    plt.boxplot([new_df1.iloc[:, x_col], new_df1.iloc[:, y_col]])
                elif graph_type == 'Bar Plot':
                    plt.bar(new_df1.iloc[:, x_col], new_df1.iloc[:, y_col])
                plt.xlabel(new_df1.columns[x_col])
                plt.ylabel(new_df1.columns[y_col])
                plt.title("Normalized Graph")
                st.pyplot(plt)
            else:
                st.error("Invalid column numbers. Please enter valid column numbers within the range.")


# Function to add a new column based on the multiplication or addition of previous columns
def add_column(df):
    if df is not None:
        col1 = st.number_input("Enter the first column number:", min_value=0, max_value=len(df.columns)-1)
        col2 = st.number_input("Enter the second column number:", min_value=0, max_value=len(df.columns)-1)
        operation = st.selectbox("Select the operation:", ['Add', 'Multiply'])
        new_col_name = st.text_input("Enter the name of the new column:")

        if st.button("Add Column"):
            if operation == 'Add':
                df[new_col_name] = df.iloc[:, col1] + df.iloc[:, col2]
            elif operation == 'Multiply':
                df[new_col_name] = df.iloc[:, col1] * df.iloc[:, col2]
            st.write("New DataFrame with the added column:")
            st.write(df)

# Function to multiply an entire column by a multiplication factor
def multiply_column(df):
    if df is not None:
        col = st.number_input("Enter the column number to multiply:", min_value=0, max_value=len(df.columns)-1)
        factor = st.number_input("Enter the multiplication factor:", min_value=0.0)
        new_col_name = st.text_input("Enter the name of the new column:")

        if st.button("Multiply Column"):
            df[new_col_name] = df.iloc[:, col] * factor
            st.write("New DataFrame with the multiplied column:")
            st.write(df)

# Streamlit App
def main():
    st.title("Interactive Data Analysis and Visualization")
    # Set up the main title and sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the navigation below to select the operation:")
    
    
    df = load_dataframe()
    original_df = df.copy() if df is not None else None

    if df is not None:
        st.sidebar.header("Options")
        option = st.sidebar.radio("Select an Option", [
            "Plot Graph based on DataFrame",
            "Plot Graph with Range",
            "Normalize and Plot",
            "Create a new column by Add or Multiply by Column",
            "creation of column by Multiplying factor"
        ])

        if option in ["Plot Graph based on DataFrame", "Plot Graph with Range", "Normalize and Plot"]:
            use_original_df = st.sidebar.radio("Use original DataFrame or modified DataFrame?", ["Original", "Modified"])
            df_to_use = original_df if use_original_df == "Original" else df

            if option == "Plot Graph based on DataFrame":
                plot_graph(df_to_use, use_original_df)
            elif option == "Plot Graph with Range":
                plot_graph_with_range(df_to_use, use_original_df)
            elif option == "Normalize and Plot":
                normalize_and_plot(df_to_use)

        elif option in ["Create a new column by Add or Multiply by Column", "creation of column by Multiplying factor"]:
            modify_original = st.sidebar.radio("Modify original DataFrame or store separately?", ["Original", "Separately"])
            if option == "Create a new column by Add or Multiply by Column":
                add_column(df)
            elif option == "creation of column by Multiplying factor":
                multiply_column(df)
            
            if modify_original == "Original":
                original_df.update(df)

if __name__ == "__main__":
    main()
