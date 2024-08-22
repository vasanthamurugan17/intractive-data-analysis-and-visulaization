import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt








# Function to load and display the DataFrame(s)
def load_dataframe():
    upload_type = st.radio("Do you want to upload a single file or multiple files?", ('Single File', 'Multiple Files'))
    
    # Handle file uploads based on the user's choice
    if upload_type == 'Single File':
        uploaded_files = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)
    else:
        uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        if upload_type == 'Single File':
            uploaded_files = [uploaded_files]  # Convert to list for consistency
        
        dataframes = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read the CSV file, starting from the 8th row for feature names
                df = pd.read_csv(uploaded_file, header=8)  # Set header=7 to use the 8th row as header
                df = df.reset_index(drop=True)  # Reset index after skipping the first 7 rows

                # Append the DataFrame to the list
                dataframes.append(df)

                # Optionally display the DataFrame
                if st.checkbox(f"Do you want to display the DataFrame from {uploaded_file.name}?"):
                    st.write(df)
            
            # Handle exceptions during file reading
            except Exception as e:
                st.error(f"Error loading file {uploaded_file.name}: {e}")
        
        return dataframes
    
    else:
        st.info("Please upload a CSV file.")
        return None
    















def normalize_and_plot(dataframes):
    # Ensure dataframes is a list and check if it's empty
    if not isinstance(dataframes, list) or len(dataframes) == 0:
        st.error("No files uploaded. Please upload at least one CSV file.")
        return

    # Step 1: Ask user to choose the normalization scope
    scope = st.radio("Do you want to normalize and plot data for:", ["Single File", "Multiple Files", "All Files"], key="normalize_scope")

    if scope == "Single File":
        # For a single file, select one DataFrame
        selected_idx = st.selectbox("Select the DataFrame:", range(len(dataframes)), format_func=lambda x: f"File {x+1}", key="single_file_selector")

        selected_dfs = [dataframes[selected_idx]]
    elif scope == "Multiple Files":
        # For multiple files, allow the user to select multiple DataFrames
        selected_dfs = st.multiselect("Select DataFrames to normalize and plot:", range(len(dataframes)), format_func=lambda x: f"File {x+1}", key="multiple_file_selector")

        if not selected_dfs:
            st.error("No files selected. Please select at least one file.")
            return
        selected_dfs = [dataframes[idx] for idx in selected_dfs]

    elif scope == "All Files":
        # Use all DataFrames
        selected_dfs = dataframes

    # Ensure there are selected DataFrames and that none of them are empty
    if not selected_dfs or any(df.empty for df in selected_dfs):
        st.error("One or more selected DataFrames are empty.")
        return

    # Step 2: Select columns and plot type
    graph_type = st.selectbox("Select the graph type:", ['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'], key="normalize_graph_type")
    
    # Allow multiple features to be selected for the x-axis and y-axis
    x_cols = st.multiselect("Select the columns for the x-axis:", dataframes[0].columns, key="normalize_x_axis_cols")
    y_cols = st.multiselect("Select the columns for the y-axis:", dataframes[0].columns, key="normalize_y_axis_cols")

    # Define colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
    color_index = 0

    if st.button("Normalize and Plot"):
        plt.figure(figsize=(10, 6))
        for i, df in enumerate(selected_dfs):
            new_df = df.copy()
            max_values = new_df.iloc[:, 1:7].max()  # Adjust column range as necessary
            for column in new_df.columns[1:7]:
                new_df[column] = new_df[column] / max_values[column]

            for x_col in x_cols:
                for y_col in y_cols:
                    color = colors[color_index % len(colors)]
                    color_index += 1  # Increment the color index to ensure unique colors
                    if graph_type == 'Scatter Plot':
                        plt.scatter(new_df[x_col], new_df[y_col], label=f"Dataset {i+1}: {x_col} vs {y_col}", color=color)
                    elif graph_type == 'Line Plot':
                        plt.plot(new_df[x_col], new_df[y_col], label=f"Dataset {i+1}: {x_col} vs {y_col}", color=color)
                    elif graph_type == 'Box Plot':
                        plt.boxplot([new_df[x_col], new_df[y_col]], positions=[i*2, i*2+1], labels=[f"Dataset {i+1} - {x_col}", f"Dataset {i+1} - {y_col}"])
                    elif graph_type == 'Bar Plot':
                        plt.bar(new_df[x_col], new_df[y_col], label=f"Dataset {i+1}: {x_col} vs {y_col}", color=color)

        plt.xlabel(", ".join(x_cols))
        plt.ylabel(", ".join(y_cols))
        plt.title("Normalized Graph")
        plt.legend()
        st.pyplot(plt)




























def plot_graph(dataframes):
    if dataframes:
        plot_type = st.radio("Do you want to plot the graph based on a single file or multiple files?", ("Single File", "Multiple Files"))

        if plot_type == "Single File":
            df_idx = st.selectbox("Select the DataFrame:", range(len(dataframes)))
            df = dataframes[df_idx]
            x_cols = st.multiselect("Select the column(s) for the x-axis:", df.columns)
            y_cols = st.multiselect("Select the column(s) for the y-axis:", df.columns)
            if st.button("Plot Graph (Single File)"):
                plt.figure(figsize=(10, 6))
                for x_col, y_col in zip(x_cols, y_cols):
                    plt.plot(df[x_col], df[y_col], label=f"File {df_idx+1} - {x_col} vs {y_col}")
                plt.xlabel(', '.join(x_cols))
                plt.ylabel(', '.join(y_cols))
                plt.title(f"Graph - File {df_idx+1}")
                plt.legend()
                st.pyplot(plt)

        elif plot_type == "Multiple Files":
            selected_dfs = st.multiselect("Select DataFrames for comparison:", range(len(dataframes)), format_func=lambda x: f"File {x+1}")
            x_cols = st.multiselect("Select the column(s) for the x-axis:", dataframes[0].columns)
            y_cols = st.multiselect("Select the column(s) for the y-axis:", dataframes[0].columns)
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'pink']  # Predefined color list
            if st.button("Plot Graph (Multiple Files)"):
                plt.figure(figsize=(10, 6))
                for i, df_idx in enumerate(selected_dfs):
                    df = dataframes[df_idx]
                    color = colors[i % len(colors)]  # Cycle through the predefined colors
                    for x_col, y_col in zip(x_cols, y_cols):
                        plt.plot(df[x_col], df[y_col], label=f"File {df_idx+1} - {x_col} vs {y_col}", color=color)
                plt.xlabel(', '.join(x_cols))
                plt.ylabel(', '.join(y_cols))
                plt.title("Comparison Plot")
                plt.legend()
                st.pyplot(plt)

        normalize = st.radio("Would you like to normalize the data?", ('Yes', 'No'))
        normalize = normalize == 'Yes'
        normalize_and_plot(dataframes)














def plot_graph_with_range(dataframes):
    if not isinstance(dataframes, list) or len(dataframes) == 0:
        st.error("No files uploaded. Please upload at least one CSV file.")
        return

    operation_choice = st.radio("Do you want to perform this operation on a single file or multiple files?", ["Single File", "Multiple Files"], key="operation_choice")

    selected_dfs = []
    if operation_choice == "Single File":
        selected_idx = st.selectbox("Select the file", range(len(dataframes)), key="single_file_selector_range")
        selected_dfs.append(dataframes[selected_idx])
    else:
        selected_idxs = st.multiselect("Select the files", range(len(dataframes)), key="multiple_file_selector_range")
        selected_dfs = [dataframes[idx] for idx in selected_idxs]

    if not selected_dfs or any(df.empty for df in selected_dfs):
        st.error("No files selected or one or more selected DataFrames are empty.")
        return

    range_type = st.radio("Do you want to create the DataFrame based on row or value?", ["Row", "Value"], key="range_type")

    new_dfs = []

    if range_type == "Row":
        start_row = st.number_input(f"Enter the starting row (from) for all selected DataFrames:", min_value=0, max_value=len(selected_dfs[0])-1, key="start_row_all")
        end_row = st.number_input(f"Enter the ending row (to) for all selected DataFrames:", min_value=0, max_value=len(selected_dfs[0])-1, key="end_row_all")
        
        apply_same_range = st.checkbox("Apply the same row range to all selected DataFrames?", key="apply_same_range_row")
        if apply_same_range:
            for df in selected_dfs:
                new_df = df.iloc[start_row:end_row+1]
                new_dfs.append(new_df)
        else:
            for i, df in enumerate(selected_dfs):
                start_row = st.number_input(f"Enter the starting row (from) for DataFrame {i+1}:", min_value=0, max_value=len(df)-1, key=f"start_row_{i}")
                end_row = st.number_input(f"Enter the ending row (to) for DataFrame {i+1}:", min_value=0, max_value=len(df)-1, key=f"end_row_{i}")
                new_df = df.iloc[start_row:end_row+1]
                new_dfs.append(new_df)

    else:  # Range based on values
        x_col_name = st.selectbox(f"Select the column for the value-based range:", selected_dfs[0].columns, key="x_col_name_all")
        min_val = st.number_input(f"Enter the minimum value for {x_col_name}:", value=selected_dfs[0][x_col_name].min(), key="min_val_all")
        max_val = st.number_input(f"Enter the maximum value for {x_col_name}:", value=selected_dfs[0][x_col_name].max(), key="max_val_all")

        apply_same_range = st.checkbox("Apply the same min and max values to all selected DataFrames?", key="apply_same_range_value")
        if apply_same_range:
            for df in selected_dfs:
                new_df = df[(df[x_col_name] >= min_val) & (df[x_col_name] <= max_val)]
                new_dfs.append(new_df)
        else:
            for i, df in enumerate(selected_dfs):
                x_col_name = st.selectbox(f"Select the column for the value-based range (DataFrame {i+1}):", df.columns, key=f"x_col_name_{i}")
                min_val = st.number_input(f"Enter the minimum value for {x_col_name} (DataFrame {i+1}):", value=df[x_col_name].min(), key=f"min_val_{i}")
                max_val = st.number_input(f"Enter the maximum value for {x_col_name} (DataFrame {i+1}):", value=df[x_col_name].max(), key=f"max_val_{i}")
                new_df = df[(df[x_col_name] >= min_val) & (df[x_col_name] <= max_val)]
                new_dfs.append(new_df)

    # Graph plotting and normalization options as before
    graph_type = st.selectbox("Select the graph type:", ['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'], key="range_graph_type")
    x_cols = st.multiselect("Select the column(s) for the x-axis:", new_dfs[0].columns, key="range_x_axis_col")
    y_cols = st.multiselect("Select the column(s) for the y-axis:", new_dfs[0].columns, key="range_y_axis_col")

    # Checkbox to view the modified dataset
    if st.checkbox("View modified dataset?", key="view_modified_data"):
        for i, new_df in enumerate(new_dfs):
            st.write(f"Modified DataFrame {i+1}")
            st.write(new_df)

    # Button to plot graph
    if st.button("Plot Graph with Range", key="plot_graph_with_range"):
        plt.figure(figsize=(10, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
        color_count = len(new_dfs)
        if color_count > len(colors):
            colors *= (color_count // len(colors)) + 1
        colors = colors[:color_count]

        for i, new_df in enumerate(new_dfs):
            for x_col, y_col in zip(x_cols, y_cols):
                if graph_type == 'Scatter Plot':
                    plt.scatter(new_df[x_col], new_df[y_col], label=f"Dataset {i+1} - {x_col} vs {y_col}", color=colors[i])
                elif graph_type == 'Line Plot':
                    plt.plot(new_df[x_col], new_df[y_col], label=f"Dataset {i+1} - {x_col} vs {y_col}", color=colors[i])
                elif graph_type == 'Box Plot':
                    plt.boxplot([new_df[x_col], new_df[y_col]], positions=[i*2, i*2+1], labels=[f"Dataset {i+1} - {x_col}", f"Dataset {i+1} - {y_col}"])
                elif graph_type == 'Bar Plot':
                    plt.bar(new_df[x_col], new_df[y_col], label=f"Dataset {i+1} - {x_col} vs {y_col}", color=colors[i])

        plt.xlabel(', '.join(x_cols))
        plt.ylabel(', '.join(y_cols))
        plt.title(f"{graph_type} - Comparison")
        plt.legend()
        st.pyplot(plt)

    # Normalization option
    normalize = st.radio("Would you like to normalize the data?", ('Yes', 'No'), key="range_normalize_option")
    if normalize == 'Yes':
        normalize_and_plot(new_dfs)



















def add_column(dataframes):
    if not dataframes:
        st.error("No files uploaded. Please upload at least one CSV file.")
        return

    # Step 1: Ask user to choose the scope
    scope = st.radio("Do you want to add a column for:", ["Single File", "Multiple Files", "All Files"], key="add_column_scope")

    if scope == "Single File":
        # For a single file, select one DataFrame
        selected_idx = st.selectbox("Select the DataFrame:", range(len(dataframes)), format_func=lambda x: f"File {x+1}", key="single_file_selector")
        selected_dfs = [dataframes[selected_idx]]

    elif scope == "Multiple Files":
        # For multiple files, allow the user to select multiple DataFrames
        selected_dfs = st.multiselect("Select DataFrames to add column:", range(len(dataframes)), format_func=lambda x: f"File {x+1}", key="multiple_file_selector")

        if not selected_dfs:
            st.error("No files selected. Please select at least one file.")
            return
        selected_dfs = [dataframes[idx] for idx in selected_dfs]

    elif scope == "All Files":
        # Use all DataFrames
        selected_dfs = dataframes

    # Get columns from the first DataFrame for initial selection
    if selected_dfs:
        columns = selected_dfs[0].columns.tolist()
    else:
        columns = []

    # Initialize or update session state for columns
    if 'current_columns' not in st.session_state:
        st.session_state.current_columns = columns
    else:
        # Update columns in session state based on the selected DataFrames
        st.session_state.current_columns = list(set().union(*[df.columns for df in selected_dfs]))

    # Step 2: Select columns and operation
    col1 = st.selectbox("Select the first column:", st.session_state.current_columns, key="first_column")
    col2 = st.selectbox("Select the second column:", st.session_state.current_columns, key="second_column")
    operation = st.selectbox("Select the operation:", ['Add', 'Multiply'])
    new_col_name = st.text_input("Enter the name of the new column:")

    # Plot the graphs based on current DataFrames before the operation
    plot_type = st.selectbox("Select the graph type to plot:", ['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'])

    # Allow multiple selections for x and y columns
    x_cols = st.multiselect("Select the columns for the x-axis:", st.session_state.current_columns, key="plot_x_cols")
    y_cols = st.multiselect("Select the columns for the y-axis:", st.session_state.current_columns, key="plot_y_cols")

    # Define colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
    color_count = len(selected_dfs)
    if color_count > len(colors):
        colors *= (color_count // len(colors)) + 1
    colors = colors[:color_count]

    if st.button("Plot Updated DataFrames"):
        plt.figure(figsize=(12, 8))
        for i, df in enumerate(selected_dfs):
            for x_col in x_cols:
                for y_col in y_cols:
                    if x_col in df.columns and y_col in df.columns:
                        st.write(f"Plotting DataFrame from file {dataframes.index(df)+1}...")
                        if plot_type == 'Scatter Plot':
                            plt.scatter(df[x_col], df[y_col], label=f"File {dataframes.index(df)+1}: {x_col} vs {y_col}", color=colors[i])
                        elif plot_type == 'Line Plot':
                            plt.plot(df[x_col], df[y_col], label=f"File {dataframes.index(df)+1}: {x_col} vs {y_col}", color=colors[i])
                        elif plot_type == 'Box Plot':
                            plt.boxplot([df[x_col], df[y_col]], positions=[i*2, i*2+1], labels=[f"File {dataframes.index(df)+1} - {x_col}", f"File {dataframes.index(df)+1} - {y_col}"])
                        elif plot_type == 'Bar Plot':
                            plt.bar(df[x_col], df[y_col], label=f"File {dataframes.index(df)+1}: {x_col} vs {y_col}", color=colors[i])
                    else:
                        st.warning(f"Selected columns {x_col} or {y_col} do not exist in DataFrame from file {dataframes.index(df)+1}.")

        plt.xlabel("X-axis Columns")
        plt.ylabel("Y-axis Columns")
        plt.title("Graph of Selected DataFrames")
        plt.legend()
        st.pyplot(plt)

    if st.button("Add Column"):
        # Update the DataFrames with the new column
        for df in selected_dfs:
            if operation == 'Add':
                df[new_col_name] = df[col1] + df[col2]
            elif operation == 'Multiply':
                df[new_col_name] = df[col1] * df[col2]

        # Update session state with the new columns
        for df in selected_dfs:
            st.session_state.current_columns = list(df.columns)
            st.write(f"Updated DataFrame from file {dataframes.index(df)+1}:")
            st.write(df)

























def multiply_column(dataframes):
    if not dataframes:
        st.error("No files uploaded. Please upload at least one CSV file.")
        return

    # Step 1: Ask user to choose the scope
    scope = st.radio("Do you want to multiply a column for:", ["Single File", "Multiple Files", "All Files"], key="multiply_column_scope")

    if scope == "Single File":
        # For a single file, select one DataFrame
        selected_idx = st.selectbox("Select the DataFrame:", range(len(dataframes)), format_func=lambda x: f"File {x+1}", key="single_file_selector")
        selected_dfs = [dataframes[selected_idx]]

    elif scope == "Multiple Files":
        # For multiple files, allow the user to select multiple DataFrames
        selected_dfs = st.multiselect("Select DataFrames to multiply column:", range(len(dataframes)), format_func=lambda x: f"File {x+1}", key="multiple_file_selector")

        if not selected_dfs:
            st.error("No files selected. Please select at least one file.")
            return
        selected_dfs = [dataframes[idx] for idx in selected_dfs]

    elif scope == "All Files":
        # Use all DataFrames
        selected_dfs = dataframes

    # Get columns from the first DataFrame for initial selection
    if selected_dfs:
        columns = selected_dfs[0].columns.tolist()
    else:
        columns = []

    # Initialize or update session state for columns
    if 'current_columns' not in st.session_state:
        st.session_state.current_columns = columns
    else:
        # Update columns in session state based on the selected DataFrames
        st.session_state.current_columns = list(set().union(*[df.columns for df in selected_dfs]))

    # Step 2: Select columns and factor
    col = st.selectbox("Select the column to multiply:", st.session_state.current_columns, key="multiply_column_selector")
    factor = st.number_input("Enter the multiplication factor:", min_value=0.0)
    new_col_name = st.text_input("Enter the name of the new column:")

    # Plot the graphs based on current DataFrames before the operation
    plot_type = st.selectbox("Select the graph type to plot:", ['Scatter Plot', 'Line Plot', 'Box Plot', 'Bar Plot'])

    # Allow multiple selections for the x-axis and y-axis columns
    x_cols = st.multiselect("Select columns for the x-axis:", st.session_state.current_columns, key="plot_x_cols")
    y_cols = st.multiselect("Select columns for the y-axis:", st.session_state.current_columns, key="plot_y_cols")

    # Define colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
    color_count = len(selected_dfs)
    if color_count > len(colors):
        colors *= (color_count // len(colors)) + 1
    colors = colors[:color_count]

    if st.button("Plot Updated DataFrames"):
        plt.figure(figsize=(10, 6))
        for i, df in enumerate(selected_dfs):
            for x_col in x_cols:
                for y_col in y_cols:
                    if x_col in df.columns and y_col in df.columns:
                        st.write(f"Plotting DataFrame from file {dataframes.index(df)+1}...")
                        if plot_type == 'Scatter Plot':
                            plt.scatter(df[x_col], df[y_col], label=f"Dataset {dataframes.index(df)+1}: {x_col} vs {y_col}", color=colors[i])
                        elif plot_type == 'Line Plot':
                            plt.plot(df[x_col], df[y_col], label=f"Dataset {dataframes.index(df)+1}: {x_col} vs {y_col}", color=colors[i])
                        elif plot_type == 'Box Plot':
                            plt.boxplot([df[x_col], df[y_col]], positions=[i*2, i*2+1], labels=[f"Dataset {dataframes.index(df)+1} - {x_col}", f"Dataset {dataframes.index(df)+1} - {y_col}"])
                        elif plot_type == 'Bar Plot':
                            plt.bar(df[x_col], df[y_col], label=f"Dataset {dataframes.index(df)+1}: {x_col} vs {y_col}", color=colors[i])
                    else:
                        st.warning(f"Selected columns {x_col} or {y_col} do not exist in DataFrame from file {dataframes.index(df)+1}.")

        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.title("Graph of Selected DataFrames")
        plt.legend()
        st.pyplot(plt)

    if st.button("Multiply Column"):
        # Update the DataFrames with the new column
        for df in selected_dfs:
            df[new_col_name] = df[col] * factor

        # Update session state with the new columns
        for df in selected_dfs:
            st.session_state.current_columns = list(df.columns)
            st.write(f"Updated DataFrame from file {dataframes.index(df)+1}:")
            st.write(df)


# Streamlit App
def main():
    st.title("Interactive Data Analysis and Visualization")
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the navigation below to select the operation:")
    
    dataframes = load_dataframe()
    if dataframes:
        df = dataframes[0]  # Assuming operations are on the first DataFrame for simplicity
        original_df = df.copy()

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
                plot_graph(dataframes)
            elif option == "Plot Graph with Range":
                plot_graph_with_range(dataframes)
            elif option == "Normalize and Plot":
                normalize_and_plot(dataframes)

        elif option in ["Create a new column by Add or Multiply by Column", "creation of column by Multiplying factor"]:
            modify_original = st.sidebar.radio("Modify original DataFrame or store separately?", ["Original", "Separately"])
            if option == "Create a new column by Add or Multiply by Column":
                add_column(dataframes)
            elif option == "creation of column by Multiplying factor":
                multiply_column(dataframes)
            
            if modify_original == "Original":
                original_df.update(df)

if __name__ == "__main__":
    main()

