### No-Code ML Pipeline Builder

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?style=for-the-badge&logo=streamlit)](https://nocodemlpipelinebuilder.streamlit.app/)


This project involves developing a web-based application that enables users to construct and execute simple machine learning (ML) pipelines without writing any code. The primary objective is to democratize ML workflows, allowing non-technical users to design functionality, create clean user interfaces (UIs), and transform complex processes into an intuitive, visual experience. It evaluates skills in building reliable software with a focus on user-centric design rather than academic expertise.

Core features start with dataset upload, where users can import CSV or Excel files. The app displays essential dataset details like row count, column count, and names, while gracefully handling invalid formats with error messages. Next, data preprocessing offers selectable options such as Standardization (using StandardScaler) and Normalization (MinMaxScaler), applied directly through the UI for seamless integration.

Users then perform a train-test split, choosing ratios like 70-30 or 80-20, with clear indicators showing the dataset division. Model selection is straightforward, limited to one model at a time: Logistic Regression or Decision Tree Classifier. Once selected, the model trains on the processed data, producing visible outputs including execution status, accuracy or other performance metrics, and optional visualizations (e.g., confusion matrices or decision trees if applicable).

The user experience emphasizes a drag-and-drop or step-based interface, visually representing the pipeline flow from data upload to preprocessing, splitting, modeling, and results. No coding is required, ensuring each step is self-explanatory and beginner-friendly. For inspiration, the design draws from tools like Orange Data Mining, prioritizing logical progression and clarity.

Evaluation prioritizes functionality (reliable end-to-end implementation), UI quality (clean, intuitive structure with visual flow), and ease of use (self-explanatory interactions for novices). Overall, the project aims to deliver working software with thoughtful UX, simplifying ML for broader accessibility.

## Key Features
- **Dataset Upload**: Supports CSV/Excel files with display of rows, columns, names, and preview. Handles invalid formats gracefully.
- **Target Selection**: Users choose a target column; app auto-selects numeric features for processing.
- **Data Preprocessing**: Options for Standardization (StandardScaler) or Normalization (MinMaxScaler) on numeric data.
- **Train-Test Split**: Adjustable split ratio (10-50%) with display of set sizes.
- **Model Selection**: Choose Logistic Regression or Decision Tree Classifier.
- **Results Visualization**: Shows execution status, accuracy, and confusion matrix heatmap (for up to 10 classes).
- **User-Friendly UI**: Step-based flow with success/error messages, spinners, and no code required.

## Technologies Used
- **Programming Languages**: Python
- **Libraries**: Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Models**: Logistic Regression, Decision Tree Classifier
- **Tools**: Visual Studio Code, GitHub

## Step by Step procedure to run this project

  ### Install the requirements
      pip install -r requirements.txt
  ### To run the streamlit file,  Use
      python3 -m streamlit run app.py
