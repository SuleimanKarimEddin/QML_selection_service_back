# Import necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from dwave.plugins.sklearn import SelectFromQuadraticModel
from fpdf import FPDF
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import tempfile

app = FastAPI()

def generate_feature_selection_report(data, target_column_name):
    X = data.drop(columns=[target_column_name])
    y = data[target_column_name]    
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    quantum_selector = SelectFromQuadraticModel(num_features=6, time_limit=5)
    X_train_quantum = quantum_selector.fit_transform(X_train, y_train)
    X_test_quantum = quantum_selector.transform(X_test)
    quantum_selected_features = X.columns[quantum_selector.get_support()]
    rfe_selector = RFE(RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=6, step=1)
    X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
    X_test_rfe = rfe_selector.transform(X_test)
    rfe_selected_features = X.columns[rfe_selector.get_support()]

    def evaluate_classifier(X_train, X_test, y_train, y_test, method, selected_features):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        report = {
            "Method": method,
            "Selected Features": selected_features,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred)
        }

        return report

    quantum_results = evaluate_classifier(X_train_quantum, X_test_quantum, y_train, y_test, "RFE", quantum_selected_features)
    rfe_results = evaluate_classifier(X_train_rfe, X_test_rfe, y_train, y_test, "Quantum Annealing", rfe_selected_features)
    results = pd.DataFrame({
        'Method': ['RFE', 'Quantum Annealing'],
        'Accuracy': [quantum_results["Accuracy"], rfe_results["Accuracy"]],
        'Precision': [quantum_results["Precision"], rfe_results["Precision"]],
        'Recall': [quantum_results["Recall"], rfe_results["Recall"]],
        'F1-score': [quantum_results["F1-score"], rfe_results["F1-score"]]
    })

    # Plotting the results using seaborn
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.barplot(x="Method", y="Accuracy", data=results, ax=axes[0, 0])
    axes[0, 0].set_title("Accuracy Comparison")

    sns.barplot(x="Method", y="Precision", data=results, ax=axes[0, 1])
    axes[0, 1].set_title("Precision Comparison")

    sns.barplot(x="Method", y="Recall", data=results, ax=axes[1, 0])
    axes[1, 0].set_title("Recall Comparison")

    sns.barplot(x="Method", y="F1-score", data=results, ax=axes[1, 1])
    axes[1, 1].set_title("F1-score Comparison")

    plt.tight_layout()
    plt.savefig("feature_selection_comparison.png")
    plt.show()

    # Create PDF report
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Feature Selection Methods Comparison Report', 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(10)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

        def add_image(self, image_path):
            self.image(image_path, x=10, y=None, w=190)

    pdf = PDF()
    pdf.add_page()

    # Add dataset information
    pdf.chapter_title('Dataset Information')
    pdf.chapter_body(f'Dataset shape: {X.shape}\nNumber of classes: {len(np.unique(y))}\n')

    # Add results for each method
    methods = [rfe_results, quantum_results]
    for method in methods:
        pdf.chapter_title(f'{method["Method"]} Method')
        body = (
            f'Selected Features: {", ".join(method["Selected Features"])}\n'
            f'Accuracy: {method["Accuracy"]}\n'
            f'Precision: {method["Precision"]}\n'
            f'Recall: {method["Recall"]}\n'
            f'F1-score: {method["F1-score"]}\n'
            f'Confusion Matrix:\n{method["Confusion Matrix"]}\n'
            f'Classification Report:\n{method["Classification Report"]}\n'
        )
        pdf.chapter_body(body)

    # Add comparison plots
    pdf.chapter_title('Comparison Plots')
    pdf.add_image('feature_selection_comparison.png')



    # Save PDF
    pdf_file_path = 'QFSS-Report.pdf'
    pdf.output(pdf_file_path)

    return pdf_file_path



def generate_empty_pdf(file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.output(file_path)

@app.post("/pdf/")
async def create_upload_file(file: UploadFile = File(...), target_column_name: str = Form(...)):
    # Read the uploaded CSV file into a pandas DataFrame
    data = pd.read_csv(file.file)
    
    # Create a temporary file to store the empty PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf_path = temp_pdf.name
        generate_empty_pdf(temp_pdf_path)
    
    # Return the empty PDF as a FileResponse
    return FileResponse(temp_pdf_path, media_type='application/pdf', filename='feature_selection_report.pdf')



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), target_column_name: str = Form(...)):
    data = pd.read_csv(file.file)
    pdf_file_path = generate_feature_selection_report(data, target_column_name)
    return FileResponse(pdf_file_path, media_type='application/pdf', filename='feature_selection_report.pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)