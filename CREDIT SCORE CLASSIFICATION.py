import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from fpdf import FPDF
data = pd.DataFrame({
    'Age': [25, 45, 35, 50, 23],
    'Income': [50000, 80000, 60000, 90000, 45000],
    'CreditScore': [1, 0, 1, 0, 1]   
})
X = data.drop('CreditScore', axis=1)
y = data['CreditScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("Classification Report:\n", report)
joblib.dump(model, 'credit_score_classifier.pkl')
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Credit Score Classification Report', 0, 1, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()
pdf = PDF()
pdf.add_page()
pdf.chapter_title('Model Performance')
pdf.chapter_body(f'Accuracy Score: {accuracy}\n\nClassification Report:\n{report}')
pdf.output('credit_score_classification_report.pdf')
