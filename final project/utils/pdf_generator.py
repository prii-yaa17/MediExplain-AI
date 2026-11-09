from fpdf import FPDF
from datetime import datetime
import tempfile
from PIL import Image

class PDFReportGenerator(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(0, 51, 102) # Dark blue
        self.cell(0, 10, 'Explainable AI-Based Medical Diagnosis System', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Report Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 10, f'  {title}', 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, text)
        self.ln()

    def add_patient_details(self, details_dict):
        self.chapter_title('Patient Input Details')
        self.set_font('Arial', '', 11)
        for key, value in details_dict.items():
            self.cell(60, 8, f"{key}:", border=1)
            self.cell(130, 8, f"{value}", border=1, ln=1)
        self.ln(5)

    def add_prediction_result(self, risk_label, probability):
        self.chapter_title('Diagnosis Prediction')
        self.set_font('Arial', 'B', 14)
        
        if risk_label.lower() == 'high':
            self.set_text_color(204, 0, 0) # Red
        else:
            self.set_text_color(0, 153, 0) # Green
            
        self.cell(0, 10, f"Disease Risk: {risk_label} ({probability:.1f}%)", 0, 1, 'L')
        self.set_text_color(0, 0, 0) # Reset color
        self.ln(5)

    def add_explanation(self, text):
        self.chapter_title('AI Explanation Summary')
        self.chapter_body(text)
        
    def add_recommendations(self, recommendations):
        self.chapter_title('Health Recommendations with Reasons')
        for i, rec in enumerate(recommendations, 1):
            self.set_x(10) # Explicit margin reset
            self.set_font('Arial', 'B', 11)
            self.set_text_color(0, 102, 204) # Blue shade for tips
            self.multi_cell(0, 8, f"Tip {i}: {rec['Tip']}")
            
            self.set_x(10) # Explicit margin reset
            self.set_font('Arial', 'I', 11)
            self.set_text_color(100, 100, 100) # Gray for reasons
            self.multi_cell(0, 8, f"Reason: {rec['Reason']}")
            self.ln(4)
        self.set_text_color(0, 0, 0) # Reset color

    def add_plot(self, plot_fig):
        """Saves a matplotlib fig to a tempfile and injects it into PDF."""
        self.chapter_title('Feature Contribution Breakdown')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plot_fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
            
        # We assume A4 width is 210mm. Subtract margins.
        self.image(tmpfile.name, x=15, w=180)
        self.ln(5)


def create_pdf_report(patient_data, risk_label, probability, explanation_text, plot_fig=None, recommendations=None):
    """
    Orchestrates the creation of the diagnostic PDF.
    Returns: The raw bytes of the PDF.
    """
    pdf = PDFReportGenerator()
    pdf.add_page()
    
    # 1. Add Input Details
    pdf.add_patient_details(patient_data)
    
    # 2. Add Result
    pdf.add_prediction_result(risk_label, probability)
    
    # 3. Add NLP explanation
    pdf.add_explanation(explanation_text)
    
    # Add Recommendations
    if recommendations:
        pdf.add_recommendations(recommendations)
    
    # 4. Add the SHAP graph
    if plot_fig is not None:
        pdf.add_plot(plot_fig)
        
    # fpdf2 returns a bytearray. Streamlit download_button explicitly requires bytes.
    return bytes(pdf.output())
