class HealthRecommender:
    def __init__(self):
        # Base templates for recommendations mapping feature names to tips and logic
        pass

    def generate_recommendations(self, patient_data, importance_df, prediction):
        """
        Generates personalized health recommendations using hybrid logic (rules + SHAP).
        """
        recommendations = []
        
        # Extract SHAP data into a quick lookup dictionary for positive contributors
        shap_contributions = {}
        for _, row in importance_df.iterrows():
            if row['Importance'] > 0:
                shap_contributions[row['Feature']] = row['Importance']
                
        # 1. Blood Pressure Rule
        bp = patient_data.get('trestbps', 0)
        bp_shap = shap_contributions.get('trestbps', 0)
        if bp > 130 or bp_shap > 0:
            reason = "High resting blood pressure strongly contributed to your risk profile." if bp_shap > 0 else "Your blood pressure is slightly elevated above the ideal threshold."
            recommendations.append({
                "Tip": "Reduce daily salt intake and incorporate relaxing activities to manage blood pressure.",
                "Reason": reason
            })
            
        # 2. Cholesterol Rule
        chol = patient_data.get('chol', 0)
        chol_shap = shap_contributions.get('chol', 0)
        if chol > 200 or chol_shap > 0:
            reason = "Elevated cholesterol levels significantly pushed your risk higher." if chol_shap > 0 else "Your serum cholesterol indicates potential buildup risks."
            recommendations.append({
                "Tip": "Avoid saturated fats and processed foods; prioritize a high-fiber diet with oats and vegetables.",
                "Reason": reason
            })
            
        # 3. Fasting Blood Sugar
        fbs = patient_data.get('fbs', 0)
        fbs_shap = shap_contributions.get('fbs', 0)
        if fbs == 1 or fbs_shap > 0:
            reason = "High fasting blood sugar (glucose) was a key factor increasing cardiovascular risk." if fbs_shap > 0 else "Your fasting blood sugar is elevated."
            recommendations.append({
                "Tip": "Limit intake of sugary drinks and refined carbs. Consider monitoring your glucose levels.",
                "Reason": reason
            })
            
        # 4. BMI (Using the mock UI input if passed, otherwise default logic)
        bmi = patient_data.get('BMI', 22.0)
        if isinstance(bmi, str):
            try:
                bmi = float(bmi)
            except:
                bmi = 22.0
                
        if bmi >= 25.0:
            recommendations.append({
                "Tip": "Aim for a caloric deficit and 30 minutes of moderate exercise daily for weight management.",
                "Reason": "A higher Body Mass Index (BMI) creates additional systemic cardiovascular strain."
            })
            
        # 5. Heart Rate Capability (thalach)
        thalach = patient_data.get('thalach', 150)
        thalach_shap = shap_contributions.get('thalach', 0)
        if thalach < 120 or thalach_shap > 0:
            reason = "A lower maximum achieved heart rate impacted the model's confidence in your risk." if thalach_shap > 0 else "Your maximum heart rate capacity appears lower than average."
            recommendations.append({
                "Tip": "Engage in gradual, light aerobic exercises (like brisk walking or swimming) to improve cardiovascular stamina.",
                "Reason": reason
            })
            
        # 6. Age generic advice if age > 60 and high risk
        age = patient_data.get('age', 50)
        age_shap = shap_contributions.get('age', 0)
        if (age > 60 and prediction == 1) or age_shap > 0:
            recommendations.append({
                "Tip": "Schedule routine cardiovascular check-ups and follow physician guidance closely.",
                "Reason": "Advanced age naturally increases vulnerability to heart complications, making monitoring essential."
            })

        # Add a default positive reinforcement if risk is low and no major flags triggered
        if prediction == 0 and len(recommendations) <= 1:
            recommendations.append({
                "Tip": "Maintain your current healthy lifestyle, prioritize sleep, and stay hydrated.",
                "Reason": "The model found very few risk indicators, indicating good overall heart health."
            })
            
        return recommendations
