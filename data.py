import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PsychiatricDataAnalytics:
    """
    A class for processing and analyzing psychiatric patient data
    to derive insights on treatment effectiveness, patient outcomes,
    and research support.
    """
 
    def __init__(self, csv_path):
        """Load patient data from CSV"""
        self.csv_path = csv_path
        self.patients_df = None
        
    def load_sample_data(self):
        print("Loading data...")
        self.patients_df = pd.read_csv(self.csv_path)
        print("Original data loaded:")
        print(self.patients_df.head(5))

        # Delete all columns that contain the words 'delta', 'alpha', 'theta', or 'gamma'
        keywords = ['delta', 'alpha', 'theta', 'gamma', 'highbeta', 'beta', '122']
        drop_cols = [col for col in self.patients_df if any(kw in col.lower() for kw in keywords)]
        # self.patients_df.drop(columns=drop_cols, inplace=True)
        print("Cleaned data")
        cleaned_df = self.patients_df.drop(columns=drop_cols)
        # Replace NaN with No data
        filled_na = cleaned_df.fillna("No value provided")
        self.patients_df = filled_na
        
        print(self.patients_df.head(5))
        

        # Show remaining columns

        print("Remaining columns :", self.patients_df.columns.tolist())

    
    def analyze_diagnosis_distribution(self):
        """Analyze distribution of specific disorder"""
        # Visually representing the disorder distribution:
        diagnosis_counts = self.patients_df["specific.disorder"].value_counts().reset_index()
        diagnosis_counts.columns = ["specific.disorder", "count"]

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x="specific.disorder", y="count", data=diagnosis_counts)
        plt.title("Distribution of Specific Disorder")
        plt.xticks(rotation=45)
        plt.savefig("diagnosis_distribution.png")
        plt.tight_layout()
        plt.show()

        return diagnosis_counts
    
    def analyze_diagnosis_per_age(self):
        """Analyze distribution of specific disorder"""
        # Visually representing the disorder distribution:
        diagnosis_age = self.patients_df.groupby("specific.disorder")["age"].mean().reset_index()
        diagnosis_age.columns = ["specific.disorder", "avg_age"]

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x="specific.disorder", y="avg_age", data=diagnosis_age)
        plt.title("Distribution of Specific Disorder per Age")
        plt.xticks(rotation=45)
        plt.savefig("diagnosis_per_age.png")
        plt.tight_layout()
        plt.show()

        return diagnosis_age
    
    def analyze_disorder_by_iq(self):
        # Ensure IQ is numeric
        self.patients_df["IQ"] = pd.to_numeric(self.patients_df["IQ"], errors="coerce")

        disorder_by_iq = self.patients_df.groupby("specific.disorder")["IQ"].mean().sort_values(ascending=False).reset_index()
        # disorder_by_iq.columns = ["specific.disorder", "IQ"]

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x="specific.disorder", y="IQ", data=disorder_by_iq)
        plt.title("Distribution of Specific Disorder per IQ")
        plt.xticks(rotation=45)
        plt.savefig("diagnosis_per_IQ.png")
        plt.tight_layout()
        plt.show()

        return disorder_by_iq
    
    def run_complete_analysis(self):
        """Execute a complete analysis pipeline"""
        self.load_sample_data()
        
        diagnosis_dist = self.analyze_diagnosis_distribution()
        diagnosis_per_age = self.analyze_diagnosis_per_age()
        iq_by_disorder = self.analyze_disorder_by_iq()
        
        return {
            "diagnosis_distribution": diagnosis_dist,
            "diagnosis per age": diagnosis_per_age,
            "disorder_by_iq": disorder_by_iq
        }

if __name__ == "__main__":
    analytics = PsychiatricDataAnalytics("EEG.machinelearing_data_BRMH.csv")
    results = analytics.run_complete_analysis()
    print("Analysis complete")