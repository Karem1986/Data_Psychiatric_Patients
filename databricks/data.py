from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum, when, datediff, current_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PsychiatricDataAnalytics:
    """
    A class for processing and analyzing psychiatric patient data
    to derive insights on treatment effectiveness, patient outcomes,
    and research support.
    """
    
    def __init__(self, spark=None):
        """Initialize the Spark session if not provided"""
        if not spark:
            self.spark = SparkSession.builder \
                .appName("PsychiatricDataAnalytics") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark
            
    def define_schemas(self):
        """Define the schemas for various data sources"""
        # Patient demographic schema
        self.patient_schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("age", IntegerType(), True),
            StructField("gender", StringType(), True),
            StructField("diagnosis_primary", StringType(), True),
            StructField("diagnosis_secondary", ArrayType(StringType()), True),
            StructField("first_visit_date", DateType(), True)
        ])
        
        # Treatment session schema
        self.session_schema = StructType([
            StructField("session_id", StringType(), False),
            StructField("patient_id", StringType(), False),
            StructField("provider_id", StringType(), False),
            StructField("session_date", DateType(), False),
            StructField("session_type", StringType(), True),
            StructField("session_notes", StringType(), True),
            StructField("treatment_approach", StringType(), True)
        ])
        
        # Medication schema
        self.medication_schema = StructType([
            StructField("prescription_id", StringType(), False),
            StructField("patient_id", StringType(), False),
            StructField("medication_name", StringType(), False),
            StructField("dosage", StringType(), True),
            StructField("prescribed_date", DateType(), False),
            StructField("end_date", DateType(), True)
        ])
        
        # Assessment schema
        self.assessment_schema = StructType([
            StructField("assessment_id", StringType(), False),
            StructField("patient_id", StringType(), False),
            StructField("assessment_type", StringType(), False),
            StructField("assessment_date", DateType(), False),
            StructField("score", FloatType(), True),
            StructField("assessment_notes", StringType(), True)
        ])
        
    def load_sample_data(self):
        """Load sample data for demonstration purposes"""
        # Sample patient data---After testing USE REAL DATA FROM API or DATABASE
        patient_data = [
            ("P001", 34, "F", "Major Depressive Disorder", ["Generalized Anxiety"], "2022-01-15"),
            ("P002", 28, "M", "Bipolar Disorder", ["Insomnia"], "2022-02-20"),
            ("P003", 45, "F", "Post-Traumatic Stress Disorder", ["Depression"], "2022-01-05"),
            ("P004", 19, "M", "Attention Deficit Hyperactivity Disorder", ["Anxiety"], "2022-03-10"),
            ("P005", 52, "F", "Generalized Anxiety Disorder", ["Insomnia", "Depression"], "2022-02-01")
        ]
        
        #  In Databrics community edition, the data is loaded from a CSV or JSON file
        #  Use the "EEG_machinelearing_data_BRMH-2.csv" file if you decide to test this with 
        #  the Databricks community edition
        
        
        # Convert to DataFrame
        self.patients_df = self.spark.createDataFrame(patient_data, 
            ["patient_id", "age", "gender", "diagnosis_primary", "diagnosis_secondary", "first_visit_date"])

    def analyze_diagnosis_distribution(self):
        """Analyze distribution of primary diagnoses"""
        diagnosis_counts = self.patients_df.groupBy("diagnosis_primary") \
            .count() \
            .orderBy(col("count").desc())
        
        # Convert to Pandas for visualization
        diagnosis_pd = diagnosis_counts.toPandas()
    
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x="diagnosis_primary", y="count", data=diagnosis_pd)
        plt.title("Distribution of Primary Diagnoses")
        plt.xticks(rotation=45)
        plt.tight_layout()
    
        # Display the plot when running locally
        # In Databricks this would be handled by display()
        plt.savefig("diagnosis_distribution.png")  # Save the plot as an image file
    
        # Only call plt.show() when not running in Databricks
        try:
            from databricks.connect import is_databricks_runtime
            if not is_databricks_runtime():
                plt.show()
        except ImportError:
            plt.show()  # Not running in Databricks environment
    
        return diagnosis_counts
    
    def analyze_treatment_effectiveness(self, sessions_df, assessments_df):
        """
        Analyze treatment effectiveness by comparing assessment scores
        before and after specific treatment approaches
        """
        # Join sessions with assessments
        treatment_results = sessions_df.join(
            assessments_df,
            (sessions_df.patient_id == assessments_df.patient_id) &
            (sessions_df.session_date <= assessments_df.assessment_date),
            "inner"
        )
        
        # Group by treatment approach and calculate average assessment scores
        effectiveness_by_treatment = treatment_results.groupBy("treatment_approach") \
            .agg(
                avg("score").alias("avg_score"),
                count("assessment_id").alias("assessment_count")
            ) \
            .orderBy(col("avg_score").desc())
            
        return effectiveness_by_treatment

    def identify_medication_patterns(self, medication_df):
        """
        Identify patterns in medication prescriptions and their associations
        with specific diagnoses
        """
        # Join medication data with patient data
        med_patient_df = medication_df.join(
            self.patients_df,
            "patient_id",
            "inner"
        )
        
        # Analyze medication frequency by diagnosis
        med_by_diagnosis = med_patient_df.groupBy("diagnosis_primary", "medication_name") \
            .count() \
            .orderBy(col("diagnosis_primary"), col("count").desc())
            
        # Calculate medication duration
        med_duration_df = medication_df.withColumn(
            "duration_days",
            when(col("end_date").isNotNull(), 
                 datediff(col("end_date"), col("prescribed_date"))
            ).otherwise(
                datediff(current_date(), col("prescribed_date"))
            )
        )
        
        return med_by_diagnosis, med_duration_df
    
    def generate_provider_insights_dashboard(self, sessions_df, assessments_df):
        """
        Generate insights dashboard for psychiatric providers
        showing patient progress and treatment effectiveness
        """
        # Join data
        provider_insights = sessions_df.join(
            assessments_df, 
            (sessions_df.patient_id == assessments_df.patient_id),
            "inner"
        ).join(
            self.patients_df,
            "patient_id",
            "inner"
        )
        
        # Calculate metrics by provider
        provider_metrics = provider_insights.groupBy("provider_id") \
            .agg(
                count("distinct patient_id").alias("patient_count"),
                avg("score").alias("avg_assessment_score"),
                count("session_id").alias("session_count")
            )
            
        return provider_metrics
    
    def run_complete_analysis(self):
        """Execute a complete analysis pipeline"""
        self.define_schemas()
        self.load_sample_data()
        
        # In a real implementation, we would load actual data and run all analyses
        diagnosis_dist = self.analyze_diagnosis_distribution()
        
        return {
            "diagnosis_distribution": diagnosis_dist
        }


if __name__ == "__main__":
    # For testing in local environment
    analytics = PsychiatricDataAnalytics()
    results = analytics.run_complete_analysis()
    print("Analysis complete")