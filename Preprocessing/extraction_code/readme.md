1. Run find_group.ipynb
    This notebook will generate the outcome.csv file used in later steps.

    It identifies the outcome (MCI-AD diagnosis) for each patient using data from the diagnosis and demographic tables.

    Note: In previous datasets, diagnosis information was distributed across the condition and observation tables. Please update the code accordingly if this applies to your current dataset.

2. Run feature_extraction.ipynb
    Follow the subsections in the notebook to extract domain-specific features:

    2.1 get_location (optional)
        Skip this section if location information is not needed.

        Consider using the cells under the "resample" section if needed.

    2.2 get_sub_sample
        Extracts case and control sub-datasets based on outcome.

        Generates two folders: case_all and control_all, storing relevant tables for each group.

    2.3 get_notes (optional)
        Extracts notes data for case and control groups.

        Skip this section if your dataset does not contain notes.

    2.4 extract_features
        Identifies outcomes for the extracted case and control populations.

        Previous experimental setup:

        Case group: Patients with MCI+AD or AD.

        Control group: Patients with MCI.

        Assigns outcome windows (1, 3, and 5 years) based on each patient's index date.

        Update this logic if your experiment uses different outcome definitions.

        Outputs baseline feature files for both case and control groups.

    2.5 get_demog
        Generates demographic feature files for both groups.

    2.6 get_m2
        Creates yearly m2 feature files for case and control groups.

        After generating features:

        Perform 1:1 propensity score matching using the following covariates: birth_year, gender_source_value, race_ethnicity, and cci_score.

        Reset the control group's index date to the diagnosis date closest to the matched case's index date.

        Re-extract yearly m2 features for the updated control group.

    2.7 get_m3_measurement
        Generates m3 features for both groups by year.

        Skip OMOP-LOINC code mapping if using PCORnet-formatted data.

    2.8 get_m5 (optional)
        Generates m5 note-based features.

        Skip if your dataset does not include notes.

    2.9 get_m4
        Generates m4 features (e.g., medications) for each year.

        Skip OMOP-RxNorm mapping if using PCORnet-formatted data.

    2.10 concat
        Combines all processed files to generate the final matched case-control dataset for each year.

