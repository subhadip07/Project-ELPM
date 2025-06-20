import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import io
import base64

def display_association_rules(df, min_support=0.05, min_confidence=0.3, min_lift=0.2): #top_n_rules=20):
    """
    Displays association rules to the user in a readable format.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
        min_lift (float): Minimum lift threshold.
    """
    try:
        for col in df.columns:
          if pd.api.types.is_numeric_dtype(df[col]):  # Check if numeric
            df[col] = df[col].astype('category')  # Convert to categorical

        df_encoded = pd.get_dummies(df)
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]

        st.subheader("Generated Association Rules")

        if not rules.empty:
            with st.expander("Association Rule Metrics Explained"):  # Info button
                # ... (Explanations of metrics )
                st.write("""
            Here's an explanation of the metrics shown in the table:

            * **Antecedents:** The items that are found together in a transaction (the "if" part of a rule).  For example, `(feature_0_1, feature_2_3)` means that feature_0 has value 1 AND feature_2 has value 3.

            * **Consequents:** The items that are likely to be present when the antecedents are present (the "then" part of a rule). For example, `(feature_1_5)` means that feature_1 has value 5.

            * **Support:** The percentage of transactions in the dataset that contain *both* the antecedents and the consequents.  A higher support means the rule applies to a larger portion of the data.

            * **Confidence:** The percentage of transactions that contain the antecedents that *also* contain the consequents.  It's a measure of how often the consequents are found in transactions that contain the antecedents.  A higher confidence means the rule is more reliable.

            * **Lift:**  A measure of how much more likely the antecedents and consequents are to occur together than if they were statistically independent.  A lift greater than 1 indicates a positive association.  A lift close to 1 means there's little to no association. A higher lift indicates a stronger association.
                        """)
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  # Display table

            # --- Sortable Table ---
            st.subheader("Association Rules (Sortable)")

            # Create a copy to avoid modifying the original DataFrame
            rules_display = rules.copy()

            # Allow sorting by different columns
            sort_by = st.selectbox("Sort By", ["lift", "confidence", "support"], index=0) #Default is lift
            ascending = st.checkbox("Ascending", value=False) #Default is descending

            rules_display = rules_display.sort_values(by=sort_by, ascending=ascending)

            st.dataframe(rules_display)  # Use st.dataframe for interactive table


            # --- Download Button (CSV - All Rules) ---
            st.subheader("Download All Association Rules (CSV)")

            csv_buffer = io.StringIO()  # Use StringIO for creating the CSV string
            rules.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()  # Get the string value from StringIO

            b64 = base64.b64encode(csv_string.encode()).decode()  # Encode to base64
            href = f'<a href="data:file/csv;base64,{b64}" download="all_association_rules.csv">Download</a>'
            st.markdown(href, unsafe_allow_html=True) #Use markdown to create download link


            
            top_10_rules = rules.nlargest(10, 'lift')  
            st.subheader("Top 10 Associated Features Based on Lift value: ")
            if not top_10_rules.empty: #Check if there are any top 10 rules
                for index, row in top_10_rules.iterrows(): #Iterate to top 10 rules
                    antecedents = ", ".join([str(x) for x in row['antecedents']])
                    consequents = ", ".join([str(x) for x in row['consequents']])
                    st.write(f"**If** {antecedents}  **then** {consequents}")
                    st.write(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
                    st.write("---")

        else:
            st.info("No association rules found with the selected parameters.")

    except Exception as e:
        st.error(f"Error processing dataset: {e}")


def association():
    st.title("Association Rule Mining")
    st.write("Association analysis is a data mining technique used to find the probability of the co-occurrence of items in a collection. It is used to identify patterns within data based on the concept of strong association. It is used to find the likelihood of relationships between products or events. The most common application of association analysis is in market basket analysis.")
    st.write("In this section, we will perform association analysis on the dataset to find the most frequent itemsets and association rules.")
    st.write("The dataset used for this analysis is the same dataset used for classification and prediction.")
    
    # Check if DataFrame is loaded
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Home section.")
        return
    
    df = st.session_state.df.copy()
    
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Convert categorical data to binary format (One-Hot Encoding)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Check if numeric
            df[col] = df[col].astype('category')  # Convert to categorical

    # --- Set parameter values directly ---
    min_support = 0.05  
    min_confidence = 0.5  
    min_lift = 1.0       
    
    
    display_association_rules(df, min_support, min_confidence, min_lift) # Call the function to display association rules

    
    