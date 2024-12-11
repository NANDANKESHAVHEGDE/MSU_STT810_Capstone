# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import gdown
import zipfile
import io
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from datetime import datetime
import pytz
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="E-commerce Recommendation Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Step 1: Load Data and Preprocessing ---
@st.cache_data(show_spinner=False, persist=False)
def load_data():
    """
    Downloads and loads the data directly from Google Drive without saving to disk.

    Returns:
    - DataFrame: Loaded and preprocessed data.
    """
    # Google Drive File ID
    file_id = "1YcnadUrqyq68Cag_7diw9JW9yUPDvkhr"
    url = f'https://drive.google.com/uc?id={file_id}'

    # Create an in-memory BytesIO object to hold the downloaded file
    zip_bytes = io.BytesIO()

    # Download the file into the BytesIO object
    gdown.download(url, output=zip_bytes, quiet=False, fuzzy=True)
    zip_bytes.seek(0)  # Reset pointer to the start of the file

    # Check if the downloaded file is a valid zip archive
    if zipfile.is_zipfile(zip_bytes):
        with zipfile.ZipFile(zip_bytes, 'r') as z:
            # Get the first file in the zip archive (assumes there's only one)
            pickle_filename = z.namelist()[0]
            with z.open(pickle_filename) as pickle_file:
                # Load the pickle file directly from the zip archive
                data = pickle.load(pickle_file)
                return data
    else:
        st.error("The downloaded file is not a valid zip file.")
        return None

data = load_data()

# Define the base directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust relative to your `app` folder

# Build paths dynamically
price_image = BASE_DIR / "Images" / "Data_prep2.PNG"
event_types_image = BASE_DIR / "Images" / "Data_prep3.PNG"
pmf_image= BASE_DIR / "Images" / "Data_prep1.PNG"
hpt_image1= BASE_DIR / "Images" / "Hypothesis_testing1.PNG"
hpt_image2= BASE_DIR / "Images" / "Hypothesis_testing2.PNG"
hpt_image3= BASE_DIR / "Images" / "Hypothesis_testing3.PNG"
by_image1= BASE_DIR / "Images" / "Bayesian_approach1.PNG"
by_image2 = BASE_DIR / "Images" / "Bayesian_approach2.PNG"

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Overview", "Data Preparation", "EDA", "Hypothesis Testing","Recommendations - Bayesian approach","Price Analysis", "Recommendations - Frequentist approach"]
selected_page = st.sidebar.radio("Go to", pages)

# Custom CSS styling for better theme
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        font-family: Arial, sans-serif;
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2a9d8f;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        font-weight: normal;
        color: #264653;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 18px;
    }
    .metric-title {
        font-weight: bold;
        font-size: 20px;
        color: #264653;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #e76f51;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Overview Page
if selected_page == "Overview":
    st.markdown("<div class='main-header'>E-commerce Recommendation System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>"
        "An advanced recommendation engine developed as part of the Statistics 810 capstone project at Michigan State University. "
        "This system leverages user interaction data to provide personalized product suggestions using statistical methods and machine learning."
        "</div>",
        unsafe_allow_html=True,
    )

    st.header("Project Highlights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='metric-box'>
                <div class='metric-title'>Project Focus</div>
                <div>Recommendation Models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='metric-box'>
                <div class='metric-title'>Key Techniques</div>
                <div>Collaborative Filtering, Bayesian Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.header("Project Objectives")
    st.markdown(
        """
        - **Data Preprocessing:** Clean, standardize, and enrich user interaction data.
        - **Recommendation Model Development:** Implement Hypothesis testing, Bayesian calculation, collaborative filtering, hybrid models, and multivariate statistics.
        - **Feature Engineering:** Derive insights using temporal features, session behavior, and user preferences.
        - **Dashboard Implementation:** Visualize recommendations and key metrics interactively.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 16px;'>
        Developed as part of Statistics 810 Capstone, Michigan State University.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add a link to the GitHub repository
    st.markdown(
        """
        <div style='text-align: center; font-size: 16px; margin-top: 20px;'>
        <a href='https://github.com/NANDANKESHAVHEGDE/MSU_STT810_Capstone' target='_blank' style='text-decoration: none; color: #2a9d8f;'>
        Explore the Project on GitHub
        </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Data Preparation Page with Styling
elif selected_page == "Data Preparation":
    # Page Title
    st.markdown("<div class='main-header'>Data Preparation</div>", unsafe_allow_html=True)
    
    # Description of the Process
    st.markdown(
        """
        <div class='sub-header'>
        The dataset, originally comprising 20 million records, was carefully reduced to 1 million records using stratified sampling. 
        This ensures data integrity while retaining critical behavioral patterns for recommendation modeling.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bullet Points: Data Preparation Summary
    st.markdown("<div class='sub-header'>Steps Followed in Data Preparation</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <ul style='line-height: 1.8; font-size: 16px; color: #264653;'>
            <li><b>Stratified Sampling:</b> Users were sampled to retain their complete behavior (all events) rather than random sampling.</li>
            <li><b>Proportional Distributions:</b> Key distributions of event types, categories, and user interactions were preserved to ensure representativeness.</li>
            <li><b>Focus on Behavioral Trends:</b> By sampling complete user profiles, critical behavioral insights were maintained for recommendation modeling.</li>
            <li><b>Aligned Temporal and Categorical Features:</b> Temporal patterns and categorical attributes remain consistent with the original dataset.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    # Visualization 1: Distribution of Price
    st.markdown("<div class='sub-header'>Distribution of Price</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 16px; line-height: 1.6; color: #264653;'>
        This visualization compares the price distributions between the original (20M records) and sampled (1M records) datasets:
        <ul>
            <li>The KDE (Kernel Density Estimate) lines for both datasets align closely, indicating that the price distribution is well-preserved.</li>
            <li>No significant skewness or loss of variability is observed in the sampled dataset.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display Price Distribution Image
    price_image = Image.open(price_image) # Replace with the correct path
    st.image(price_image, caption="Price Distribution (Original vs. Sampled)", use_column_width=True)

    # Visualization 2: Distribution of Event Types
    st.markdown("<div class='sub-header'>Distribution of Event Types</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 16px; line-height: 1.6; color: #264653;'>
        The chart illustrates the proportion of different event types (e.g., view, cart, purchase) between the datasets:
        <ul>
            <li>Both datasets have near-identical proportions for event types, confirming that stratified sampling maintained interaction trends.</li>
            <li>For example, <b>view</b> events dominate (~40%), while <b>purchase</b> events form a smaller but critical proportion.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display Event Types Distribution Image
    event_types_image =Image.open(event_types_image) # Replace with the correct path
    st.image(event_types_image, caption="Event Types Distribution (Original vs. Sampled)", use_column_width=True)

    # Visualization 3: PMF of User Activity
    st.markdown("<div class='sub-header'>PMF of User Activity</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 16px; line-height: 1.6; color: #264653;'>
        This PMF (Probability Mass Function) shows the proportion of users based on their event counts:
        <ul>
            <li>The alignment between the original and sampled datasets demonstrates that user activity patterns are preserved.</li>
            <li>For instance, a significant proportion of users (~40%) interacted only once, as indicated by the tall bar for "1 event."</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display PMF of User Activity Image
    pmf_image = Image.open(pmf_image) # Replace with the correct path
    st.image(pmf_image, caption="PMF of User Activity (Original vs. Sampled)", use_column_width=True)

    # Concluding Note
    st.markdown("<div class='sub-header'>Key Takeaways</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 16px; line-height: 1.6; color: #264653;'>
        <ul>
            <li>The stratified sampling approach ensures that the 1M record dataset faithfully represents the original 20M dataset.</li>
            <li>Complete user behavior profiles and proportional distributions are preserved for robust recommendation modeling.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif selected_page == "EDA":
    # Display header
    st.markdown("<div class='main-header'>Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)
    st.write("Loading data...")

    # Load the data using the updated function
    try:
        data = load_data()
        if data is not None:
            st.write("Data loaded successfully!")

            # Visualization 1: Distribution of Event Types
            st.markdown("<div class='sub-header'>1. Distribution of Event Types</div>", unsafe_allow_html=True)
            event_types_fig = plt.figure(figsize=(8, 5))
            sns.countplot(x='event_type', data=data, palette='viridis')
            plt.title("Distribution of Event Types")
            plt.xlabel("Event Type")
            plt.ylabel("Count")
            st.pyplot(event_types_fig)
            st.markdown(
                """
                - **Insight**:
                - This plot provides an overview of user activities, such as views, purchases, or cart additions.
                - The dominance of certain event types (e.g., views) may suggest areas where users are more engaged.
                - A balanced mix of events indicates healthy interaction patterns, while skewed distributions might highlight bottlenecks.
                """
            )

            # Visualization 2: Distribution of Prices
            st.markdown("<div class='sub-header'>2. Distribution of Prices</div>", unsafe_allow_html=True)
            price_fig = plt.figure(figsize=(8, 5))
            sns.histplot(data['price'], kde=True, bins=30, color='blue')
            plt.title("Distribution of Prices")
            plt.xlabel("Price")
            plt.ylabel("Density")
            st.pyplot(price_fig)
            st.markdown(
                """
                - **Insight**:
                - The price distribution helps identify popular price points for products.
                - Peaks in the distribution indicate the price ranges where most products fall, which can guide pricing strategies.
                - The presence of a long tail might indicate outliers or niche products with significantly higher or lower prices.
                """
            )

            # Visualization 3: Hourly Interaction Trends
            st.markdown("<div class='sub-header'>3. Hourly Interaction Trends</div>", unsafe_allow_html=True)
            hourly_trends_fig = plt.figure(figsize=(8, 5))
            hourly_trends = data.groupby('event_hour')['event_type'].count()
            sns.lineplot(x=hourly_trends.index, y=hourly_trends.values, marker='o', color='green')
            plt.title("Hourly Interaction Trends")
            plt.xlabel("Hour of the Day")
            plt.ylabel("Number of Interactions")
            st.pyplot(hourly_trends_fig)
            st.markdown(
                """
                - **Insight**:
                - Interaction trends across the day reveal the times when users are most active.
                - Peaks during specific hours (e.g., evening) may suggest opportunities to schedule promotions or notifications.
                - A consistent pattern throughout the day might indicate steady engagement, whereas variability highlights specific focus hours.
                """
            )

            # Visualization 4: Top Brands by Interaction
            st.markdown("<div class='sub-header'>4. Top Brands by Interaction</div>", unsafe_allow_html=True)
            top_brands_fig = plt.figure(figsize=(10, 6))
            top_brands = data['brand'].value_counts().head(10)
            sns.barplot(x=top_brands.values, y=top_brands.index, palette='viridis')
            plt.title("Top 10 Brands by Interaction")
            plt.xlabel("Number of Interactions")
            plt.ylabel("Brand")
            st.pyplot(top_brands_fig)
            st.markdown(
                """
                - **Insight**: 
                - Popular brands drive significant user engagement, as shown by their interaction counts.
                - This data helps identify top-performing brands and can be used for partnership or promotional strategies.
                - Brands with fewer interactions might need better visibility or targeted campaigns to boost engagement.
                """
            )

            # Visualization 5: User Activity Distribution
            st.markdown("<div class='sub-header'>5. User Activity Distribution</div>", unsafe_allow_html=True)
            user_activity_fig = plt.figure(figsize=(8, 5))
            sns.histplot(data['total_events'], bins=30, kde=False, color='orange')
            plt.title("User Activity Distribution")
            plt.xlabel("Total Events per User")
            plt.ylabel("Frequency")
            st.pyplot(user_activity_fig)
            st.markdown(
                """
                - **Insight**: 
                - The activity distribution illustrates the range of user engagement on the platform.
                - A high number of low-interaction users might suggest casual visitors, whereas high-interaction users indicate power users.
                - Tailoring strategies for these segments can help convert casual users into frequent shoppers.
                """
            )

        else:
            st.error("Failed to load data.")
    except Exception as e:
        st.error(f"Error loading data: {e}")


# Hypothesis Testing Tab
elif selected_page == "Hypothesis Testing":
    st.markdown("<div class='main-header'>Hypothesis Testing in E-commerce</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='sub-header'>
        Hypothesis testing provides a statistical foundation for understanding patterns in purchase behavior. This section includes:
        <ul>
            <li><b>One-Sample T-Test:</b> Comparison of brand-specific mean purchase values against the population.</li>
            <li><b>Two-Sample T-Test:</b> Analysis of mean purchase values between high and low premium products.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Distribution of Product Premiumness
    hpt_image3_path = Image.open(hpt_image3) # Replace with the correct path
    st.markdown("### Distribution of Product Premiumness")
    st.image(hpt_image3_path, caption="Distribution of Product Premiumness with Mean log Price")

    st.markdown(
        """
        **Insight:**  
        This visualization highlights the count of products across low, medium, and high premium categories along with their mean log price.  
        - The high premium category has the highest mean log price, as expected, while the low premium category features the most products by count.
        - Medium premium products serve as a balance between price and volume.  
        """
    )

    hpt_image1_path = Image.open(hpt_image1) # Replace with the correct path
    st.markdown("### One-Sample T-Test: Brand 'runail'")
    st.image(hpt_image1_path, caption="One-Sample T-Test: Brand 'runail'")
    
    st.markdown(
        """
        **Context:**  
        This analysis investigates whether the mean purchase value for products under the brand 'runail' differs significantly from the mean purchase value across the entire population.  

        **Results Interpretation:**  
        - **Sample Size:** The test compares a subset of 23,325 purchases for 'runail' against the population of 199,743 purchases.  
        - **Mean Purchase Values:**  
          - 'runail': 0.219121 (indicating a higher average purchase value for this brand).  
          - Population: 0.146053.  
        - **T-Statistic and P-Value:**  
          - The high t-statistic value (10.018781) and extremely small p-value (close to 0) indicate strong evidence against the null hypothesis.  
        - **Confidence Interval:**  
          - The confidence interval (-0.204826, 0.233416) does not include 0, further supporting the rejection of the null hypothesis.  

        **Conclusion:**  
        There is a statistically significant difference in purchase values for 'runail' compared to the population.  
        """
    )
    
    hpt_image2_path = Image.open(hpt_image2) # Replace with the correct path
    st.markdown("### Two-Sample T-Test: High vs. Low Premium Products")
    st.image(hpt_image2_path, caption="Two-Sample T-Test: High vs. Low Premium Products")
    
    st.markdown(
        """
        **Context:**  
        This analysis examines whether there is a significant difference in the average purchase values between high premium and low premium products.  

        **Results Interpretation:**  
        - **Sample Size:** A comparison between 219,529 purchases of high premium products and 203,578 purchases of low premium products ensures robust statistical power.  
        - **Mean Purchase Values:**  
          - High Premium: 0.062538 (lower mean purchase value).  
          - Low Premium: 0.129965 (higher mean purchase value).  
        - **T-Statistic and P-Value:**  
          - A large negative t-statistic (-74.460993) and a p-value of 0.000000 provide strong evidence to reject the null hypothesis.  
        - **Confidence Intervals:**  
          - High Premium: (0.0615, 0.0635) — narrow range, indicating precise estimates.  
          - Low Premium: (0.1284, 0.1314) — narrow range, reflecting the higher purchase value for low premium products.  
        - **Effect Size (Cohen's \(d\)):**  
          - A value of -0.229109 indicates a small to medium effect size, suggesting that the difference, while statistically significant, may not be practically large.  
        - **Power of the Test:**  
          - A power of 1.000000 confirms that the test is highly reliable in detecting the difference.  

        **Conclusion:**  
        The statistically significant difference highlights that low premium products generally have higher purchase values than high premium products.  
        """
    )

    st.markdown(
        """
        <div class='sub-header'>
        **Actionable Insights:**
        <ul>
            <li>The higher purchase values for 'runail' suggest potential for premium branding or marketing campaigns focusing on this brand.</li>
            <li>The observed preference for low premium products may indicate consumer price sensitivity, particularly for value-driven segments.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Bayesian Recommendation Tab
elif selected_page == "Recommendations - Bayesian approach":
    st.markdown("<div class='main-header'>Bayesian Approach to Recommendations</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='sub-header'>
        The Bayesian approach provides a robust probabilistic framework for recommendation systems. By incorporating prior knowledge and observed data,
        it refines the estimation of the likelihood of a user purchasing a product. Below, we demonstrate its application in the context of e-commerce recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explain the Bayesian Model
    st.markdown("### Bayesian Model Overview")
    st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")

    st.markdown(
        """
        **Explanation:**  
        The posterior probability \( P(A|B) \) represents the probability of an event \( A \) (e.g., a user purchasing a product) given evidence \( B \) 
        (e.g., cart additions or past purchases). The Bayesian formula uses:
        - \( P(A) \): The prior probability, representing the baseline likelihood of an event before observing evidence.
        - \( P(B|A) \): The likelihood, indicating how probable the evidence is if the event \( A \) is true.
        - \( P(B) \): The marginal probability of observing the evidence.

        This framework enables dynamic updates to recommendations as new data becomes available.
        """
    )

    # Display the Dataframe with Posterior Probabilities
    by_image2_path = Image.open(by_image2) # Replace with the correct path
    st.markdown("### Posterior Probability Table")
    st.image(by_image2_path, caption="Bayesian Posterior Probabilities for Recommendations")

    # Insights
    st.markdown(
        """
        **Insights from Posterior Probabilities:**  
        - The posterior probabilities provide a refined ranking of product recommendations based on user interaction data.  
        - Products with higher posterior probabilities are more likely to be purchased by the corresponding user.  
        - This approach accounts for prior behaviors (e.g., cart additions) and updates dynamically with new evidence.
        """
    )

    # Use Case Examples
    by_image1_path = Image.open(by_image1) # Replace with the correct path
    st.markdown("### Use Case Examples")
    st.image(by_image1_path, caption="Posterior Probabilities for Top Users and Products")

    st.markdown(
        """
        **Key Observations:**  
        - **Top Users:** For users with a rich history of interactions, the Bayesian model provides personalized recommendations with high confidence.
        - **Top Products:** Products frequently added to carts but rarely purchased may receive lower posterior probabilities, highlighting conversion opportunities.
        - **Example:** User 399445569 shows a 2.53\% likelihood of purchasing Product 5809910, which aligns with their interaction history.

        **Actionable Insights:**  
        - Utilize high posterior probability products for targeted promotions or personalized notifications.
        - For products with low posterior probabilities, consider strategies to improve visibility or address conversion barriers.
        """
    )

    # Conclusion Section
    st.markdown(
        """
        <div class='sub-header'>
        **Conclusion:**  
        The Bayesian approach introduces flexibility and precision into the recommendation process, enabling data-driven, probabilistic decision-making. 
        This methodology seamlessly integrates prior knowledge, observed behaviors, and dynamic updates to generate effective and scalable recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Price Sensitivity Page
elif selected_page == "Price Analysis":
    st.markdown("<div class='main-header'>Price Sensitivity Analysis</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='sub-header'>
        Understanding user sensitivity to product prices is essential for optimizing pricing strategies.
        This section explores log-normal price distributions, purchasing behavior by price categories,
        and user clustering based on price sensitivity. Additionally, a Monte Carlo simulation assesses potential revenue variability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Load and Preprocess Data ---
    st.markdown("### Log-Normal Distribution Fit for Prices")
    log_prices = np.log(data['price'] + 1)

    # Fit log-normal distribution
    shape, loc, scale = stats.lognorm.fit(log_prices, floc=0)
    x = np.linspace(log_prices.min(), log_prices.max(), 100)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)

    # Visualization
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(log_prices, bins=30, kde=False, stat='density', ax=ax1, label='Data')
    ax1.plot(x, pdf, label='Log-Normal Fit', color='red')
    ax1.set_title("Log-Normal Distribution Fit for Prices")
    ax1.set_xlabel("Log Price")
    ax1.set_ylabel("Density")
    ax1.legend()
    st.pyplot(fig1)

    st.markdown(
        """
        **Insights:**  
        - Prices follow a log-normal distribution, with most products priced in the lower range.
        - The red curve represents the fitted log-normal distribution, confirming the statistical alignment with observed data.
        """
    )

    # Empirical vs Fitted CDF
    st.markdown("### Empirical vs Fitted CDF")
    empirical_cdf = np.arange(1, len(log_prices) + 1) / len(log_prices)
    sorted_prices = np.sort(log_prices)
    fitted_cdf = stats.lognorm.cdf(sorted_prices, shape, loc, scale)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(sorted_prices, empirical_cdf, label='Empirical CDF', color='blue')
    ax2.plot(sorted_prices, fitted_cdf, label='Fitted Log-Normal CDF', color='red', linestyle='--')
    ax2.set_title("Empirical vs Fitted CDF")
    ax2.set_xlabel("Log Price")
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(
        """
        **Insights:**  
        - The empirical and fitted CDFs align closely, supporting the assumption of a log-normal price distribution.
        - Minor deviations may indicate the presence of outliers or other influencing factors.
        """
    )

    # Purchases by Price Category
    st.markdown("### Purchases by Price Category")
    data['price_category'] = data['price'].apply(lambda price: 'Low' if price < 20 else 'Medium' if 20 <= price <= 50 else 'High')
    category_counts = data[data['event_type'] == 'purchase'].groupby('price_category').size()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    category_counts.plot(kind='bar', ax=ax3, color='green')
    ax3.set_title("Purchases by Price Category")
    ax3.set_xlabel("Price Category")
    ax3.set_ylabel("Number of Purchases")
    st.pyplot(fig3)

    st.markdown(
        """
        **Insights:**  
        - Most purchases occur in the 'Low' and 'Medium' price categories.
        - This indicates user preference for more affordable products, which should be considered in pricing strategies.
        """
    )

    # User Clustering
    st.markdown("### User Clustering Based on Price Sensitivity")
    user_avg_prices = data[data['event_type'] == 'purchase'].groupby('user_id')['price'].mean().reset_index()
    user_avg_prices.rename(columns={'price': 'avg_price'}, inplace=True)

    kmeans = KMeans(n_clusters=3, random_state=42)
    user_avg_prices['cluster'] = kmeans.fit_predict(user_avg_prices[['avg_price']])

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='user_id', y='avg_price', hue='cluster', data=user_avg_prices, palette='viridis', alpha=0.7, ax=ax4)
    ax4.set_title("User Clusters Based on Average Price Sensitivity")
    ax4.set_xlabel("User ID")
    ax4.set_ylabel("Average Price")
    st.pyplot(fig4)

    st.markdown(
        """
        **Insights:**  
        - Users are segmented into three clusters based on their average spending habits.
        - These clusters provide actionable insights for targeted marketing and dynamic pricing strategies.
        """
    )

    # Monte Carlo Simulation
    st.markdown("### Monte Carlo Simulation: Revenue Variability")
    avg_price = data['price'].mean()
    std_dev_price = data['price'].std()
    conversion_rate = 0.1
    total_users = 100000
    n_simulations = 10000
    revenues = []

    for _ in range(n_simulations):
        simulated_prices = np.random.normal(avg_price, std_dev_price, total_users)
        purchases = np.random.binomial(1, conversion_rate, total_users)
        revenue = np.sum(simulated_prices * purchases)
        revenues.append(revenue)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.hist(revenues, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax5.set_title("Monte Carlo Simulation: Revenue Distribution")
    ax5.set_xlabel("Revenue")
    ax5.set_ylabel("Frequency")
    st.pyplot(fig5)

    st.markdown(
        """
        **Insights:**  
        - The Monte Carlo simulation reveals expected revenue variability due to price and conversion rate fluctuations.
        - This provides a 95% confidence interval for revenue projections, aiding in financial planning.
        """
    )

# --- Recommendation System Page ---
if selected_page == "Recommendations - Frequentist approach":
    st.markdown("<div class='main-header'>Hybrid Collaborative + Content based Recommendation System</div>", unsafe_allow_html=True)

    if data is not None:
        st.markdown(
            """
            <div class='sub-header'>
            This page presents a hybrid recommendation system that combines content-based filtering
            and collaborative filtering to provide personalized product recommendations. Enter a user ID and product ID
            to get a list of recommended products along with their associated brands.
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            # --- Preprocessing ---
            st.info("Preparing data for recommendations...")

            data['event_time'] = pd.to_datetime(data['event_time'])
            current_time = datetime.now(pytz.UTC)

            # Compute time decay
            data['time_decay'] = data['event_time'].apply(lambda x: 1 / (1 + abs((current_time - x).days)))
            data['weighted_temporal'] = data['time_decay']

            # Filter for interactions
            min_user_interactions = 3
            min_item_interactions = 3
            user_interaction_counts = data['user_id'].value_counts()
            item_interaction_counts = data['product_id'].value_counts()
            data = data[data['user_id'].isin(user_interaction_counts[user_interaction_counts >= min_user_interactions].index)]
            data = data[data['product_id'].isin(item_interaction_counts[item_interaction_counts >= min_item_interactions].index)]

            # Content-based filtering setup
            unique_products = data[['product_id', 'price', 'log_price', 'brand']].drop_duplicates()
            unique_products.reset_index(drop=True, inplace=True)

            # Sampling for cosine similarity
            sample_size = min(5000, len(unique_products))  # Limit for memory efficiency
            sampled_products = unique_products.sample(sample_size, random_state=42)

            product_features = sampled_products[['price', 'log_price']].fillna(0)
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(product_features)

            # NearestNeighbors for cosine similarity
            product_model = NearestNeighbors(metric='cosine', algorithm='brute')
            product_model.fit(normalized_features)

            def recommend_similar_products(product_id, n=10):
                try:
                    product_index = sampled_products[sampled_products['product_id'] == product_id].index[0]
                    distances, indices = product_model.kneighbors([normalized_features[product_index]], n_neighbors=n)
                    similar_products = sampled_products.iloc[indices[0]][['product_id', 'brand']]
                    return similar_products
                except IndexError:
                    return pd.DataFrame(columns=['product_id', 'brand'])

            # Collaborative filtering setup
            interaction_matrix = coo_matrix(
                (data['weighted_temporal'],
                 (data['user_id'].astype('category').cat.codes,
                  data['product_id'].astype('category').cat.codes))
            )
            interaction_matrix_csr = interaction_matrix.tocsr()

            model = NearestNeighbors(metric='cosine', algorithm='brute')
            model.fit(interaction_matrix_csr.T)

            def recommend_items(user_id, n=10):
                try:
                    user_idx = data['user_id'].astype('category').cat.codes[data['user_id'] == user_id].iloc[0]
                    user_interactions = interaction_matrix_csr[user_idx]
                    interacted_items = user_interactions.indices

                    similar_items = []
                    for item_idx in interacted_items:
                        distances, indices = model.kneighbors(interaction_matrix_csr.T[item_idx], n_neighbors=n)
                        similar_items.extend(indices.flatten())

                    similar_items = set(similar_items) - set(interacted_items)
                    recommended_products = [
                        (data['product_id'].astype('category').cat.categories[i],
                         unique_products[unique_products['product_id'] == data['product_id'].astype('category').cat.categories[i]]['brand'].values[0])
                        for i in similar_items if i < len(data['product_id'].astype('category').cat.categories)
                    ]
                    return pd.DataFrame(recommended_products, columns=['product_id', 'brand'])[:n]
                except IndexError:
                    return pd.DataFrame(columns=['product_id', 'brand'])

            # Hybrid Recommendations
            def hybrid_recommendations(user_id, product_id, n=10):
                content_recs = recommend_similar_products(product_id, n=n)
                collab_recs = recommend_items(user_id, n=n)
                hybrid_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().head(n)
                return hybrid_recs

            # --- Interactive Inputs ---
            st.markdown("### Generate Hybrid Recommendations")
            user_ids = data['user_id'].unique()
            product_ids = data['product_id'].unique()

            selected_user = st.selectbox("Select User ID", user_ids, help="Choose a user ID for recommendations.")
            selected_product = st.selectbox("Select Product ID", product_ids, help="Choose a product ID for recommendations.")

            if st.button("Generate Recommendations"):
                with st.spinner("Generating recommendations..."):
                    recommendations = hybrid_recommendations(selected_user, selected_product, n=10)
                    if not recommendations.empty:
                        st.markdown(f"**Top 10 Hybrid Recommendations for User {selected_user} and Product {selected_product}:**")
                        st.table(recommendations)
                    else:
                        st.warning("No recommendations available. Try selecting a different user or product.")

        except Exception as e:
            st.error(f"An error occurred while generating recommendations: {e}")
    else:
        st.warning("Failed to load the dataset. Please check the data source.")
