#!/usr/bin/env python
# coding: utf-8

# In[198]:


import pandas as pd

df = pd.read_csv("/Users/sivaguganjayachandran/Documents/python programming/Kaggle/e-commerce_recom_system/sampled_df.csv")

# Convert event_time to datetime format
df['event_time'] = pd.to_datetime(df['event_time'].str.replace(' UTC', ''), format='%Y-%m-%d %H:%M:%S')

# Create date column
df['date'] = df['event_time'].dt.date

# Create day_of_week column
df['day_of_week'] = df['event_time'].dt.day_name()


# In[69]:


df.head()


# In[199]:


a = df['brand'].value_counts() 
a[a> 10000]


# In[200]:


import numpy as np

# Remove rows where 'price' column has NaN values
df = df.dropna(subset=['price'])

# Optionally, you can reset the index after dropping rows
df = df.reset_index(drop=True)

# Apply logarithm to normalize price
df['log_price'] = np.log(df['price'])

# Calculate the thresholds for the 33rd and 67th percentiles
low_threshold = df['log_price'].quantile(0.333)
medium_threshold = df['log_price'].quantile(0.66)

# Define categories based on thresholds
def categorize_premiumness(log_price):
    if log_price <= low_threshold:
        return 'Low'
    elif log_price <= medium_threshold:
        return 'Medium'
    else:
        return 'High'

# Apply the categorization
df['premiumness'] = df['log_price'].apply(categorize_premiumness)

# Check the distribution of categories
print(df['premiumness'].value_counts())


# In[59]:





# In[57]:


#df['premiumness'].value_counts()

mean_price


# In[201]:


import plotly.express as px
import plotly.graph_objects as go

# Plot 1: Actual Prices with KDE
fig1 = px.histogram(
    df,
    x='price',
    nbins=30,
    title='Distribution of Actual Prices',
    labels={'price': 'Price'},
    template='plotly',
    color_discrete_sequence=['blue']
)
fig1.update_traces(opacity=0.7)
fig1.update_layout(
    xaxis_title='Price',
    yaxis_title='Frequency'
)
fig1.show()

# Plot 2: Log-Transformed Prices
fig2 = px.histogram(
    df,
    x='log_price',
    nbins=30,
    title='Distribution of Log-Transformed Prices',
    labels={'log_price': 'Log Price'},
    template='plotly',
    color_discrete_sequence=['purple']
)
fig2.update_traces(opacity=0.7)
fig2.update_layout(
    xaxis_title='Log Price',
    yaxis_title='Frequency'
)
fig2.show()

# Plot 3: Log-Transformed Prices by Premiumness
fig3 = px.histogram(
    df,
    x='log_price',
    nbins=30,
    color='premiumness',
    title='Distribution of Log-Transformed Prices by Premiumness',
    labels={'log_price': 'Log Price', 'premiumness': 'Premiumness'},
    color_discrete_map={'Low': 'blue', 'Medium': 'orange', 'High': 'green'},  # Map specific colors
    template='plotly'
)
fig3.update_traces(opacity=0.7)
fig3.update_layout(
    xaxis_title='Log Price',
    yaxis_title='Frequency',
    legend_title='Premiumness'
)
fig3.show()


# In[204]:


df['premiumness'].value_counts()


# In[205]:


import plotly.graph_objects as go

# Calculate mean price and product count
premiumness_order = ['Low', 'Medium', 'High']  # Explicit order
product_counts = df['premiumness'].value_counts().reindex(premiumness_order)
mean_price = df.groupby('premiumness')['price'].mean().reindex(premiumness_order)

# Create the figure
fig = go.Figure()

# Add bar trace for product counts
fig.add_trace(
    go.Bar(
        x=premiumness_order,
        y=product_counts,
        name='Count of Products',
        marker_color='blue',
        yaxis='y1'
    )
)

# Add line trace for mean price
fig.add_trace(
    go.Scatter(
        x=premiumness_order,
        y=mean_price,
        name='Mean log Price',
        mode='lines+markers',
        line=dict(color='red', width=2),
        yaxis='y2'
    )
)

# Update layout for dual y-axes
fig.update_layout(
    title='Distribution of Product Premiumness with Mean log Price',
    xaxis=dict(title='Premiumness Category'),
    yaxis=dict(
        title='Count of Products',
        titlefont=dict(color='royalblue'),
        tickfont=dict(color='royalblue')
    ),
    yaxis2=dict(
        title='Mean log Price',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right'
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    barmode='group',
    template='plotly_dark',
    height = 600
)

# Show the plot
fig.show()


# In[83]:


import pandas as pd

# Create a DataFrame with premiumness categories and funnel metrics
funnel_df = pd.DataFrame({
    'view': funnel_metrics['view'].tolist(),
    'view_to_cart': funnel_metrics['view_to_cart'].tolist(),
    'cart_to_purchase': funnel_metrics['cart_to_purchase'].tolist(),
    'cart_to_remove': funnel_metrics['cart_to_remove'].tolist()
}, index=['Low', 'Medium', 'High'])

# Display the DataFrame
funnel_df


# In[92]:


import plotly.graph_objects as go

# Create the figure
fig = go.Figure()

# Iterate over each row (premiumness) and plot the columns (metrics) excluding "view"
for premiumness in funnel_df.index:
    fig.add_trace(go.Scatter(
        x=funnel_df.columns[1:],  # Exclude "view" column from the x-axis
        y=funnel_df.loc[premiumness, 'view_to_cart':'cart_to_remove'],  # Values for the metrics excluding "view"
        mode='lines+markers',  # Line plot with markers at each point
        name=premiumness  # Use premiumness as the legend name
    ))

# Update layout
fig.update_layout(
    title='Funnel Metrics by Premiumness Category',
    xaxis_title='Funnel Metric',
    yaxis_title='Metric Value',
    template='plotly',
    legend_title='Premiumness Category'
)

# Show the plot
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[155]:


a = df['brand'].value_counts() 
a[a> 30000]


# # One-sample t-test

# In[158]:


# Step 1: Group by product_id, user_session, and event_type
brand_df = df.groupby(['brand','user_session', 'event_type']).size().unstack(fill_value=0)

# Step 2: Calculate the funnel metrics
# Calculating funnel metrics for each session/product
brand_df['cart'] = brand_df['cart'] 
brand_df['purchase'] = brand_df['purchase'] 
brand_df['remove_from_cart'] = brand_df['remove_from_cart'] 
brand_df['view_to_cart'] = brand_df['cart'] / brand_df['view']
brand_df['cart_to_purchase'] = brand_df['purchase'] / brand_df['cart']
brand_df['cart_to_remove'] = brand_df['remove_from_cart'] / brand_df['cart']

brand_df = brand_df.reset_index()


brand_df.head(5)


# In[166]:


import scipy.stats as stats

# Step 1: Extract the data for 'runail' and the rest of the brands
runail_data = brand_df[brand_df['brand'] == 'runail']['purchase']
population_data = brand_df[brand_df['brand'] != 'runail']['purchase']

# Step 2: Calculate the mean purchase for both 'runail' (sample) and the population
mean_runail = runail_data.mean()
mean_population = population_data.mean()

# Step 3: Perform a one-sample t-test to compare the 'runail' mean to the population mean
t_stat, p_value = stats.ttest_1samp(runail_data, mean_population)

# Step 4: Confidence interval (95% confidence)
confidence_interval = stats.t.interval(0.95, len(runail_data)-1, loc=mean_runail, scale=stats.sem(runail_data))

# Step 5: Print the detailed report
print("Analysis of Purchase Values for 'runail' vs Population:\n")
print(f"Sample Size:\n- 'runail': {len(runail_data)}\n- Population: {len(population_data)}\n")

print(f"Metric Value (Mean Purchase):")
print(f"- 'runail': {mean_runail:.6f}")
print(f"- Population: {mean_population:.6f}\n")

print("T-test Results:")
print(f"- T-statistic: {t_stat:.6f}")
print(f"- P-value: {p_value:.6f}\n")

print(f"95% Confidence Interval for 'runail':")
print(f"- ({confidence_interval[0]:.6f}, {confidence_interval[1]:.6f})\n")

# Step 6: Interpret the results
if p_value < 0.05:
    print("Conclusion:")
    print("Reject the null hypothesis: There is a statistically significant difference in purchase values between 'runail' and the population.")
else:
    print("Conclusion:")
    print("Fail to reject the null hypothesis: There is no statistically significant difference in purchase values between 'runail' and the population.")


# In[ ]:





# # Independent two-sample t-test (unpaired t-test)

# In[160]:


# Step 1: Group by product_id, user_session, and event_type
product_df = df.groupby(['product_id', 'premiumness', 'user_session','event_type']).size().unstack(fill_value=0)

# Step 2: Calculate the funnel metrics
# Calculating funnel metrics for each session/product
product_df['cart'] = product_df['cart'] 
product_df['purchase'] = product_df['purchase'] 
product_df['remove_from_cart'] = product_df['remove_from_cart'] 
product_df['view_to_cart'] = product_df['cart'] / product_df['view']
product_df['cart_to_purchase'] = product_df['purchase'] / product_df['cart']
product_df['cart_to_remove'] = product_df['remove_from_cart'] / product_df['cart']

# Step 3: Reset index to make the DataFrame more readable
product_df = product_df.reset_index()

product_df.head()


# In[164]:


import scipy.stats as stats
import numpy as np
from statsmodels.stats.power import TTestIndPower

# Extract the purchase values for High and Low premiumness
high_purchase = product_df[product_df['premiumness'] == 'High']['purchase'].dropna()
low_purchase = product_df[product_df['premiumness'] == 'Low']['purchase'].dropna()

# Calculate the central tendency (mean) for each group
mean_high = high_purchase.mean()
mean_low = low_purchase.mean()

# Perform the independent two-sample t-test
t_stat, p_value = stats.ttest_ind(high_purchase, low_purchase)

# Calculate Cohen's d (Effect size)
pooled_std = np.sqrt(((len(high_purchase)-1)*high_purchase.std()**2 + (len(low_purchase)-1)*low_purchase.std()**2) / (len(high_purchase) + len(low_purchase) - 2))
cohens_d = (mean_high - mean_low) / pooled_std

# Calculate the statistical power of the test
power_analysis = TTestIndPower()
power = power_analysis.solve_power(effect_size=cohens_d, nobs1=len(high_purchase), ratio=len(low_purchase)/len(high_purchase), alpha=0.05)

# 95% Confidence Intervals
ci_high = stats.t.interval(0.95, len(high_purchase)-1, loc=mean_high, scale=stats.sem(high_purchase))
ci_low = stats.t.interval(0.95, len(low_purchase)-1, loc=mean_low, scale=stats.sem(low_purchase))

# Print short story output
output = f"""
Analysis of Purchase Values for High vs Low Premiumness Products:

Sample size:
- High Premiumness: {len(high_purchase)}
- Low Premiumness: {len(low_purchase)}

Metric value (mean purchase):
- High Premiumness: {mean_high:.6f}
- Low Premiumness: {mean_low:.6f}

T-test Results:
- T-statistic: {t_stat:.6f}
- P-value: {p_value:.6f}

95% Confidence Intervals:
- High Premiumness: {ci_high}
- Low Premiumness: {ci_low}

Effect Size (Cohen's d): {cohens_d:.6f}
Power of the Test: {power:.6f}

Conclusion:
- { 'Reject the null hypothesis: There is a significant difference in purchase values between High and Low premiumness.' if p_value < 0.05 else 'Fail to reject the null hypothesis: No significant difference in purchase values between High and Low premiumness.' }
"""

print(output)


# # Bayesian Recommendation system 

# In[170]:


import pandas as pd
from scipy.stats import beta

df1 = pd.read_csv("/Users/sivaguganjayachandran/Documents/python programming/Kaggle/e-commerce_recom_system/sampled_df.csv")

# Preprocessing: Filter relevant event types
df1 = df1[df1['event_type'].isin(['purchase', 'cart'])]

# Compute the Prior: Total purchases / Total cart events
prior = df1[df1['event_type'] == 'purchase'].shape[0] / df1[df1['event_type'] == 'cart'].shape[0]
print(f"Prior (P(purchase)): {prior}")

# Compute Likelihood: Frequency of purchase given user-product pairs
# Filter for purchase events
purchase_counts = df1[df1['event_type'] == 'purchase'].groupby(['user_id', 'product_id']).size()
cart_counts = df1[df1['event_type'] == 'cart'].groupby(['user_id', 'product_id']).size()

# Combine counts into a single DataFrame
likelihood = pd.concat([purchase_counts, cart_counts], axis=1, keys=['purchases', 'carts']).fillna(0)
likelihood['likelihood'] = likelihood['purchases'] / likelihood['carts']

# Compute Posterior using Bayesian updating
def compute_posterior(row):
    alpha = row['purchases'] + 1  # Add 1 for Bayesian smoothing
    beta_param = row['carts'] - row['purchases'] + 1  # Add 1 for smoothing
    posterior = beta(alpha, beta_param).mean()  # Mean of Beta distribution
    return posterior

likelihood['posterior'] = likelihood.apply(compute_posterior, axis=1)

# Merge posterior probabilities back into the main DataFrame
df1 = df1.merge(likelihood[['posterior']], how='left', left_on=['user_id', 'product_id'], right_index=True)

# Rank Recommendations for Each User
recommendations = df1.groupby('user_id').apply(
    lambda x: x.sort_values(by='posterior', ascending=False).head(5)
).reset_index(drop=True)

# Display top recommendations
print(recommendations[['user_id', 'product_id', 'posterior']])

# Save recommendations to a CSV file
recommendations[['user_id', 'product_id', 'posterior']].to_csv('recommendations.csv', index=False)


# In[183]:


import pandas as pd
import numpy as np

# Step 1: Get top 10 users with the most observations
user_counts = df1['user_id'].value_counts().head(10)  # Count occurrences of user_ids and take top 10
top_users = user_counts.index  # Top 10 users

# Step 2: Get top 2 products with the most observations
product_counts = df1['product_id'].value_counts().head(2)  # Count occurrences of product_ids and take top 2
top_products = product_counts.index  # Top 2 products

# Step 3: Calculate prior probability (P(purchase)) for all products
prior_prob = df1.groupby('product_id')['event_type'].apply(lambda x: (x == 'purchase').mean()).reset_index()
prior_prob.columns = ['product_id', 'prior_prob']

# Step 4: Calculate likelihood based on user history for each product
def calculate_likelihood(user_id, product_id, df1):
    # Get user history of purchases
    user_history = df1[df1['user_id'] == user_id]
    
    # Find the category of the product
    product_category = df1[df1['product_id'] == product_id]['category_id'].values[0]
    
    # Filter user's past purchases in the same category
    user_purchases_in_category = user_history[user_history['category_id'] == product_category]
    
    # Likelihood based on number of similar products purchased by the user
    likelihood = len(user_purchases_in_category) / len(user_history) if len(user_history) > 0 else 0
    return likelihood

# Step 5: Calculate posterior probability using Bayesian update
def calculate_posterior(user_id, product_id, df1, prior_prob):
    # Get the prior probability for the product
    prior = prior_prob[prior_prob['product_id'] == product_id]['prior_prob'].values[0]
    
    # Get the likelihood of the user purchasing this product based on their history
    likelihood = calculate_likelihood(user_id, product_id, df1)
    
    # Bayesian update: Posterior ∝ Prior * Likelihood
    posterior = prior * likelihood  # Posterior before normalization
    return posterior

# Step 6: Calculate posterior probability for the top 10 users and top 2 products
posterior_probabilities = {}

for user in top_users:
    for product in top_products:
        posterior_prob = calculate_posterior(user, product, df1, prior_prob)
        posterior_probabilities[(user, product)] = posterior_prob

# Convert results to DataFrame for better readability
posterior_df = pd.DataFrame(posterior_probabilities.items(), columns=['user_product', 'posterior_prob'])
posterior_df[['user_id', 'product_id']] = pd.DataFrame(posterior_df['user_product'].to_list(), index=posterior_df.index)
posterior_df = posterior_df.drop(columns=['user_product'])

# Display the final DataFrame with posterior probabilities
print(posterior_df)


# ### Bayesian Inference for Purchase Probability
# 
# Bayes' Theorem is used to calculate the probability of a user purchasing a product, based on both the product's purchase history and the user's past interactions with similar products.
# 
# #### Formula:
# 
# \[
# P(\text{purchase} | \text{user, product}) = \frac{P(\text{purchase} | \text{product}) \cdot P(\text{purchase} | \text{user, product category})}{P(\text{purchase})}
# \]
# 
# Where:
# 
# - **Posterior Probability** \( P(\text{purchase} | \text{user, product}) \): The probability that a user will purchase a product.
# - **Prior Probability** \( P(\text{purchase} | \text{product}) \): The probability of purchasing the product, based on all users’ data.
# - **Likelihood** \( P(\text{purchase} | \text{user, product category}) \): The probability of a user purchasing from the product's category, based on their history.
# - **Marginal Probability** \( P(\text{purchase}) \): The overall purchase probability (often constant and omitted in simplified calculations).
# 
# In short, this combines the general likelihood of purchasing a product with the user's specific behavior towards similar products to calculate a personalized probability.

# In[194]:


import pandas as pd

# Load the dataset
df1 = df.copy()

# Preprocessing: Filter relevant event types
df1 = df1[df1['event_type'].isin(['purchase', 'cart'])]

# Step 1: Get top 10 users with the most observations
user_counts = df1['user_id'].value_counts().head(10)  # Count occurrences of user_ids and take top 10
top_users = user_counts.index  # Top 10 users

# Step 2: Get top 2 products with the most observations
product_counts = df1['product_id'].value_counts().head(2)  # Count occurrences of product_ids and take top 2
top_products = product_counts.index  # Top 2 products

# Filter dataset for top users and top products
df1 = df1[df1['user_id'].isin(top_users) & df1['product_id'].isin(top_products)]

# Compute the Prior: Total purchases / Total events
prior_purchase = df1[df1['event_type'] == 'purchase'].shape[0] / df1.shape[0]
prior_not_purchase = 1 - prior_purchase
print(f"Prior (P(purchase)): {prior_purchase}")
print(f"Prior (P(not purchase)): {prior_not_purchase}")

# Compute Likelihood: Frequency of purchase given user-product pairs
# Filter for purchase events
purchase_counts = df1[df1['event_type'] == 'purchase'].groupby(['user_id', 'product_id']).size()
cart_counts = df1[df1['event_type'] == 'cart'].groupby(['user_id', 'product_id']).size()

# Combine counts into a single DataFrame
likelihood = pd.concat([purchase_counts, cart_counts], axis=1, keys=['purchases', 'carts']).fillna(0)
likelihood['purchase_likelihood'] = likelihood['purchases'] / likelihood['carts']
likelihood['not_purchase_likelihood'] = (likelihood['carts'] - likelihood['purchases']) / likelihood['carts']

# Compute denominator (P(user, product))
likelihood['denominator'] = (
    likelihood['purchase_likelihood'] * prior_purchase +
    likelihood['not_purchase_likelihood'] * prior_not_purchase
)

# Compute Posterior using Bayesian updating
likelihood['posterior'] = (
    likelihood['purchase_likelihood'] * prior_purchase
) / likelihood['denominator']

# Merge posterior probabilities back into the main DataFrame
df1 = df1.merge(likelihood[['posterior']], how='left', left_on=['user_id', 'product_id'], right_index=True)

# Generate all combinations of top users and top products
combinations = pd.MultiIndex.from_product([top_users, top_products], names=['user_id', 'product_id']).to_frame(index=False)

# Merge combinations with computed posteriors
recommendations = combinations.merge(likelihood[['posterior']], how='left', on=['user_id', 'product_id']).fillna(0)

# Display recommendations
recommendations

