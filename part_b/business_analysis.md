# Part B: Business Case Analysis

### B1. Problem Formulation

**(a) Formulate this as a machine learning problem. State clearly: what is the target variable, what are the candidate input features, and what type of ML problem is this? Justify your choice of problem type.**

*   **Target Variable:** The target variable is the **number of items sold** (or sales volume) for a given store, for a specific promotion, in a particular month.

*   **Candidate Input Features:**
    *   **Promotion Characteristics:** Type of promotion (Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus).
    *   **Store Characteristics:** Store ID, store size (e.g., square footage), monthly footfall, local competition density (e.g., number of competing stores within a radius), store location type (urban, semi-urban, rural), customer demographics (e.g., average age, income level, prevalent customer segments).
    *   **Temporal Features:** Month of the year, presence of holidays or special events during the promotion period.
    *   **Historical Performance:** Past performance of various promotions in that specific store or similar stores.

*   **Type of ML Problem:** This is a **regression** problem.

*   **Justification:** The target variable, 'number of items sold,' is a continuous numerical value. The goal is to predict this continuous value based on a set of input features. Predicting a continuous output makes this a classic regression task.

**(b) The company currently measures performance using total sales revenue. Explain why using items sold (sales volume) is a more reliable target variable for this problem. What broader principle does this illustrate about target variable selection in real-world ML projects?**

*   **Why 'items sold' is more reliable:** The core objective stated by the company is to "maximise the number of items sold." While sales revenue is important, promotions often involve discounts (e.g., Flat Discount, BOGO), which can increase the quantity of items sold but might reduce the average revenue per item. If the target variable were revenue, a model might recommend promotions that maximize revenue per transaction, potentially overlooking promotions that drive high volume but at a lower price point, even if those high-volume promotions better meet the stated goal of maximizing *items sold*.

*   **Broader Principle:** This illustrates the crucial principle of **direct alignment with the business objective**. The target variable chosen for a machine learning model must directly correspond to, and accurately quantify, the specific business problem or goal that the model is intended to solve. Using a proxy metric (like revenue when the goal is volume) can lead to models that are technically accurate but fail to deliver true business value because they are optimizing for the wrong outcome. It underscores the necessity of a deep understanding of the business context and clear problem definition before model development.

**(c) A junior analyst suggests running one single global model across all 50 stores. Propose and justify an alternative modelling strategy that accounts for the fact that stores in different locations respond very differently to the same promotion.**

*   **Problem with a single global model:** A single global model across all 50 stores would assume that the relationship between promotions and items sold is uniform across all store types (urban, semi-urban, rural, varying footfall, demographics). This is unlikely to be true. A promotion that works well in a high-footfall urban store might be ineffective or even detrimental in a rural store with different customer demographics. A global model would struggle to capture these localized nuances and could lead to suboptimal recommendations for many stores.

*   **Alternative Modelling Strategy: Segmented or Hierarchical Models**
    
    A more effective strategy would be to account for store heterogeneity, for example, through:
    
    1.  **Segmented Models:** Train separate, specialized models for different *segments* of stores. For instance, one model could be trained for 'urban' stores, another for 'semi-urban,' and a third for 'rural' stores. This approach allows each model to learn patterns and promotional elasticities specific to its segment, providing more accurate predictions and recommendations tailored to the unique characteristics of each store type.
    
    2.  **Hierarchical Modelling (or Mixed-Effects Models):** This approach involves building a single model that incorporates both global effects (patterns common across all stores) and store-specific effects (how each individual store deviates from the global pattern). This can be done by including store ID as a categorical feature with interaction terms, or by using more advanced techniques like hierarchical Bayesian models or multi-task learning, where the model learns store-specific parameters or embeddings while leveraging data from all stores. This strategy is particularly powerful as it allows for learning from limited data in individual stores by 'borrowing strength' from the overall dataset, while still providing tailored predictions.

*   **Justification:** Both segmented and hierarchical approaches directly address the stated problem that stores respond differently. They enable the model to learn and exploit these differences, leading to more granular, accurate, and actionable insights for promotion deployment. This ensures that the recommended promotion for each store is optimized for its specific context, thereby maximizing the total number of items sold across the entire chain more effectively than a one-size-fits-all global model.

### B2. Data and EDA Strategy

**(a) Describe how you would join these tables. What is the grain of the final modelling dataset (one row = what?), and what aggregations would you perform before modelling?**

*   **Joining Tables:**
    1.  **Transactions and Promotion Details:** We would join the `transactions` table with the `promotion details` table using `promotion_id` (assuming this is a common key that links transactions to the specific promotion applied). This would enrich each transaction record with details about the promotion type. If promotions are store-specific and time-bound, additional keys like `store_id` and `date/month` might be needed for this join.
    2.  **Store Attributes:** The `store attributes` table would be joined with the combined `transactions` and `promotion details` data using `store_id`. This would add static store characteristics (size, footfall, location type, competition density, demographics) to each transaction record.
    3.  **Calendar:** The `calendar` table would be joined using a `date` or `month` key to add temporal information (e.g., day of week, weekend flag, holiday flag) to each transaction record.

*   **Grain of the Final Modelling Dataset:**
    Given the objective is to predict 'number of items sold' for a given store, for a specific promotion, in a particular month, the most suitable grain for the final modelling dataset would be **one row per (Store ID, Month, Promotion Type)**. This means each row represents the performance of a specific promotion type in a specific store during a specific month.

*   **Aggregations before Modelling:**
    Before reaching the target grain, several aggregations would be performed:
    1.  **From Transactions:** Aggregate the `transactions` data to calculate the total `number of items sold` for each combination of (Store ID, Date, Promotion ID/Type). This will be our initial target variable aggregation.
    2.  **Monthly Aggregation:** Further aggregate the data to the (Store ID, Month, Promotion Type) grain. At this stage, we would sum the 'number of items sold' for each unique combination, creating our final target variable. For other features, we would take the average or mode of daily/transaction-level features across the month (e.g., average footfall, presence of holidays).
    3.  **Features from Calendar:** Identify monthly features from the calendar, such as the number of weekends, number of holidays, or specific holiday flags within that month.

**(b) Describe the EDA you would perform before building a model. Specify at least four analyses or charts, what you would look for in each, and how the findings would influence your feature engineering or modelling decisions.**

1.  **Promotional Effectiveness by Type and Store Location (Bar Chart/Box Plot):**
    *   **Analysis:** Visualize the average 'items sold' for each promotion type, segmented by store location type (urban, semi-urban, rural). Also, look at the distribution of 'items sold' for each promotion type using box plots.
    *   **What to Look For:** Which promotion types perform best/worst overall? Are there significant differences in promotion effectiveness across urban, semi-urban, and rural stores? Do some promotions have highly variable results?
    *   **Influence:** This would inform feature engineering (e.g., creating interaction features between `promotion_type` and `store_location_type`). It would also validate the need for segmented or hierarchical models if certain promotion types are consistently good or bad in specific store types, or if the variance is high.

2.  **Correlation Matrix/Heatmap of Numerical Features:**
    *   **Analysis:** Compute and visualize the correlation matrix between numerical features (e.g., store size, monthly footfall, local competition density) and the target variable ('items sold'), as well as inter-feature correlations.
    *   **What to Look For:** Strong positive or negative correlations with the target variable, indicating potentially important predictors. Also, highly correlated input features (multicollinearity), which might warrant dimensionality reduction or careful selection to avoid redundancy.
    *   **Influence:** Identify key drivers for 'items sold'. Suggest potential feature engineering like polynomial features or interaction terms if non-linear relationships are suspected. Highlight features for potential removal or combination if multicollinearity is high.

3.  **Time Series Analysis of 'Items Sold' (Line Plots):**
    *   **Analysis:** Plot 'items sold' over time for individual stores, groups of stores, and aggregated across all stores. Overlay major holiday periods or specific promotion deployments.
    *   **What to Look For:** Trends, seasonality (e.g., monthly peaks, holiday surges), and the immediate impact of promotions. Are there common seasonal patterns across stores? Do promotions consistently lead to spikes in sales?
    *   **Influence:** This would guide the creation of temporal features (e.g., `month_of_year`, `day_of_month`, `weeks_since_last_holiday`, `promotion_duration`). It could also inform the choice of time-series capable models or cross-validation strategies.

4.  **Distribution of 'Items Sold' and Key Categorical Features (Histograms/Count Plots):**
    *   **Analysis:** Plot histograms of the 'items sold' variable to understand its distribution (e.g., skewed, normal, presence of outliers). Use count plots or bar charts for categorical features like `promotion_type`, `store_location_type`, `customer_demographics` to see their distributions.
    *   **What to Look For:** Skewness in the target variable (might require transformation like log-transform). Imbalances in categorical feature distributions (e.g., some promotion types are very rare), which could affect model training or require resampling.
    *   **Influence:** Inform data preprocessing steps (e.g., target variable transformation). Indicate if certain categories need to be combined due to low counts or if specific handling is needed for rare categories during encoding. Highlight potential data quality issues like outliers.

**(c) You notice that 80% of transactions in the dataset occurred without any promotion. Describe how this imbalance could affect your model and what steps you would take to address it.**

*   **How Imbalance Could Affect the Model:**
    An 80% absence of promotion means that only 20% of the data points are related to active promotions. If our model is trained on this imbalanced dataset, it might become very good at predicting baseline sales (without promotion) but poor at predicting the *impact* of promotions. The model could effectively learn to ignore promotion-related features because the majority class (no promotion) dominates. This could lead to:
    *   **Bias towards the majority class:** The model might default to predicting lower sales (characteristic of non-promotion periods) even when a promotion is active.
    *   **Poor performance on promotion prediction:** The model would struggle to accurately estimate the uplift or specific impact of different promotion types, which is the core business objective.
    *   **Underfitting the minority class:** The patterns and nuances of how different promotions affect sales might not be adequately captured due to insufficient data for the 'promoted' state.

*   **Steps to Address It:**
    1.  **Resampling Techniques:**
        *   **Oversampling the Minority Class:** Duplicate instances of promotion-applied data points. This could involve simple random oversampling or more advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples for the promotion cases.
        *   **Undersampling the Majority Class:** Randomly remove instances of non-promotion data points. This should be done carefully to avoid losing valuable information but can balance the dataset.
        *   **Combination of Over- and Under-sampling:** Often, a combination of both is effective (e.g., using NearMiss for undersampling and SMOTE for oversampling).
    2.  **Weighted Loss Functions:** For some machine learning algorithms (e.g., gradient boosting, neural networks), we can assign higher weights to the minority class (promotion events) in the loss function. This makes the model penalize errors on promotion predictions more heavily, forcing it to learn their patterns.
    3.  **Generate a Baseline Model for Non-Promotional Sales:** Train a separate model or a component of a model specifically to predict baseline sales (without promotion). Then, build another model that focuses solely on predicting the *uplift* (additional sales) due to promotions. The final prediction would be baseline sales + predicted uplift.
    4.  **Feature Engineering:** Ensure that features clearly differentiate between promotional and non-promotional periods (e.g., binary `is_promotion_active` flag, `days_since_promotion_started`). This might also involve creating interaction terms between promotion flags and other features.
    5.  **Collect More Data (if feasible):** If possible, gathering more data specifically on promotional events would be the most robust solution, though often not immediately available.

The choice of method would depend on the specific dataset characteristics, model chosen, and computational resources, but resampling and weighted loss functions are common starting points.

### B3. Model Evaluation and Deployment

**(a) You have monthly store-level data spanning three years across 50 stores. Describe how you would set up the train-test split. Why is a random split inappropriate here? Which evaluation metrics would you use, and how would you interpret each in the context of this business problem?**

*   **Train-Test Split Setup:**
    Given that we have monthly store-level data spanning three years, a **time-based split** is crucial. I would split the data as follows:
    *   **Training Data:** The first two years of data (e.g., January 20XX to December 20YY) would be used for training the model.
    *   **Validation Data (Optional but Recommended):** A period immediately following the training data (e.g., January 20ZZ to June 20ZZ) could be used for hyperparameter tuning and model selection.
    *   **Test Data:** The most recent six months to one year of data (e.g., July 20ZZ to December 20ZZ, or the entire third year) would be reserved as the test set. This simulates a real-world scenario where the model predicts future performance.

*   **Why a Random Split is Inappropriate:**
    A random split would intersperse future data points into the training set, allowing the model to 'peek' into the future. In time-series or sequential data, future information often depends on past events (e.g., promotional effectiveness might change over time, or seasonal patterns exist). A model trained on future data would likely perform unrealistically well on the test set, but fail to generalize to actual future predictions. Random splitting ignores the temporal dependency in the data, leading to an overly optimistic evaluation of the model's performance in a deployment setting.

*   **Evaluation Metrics and Interpretation:**
    Since this is a regression problem (predicting 'number of items sold'), the primary metrics would be:
    1.  **Mean Absolute Error (MAE):**
        *   **Interpretation:** MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. An MAE of 50 items sold means, on average, our prediction is off by 50 items. In the business context, this directly tells the marketing team how far off, on average, their projected sales volume for a promotion is from the actual sales. A lower MAE is better, as it indicates more precise predictions.
    2.  **Root Mean Squared Error (RMSE):**
        *   **Interpretation:** RMSE is the square root of the average of the squared errors. It penalizes large errors more heavily than MAE. This is useful because large prediction errors (e.g., drastically overestimating or underestimating sales for a high-value promotion) can have significant business implications (e.g., stockouts or excess inventory). A lower RMSE is better, indicating that the model's predictions are not only close on average but also avoid large, detrimental deviations.
    3.  **R-squared (R²):**
        *   **Interpretation:** R-squared represents the proportion of the variance in the dependent variable (items sold) that is predictable from the independent variables (features). An R² of 0.80 means that 80% of the variability in items sold can be explained by our model's features. This metric helps understand how well the model captures the overall trends and variations in sales. A higher R² indicates a better fit, meaning the model is doing a good job of explaining the factors influencing sales.
    4.  **Mean Absolute Percentage Error (MAPE):**
        *   **Interpretation:** MAPE expresses the error as a percentage of the actual value. For example, a MAPE of 10% means the predictions are, on average, 10% off the actual sales. This metric is highly intuitive for business stakeholders as it provides a relative measure of error. It's particularly useful for comparing model performance across different scales of sales (e.g., a store with high sales vs. low sales). However, it can be problematic with zero or near-zero actual values.

**(b) After training, the model recommends the Loyalty Points Bonus for Store 12 in December and the Flat Discount for Store 12 in March. Using the concept of feature importance, explain how you would investigate and communicate to the marketing team why the model makes different recommendations for the same store in different months.**

To investigate and communicate why the model makes different recommendations for Store 12 in December versus March, I would leverage **feature importance** and **local interpretability techniques**:

1.  **Understand the Core Goal:** The model is optimizing for 'number of items sold'. Differences in recommendations for the same store across months imply that the *drivers* of sales for different promotion types change with temporal context or other store attributes.

2.  **Global Feature Importance (Initial Check):**
    *   **Technique:** Use model-agnostic methods like Permutation Feature Importance or model-specific methods (e.g., Gini Importance for tree-based models, coefficients for linear models) to understand which features generally have the most impact on 'items sold' across the entire dataset. This would tell us if `month_of_year`, `promotion_type`, `is_holiday_season`, `store_footfall`, etc., are generally important.
    *   **Communication:** Present a ranked list or chart of the most influential features overall. For instance, "Globally, `promotion_type` and `month_of_year` are among the top drivers of sales volume."

3.  **Local Explanations for Specific Recommendations (Key for Marketing Team):**
    *   **Technique (e.g., SHAP or LIME):** These methods can explain *why* a specific prediction was made for a single instance. I would apply them to two specific scenarios:
        *   **Scenario 1: Store 12, December, Loyalty Points Bonus.**
        *   **Scenario 2: Store 12, March, Flat Discount.**
    *   For each scenario, SHAP (SHapley Additive exPlanations) values, for example, would tell us how much each feature contributed to the model's prediction of 'items sold' when recommending that specific promotion. We'd compare the feature contributions for the recommended promotion against other promotion types for that specific month/store.

4.  **Communication to Marketing Team:**
    I would present this using a narrative-driven approach, potentially with comparison charts:
    *   **Start with the Observation:** "For Store 12, our model recommends different promotions in December (Loyalty Points Bonus) versus March (Flat Discount). Let's dive into why."
    *   **Highlight Key Differentials:**
        *   **December Context:** "In December, features like 'holiday season proximity', 'increased customer traffic due to Christmas shopping', and potentially 'customers saving points for bigger purchases' likely make the *Loyalty Points Bonus* more effective. For example, our analysis shows that in December, the 'Loyalty Points Bonus' recommendation for Store 12 was driven primarily by 'high monthly footfall' (contributing X more items than average), 'December seasonality' (contributing Y more items), and the inherent 'loyalty program attractiveness' (contributing Z more items). Other promotions, while potentially offering a boost, don't align as well with the consumer mindset and shopping patterns of this month."
        *   **March Context:** "Conversely, in March, when sales might be slower post-holiday, the model suggests a *Flat Discount*. This could be driven by factors such as 'post-holiday budget consciousness' or 'clearance opportunities'. Our analysis indicates that for March, the 'Flat Discount' recommendation for Store 12 was strongly influenced by 'desire for immediate savings' (contributing A more items), 'lower baseline sales volume for that month' (making discounts more impactful to drive traffic, contributing B more items), and 'seasonal clothing transition' (clearing old stock, contributing C more items). The Loyalty Points Bonus, while useful, might not provide the immediate incentive consumers are looking for in March."
    *   **Reinforce Business Logic:** "Essentially, the model adapts its recommendation based on the *context* of the month and the store's characteristics, identifying which promotion's inherent value proposition best aligns with customer behavior and market conditions during that specific period to maximize sales volume."

**(c) The trained model needs to generate recommendations at the start of every month for all 50 stores without being retrained each time. Describe the end-to-end deployment process: how you would save the model, how new monthly data would be prepared and fed in, and what monitoring you would put in place to detect when the model's performance has degraded and retraining is needed.**

**End-to-End Deployment Process:**

1.  **Model Saving and Versioning:**
    *   **Serialization:** Once the model is trained and validated, it would be saved (serialized) using libraries like `pickle`, `joblib`, or native model saving functions (e.g., `model.save()` for Keras/TensorFlow, `save_model()` for LightGBM/XGBoost). This saves the trained weights and architecture.
    *   **Versioning:** The saved model would be versioned (e.g., `model_v1.0.pkl`, `model_v1.1.pkl`). This ensures reproducibility and allows for easy rollback to previous versions if issues arise. Model metadata (training data, hypermeters, performance metrics) would also be stored alongside the model artifact.
    *   **Storage:** The model artifact and its metadata would be stored in a secure, accessible location, such as a cloud storage bucket (e.g., Google Cloud Storage, AWS S3) or a dedicated Model Registry service (e.g., Vertex AI Model Registry, MLflow).

2.  **New Monthly Data Preparation and Feeding:**
    *   **Data Ingestion:** At the beginning of each month, new data for the *upcoming* month would be ingested. This would include:
        *   Current month's `promotion details` (if known in advance for planning).
        *   Relevant `calendar` data for the upcoming month (holidays, weekends, etc.).
        *   Latest `store attributes` (these are generally static but could be updated).
        *   Historical `transactions` data would be needed if the model uses lagged features (e.g., last month's sales, average sales over the last 3 months).
    *   **Feature Engineering Pipeline:** The same pre-processing and feature engineering pipeline used during training *must* be applied to this new data. This includes:
        *   Joining the tables as described in B2(a).
        *   Aggregating data to the (Store ID, Month, Promotion Type) grain.
        *   Creating any derived features (e.g., `month_of_year`, `is_holiday_season`, `average_footfall_last_month`).
        *   Applying any necessary transformations (e.g., scaling numerical features, one-hot encoding categorical features) using the *same transformers fitted on the training data*.
    *   **Batch Prediction:** The prepared monthly data for all 50 stores, encompassing all five promotion types for the upcoming month, would be fed into the loaded, saved model in a batch process. The model would output predicted 'number of items sold' for each (Store ID, Month, Promotion Type) combination.
    *   **Recommendation Generation:** For each store and month, the promotion type that yields the highest predicted 'number of items sold' would be selected as the recommendation.

3.  **Monitoring for Performance Degradation and Retraining:**
    Continuous monitoring is critical for identifying **model drift** and ensuring the model remains effective over time. This involves:
    *   **Performance Monitoring (Post-Actuals):**
        *   **Metric Tracking:** Once actual sales data for the month becomes available (e.g., at the end of the month), the model's performance on the recommendations it made would be calculated using the same evaluation metrics (MAE, RMSE, MAPE, R²) as used during initial evaluation. These metrics would be tracked over time in a dashboard.
        *   **Thresholds and Alerts:** Predefined thresholds would be set for these metrics (e.g., if MAE increases by 20% or R² drops below 0.75). If a metric crosses a threshold, an automated alert (e.g., email, Slack notification) would be triggered to the MLOps team.
    *   **Data Drift Monitoring:**
        *   **Input Feature Drift:** Monitor the distributions of key input features (e.g., monthly footfall, competition density, distribution of promotion types offered) in the new incoming data compared to the training data. Significant shifts (e.g., using statistical tests like KS-test, Earth Mover's Distance) could indicate that the data the model was trained on no longer represents the current reality.
        *   **Target Variable Drift:** Monitor the distribution of the actual 'items sold' over time. A significant change could indicate a shift in underlying sales patterns that the model isn't capturing.
    *   **Concept Drift Monitoring:**
        *   **Relationship Monitoring:** This is harder to detect directly but can be inferred from performance degradation. If the relationship between features and the target changes (e.g., a promotion that used to be highly effective is no longer so), the model's predictions will suffer even if input data distributions haven't drastically changed.
    *   **Retraining Strategy:**
        *   **Trigger Conditions:** Retraining would be triggered either manually (e.g., quarterly review, business changes) or automatically when monitoring alerts indicate significant performance degradation or data/concept drift.
        *   **Retraining Process:** The retraining process would typically involve using an updated and larger dataset (e.g., the last 3-4 years of data) to retrain the model. Hyperparameters might also be re-tuned. The new model would then undergo a rigorous validation process (e.g., A/B testing, shadow deployment) before replacing the old one. Model versioning would be crucial here to manage deployments and rollbacks.