# With the rise of digital marketing led by tools including Google Analytics, Google Adwords, 
# and Facebook Ads, a key competitive advantage for businesses is using A/B testing to 
# determine effects of digital marketing efforts. 

# This is why A/B testing is a huge benefit. A/B Testing enables us to determine whether 
# changes in landing pages, popup forms, article titles, and other digital marketing decisions 
# improve conversion rates and ultimately customer purchasing behavior. A successful A/B 
# Testing strategy can lead to massive gains - more satisfied users, more engagement, and more 
# sales

# A major issue with traditional, statistical-inference approaches to A/B Testing is that it 
# only compares 2 variables - an experiment/control to an outcome. The problem is that customer 
# behavior is vastly more complex than this. Customers take different paths, spend different 
# amounts of time on the site, come from different backgrounds (age, gender, interests), and 
# more.


# A/B Testing is a tried-and-true method commonly performed using a traditional statistical inference approach grounded in a hypothesis test (e.g. t-test, z-score, chi-squared test). In plain English, 2 tests are run in parallel:
#   
# Treatment Group (Group A) - This group is exposed to the new web page, popup form, etc.
# 
# Control Group (Group B) - This group experiences no change from the current setup.



# Users have different characteristics: Different ages, genders, new vs returning, etc
# 
# Users spend different amounts of time on the website: Some hit the page right away, others spend more time on the site
# 
# Users are find your website differently: Some come from email or newsletters, others from web searches, others from social media
# 
# Users take different paths: Users take actions on the website going to different pages prior to being confronted with the event and goal



# Core packages
library(tidyverse)
library(tidyquant)

# Modeling packages
library(parsnip)
library(recipes)
library(rsample)
library(yardstick)
library(broom)

# Connector packages
library(rpart)
library(rpart.plot)
library(xgboost)


setwd("D:\\AB Testing in R")


library(help="parsnip")

?set_engine




# 3.3 Import the Data

# Import data
control_tbl <- read_csv("control_data.csv")
experiment_tbl <- read_csv("experiment_data.csv")



# 3.4 Investigate the Data
control_tbl %>% head(5)


# 3.5.1 Check for Missing Data
control_tbl %>%
  map_df(~ sum(is.na(.))) %>%
  gather(key = "feature", value = "missing_count") %>%
  arrange(desc(missing_count))


experiment_tbl %>% 
  map_df(~ sum(is.na(.))) %>%
  gather(key = "feature", value = "missing_count") %>%
  arrange(desc(missing_count))


control_tbl %>%
  filter(is.na(Enrollments))


# 3.6 Format Data
# Now that we understand the data, let's put it into the format we can use for modeling. We'll do the following:
#   
# Combine the control_tbl and experiment_tbl, adding an "id" column indicating if the data was part of the experiment or not
# Add a "row_id" column to help for tracking which rows are selected for training and testing in the modeling section
# Create a "Day of Week" feature from the "Date" column
# Drop the unnecessary "Date" column and the "Payments" column
# Handle the missing data (NA) by removing these rows.
# Shuffle the rows to mix the data up for learning
# Reorganize the columns


set.seed(123)
data_formatted_tbl <- control_tbl %>%
  
  # Combine with Experiment data
  bind_rows(experiment_tbl, .id = "Experiment") %>%
  mutate(Experiment = as.numeric(Experiment) - 1) %>%
  
  # Add row id
  mutate(row_id = row_number()) %>%
  
  # Create a Day of Week feature
  mutate(DOW = str_sub(Date, start = 1, end = 3) %>% 
           factor(levels = c("Sun", "Mon", "Tue", "Wed", 
                             "Thu", "Fri", "Sat"))
  ) %>%
  select(-Date, -Payments) %>%
  
  # Remove missing data
  filter(!is.na(Enrollments)) %>%
  
  # Shuffle the data (note that set.seed is used to make reproducible)
  sample_frac(size = 1) %>%
  
  # Reorganize columns
  select(row_id, Enrollments, Experiment, everything())

data_formatted_tbl %>% glimpse()



# 3.7 Training and Testing Sets
set.seed(123)
split_obj <- data_formatted_tbl %>%
  initial_split(prop = 0.8, strata = "Experiment")

train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)




#3.8.1 Linear Regression (Baseline)

model_01_lm <- linear_reg("regression") %>%
  set_engine("lm") %>%
  fit(Enrollments ~ ., data = train_tbl %>% select(-row_id))


model_01_lm %>%
  predict(new_data = test_tbl) %>%
  bind_cols(test_tbl %>% select(Enrollments)) %>%
  metrics(truth = Enrollments, estimate = .pred) %>%
  knitr::kable()



model_01_lm %>%
  # Format Data
  predict(test_tbl) %>%
  bind_cols(test_tbl %>% select(Enrollments)) %>%
  mutate(observation = row_number() %>% as.character()) %>%
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%
  
  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) +
  geom_point() +
  expand_limits(y = 0) +
  theme_tq() +
  scale_color_tq() +
  labs(title = "Enrollments: Prediction vs Actual",
       subtitle = "Model 01: Linear Regression (Baseline)")



linear_regression_model_terms_tbl <- model_01_lm$fit %>%
  tidy() %>%
  arrange(p.value) %>%
  mutate(term = as_factor(term) %>% fct_rev()) 


# knitr::kable() used for pretty tables
linear_regression_model_terms_tbl %>% knitr::kable()


linear_regression_model_terms_tbl %>%
  ggplot(aes(x = p.value, y = term)) +
  geom_point(color = "#2C3E50") +
  geom_vline(xintercept = 0.05, linetype = 2, color = "red") +
  theme_tq() +
  labs(title = "Feature Importance",
       subtitle = "Model 01: Linear Regression (Baseline)")



model_02_decision_tree <- decision_tree(
  mode = "regression",
  cost_complexity = 0.001, 
  tree_depth = 5, 
  min_n = 4) %>%
  set_engine("rpart") %>%
  fit(Enrollments ~ ., data = train_tbl %>% select(-row_id))


# knitr::kable() used for pretty tables
model_02_decision_tree$fit %>%
  rpart.plot(
    roundint = FALSE, 
    cex = 0.8, 
    fallen.leaves = TRUE,
    extra = 101, 
    main = "Model 02: Decision Tree")



set.seed(123)
model_03_xgboost <- boost_tree(
  mode = "regression",
  mtry = 100, 
  trees = 1000, 
  min_n = 8, 
  tree_depth = 6, 
  learn_rate = 0.2, 
  loss_reduction = 0.01, 
  sample_size = 1) %>%
  set_engine("xgboost") %>%
  fit(Enrollments ~ ., data = train_tbl %>% select(-row_id))


# Key Points:
#   
# The XGBoost model error has dropped to +/-11 Enrollments.
# The XGBoost shows that Experiment provides an information gain of 7%
# The XGBoost model tells a story that Udacity should be focusing on Page Views and secondarily Clicks to maintain or increase Enrollments. The features drive the system.

