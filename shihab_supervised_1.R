###############################################################################
# PROJECT DETAILS


#------------------------------------------------------------------------------
# ADMINSTRATIVE

# Name:         Shihab Hamati
# Matricola:    985941

# Module:       Statistical Learning
# Exam Date:    03 Nov 2022

# Part 1:       Supervised Learning


#------------------------------------------------------------------------------
# REFERENCES

# Dataset:      "Early stage diabetes risk prediction dataset"
# Dataset date: 12 Jul 2020
# Link:         https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.#

# Description:  
# - A dataset consisting of questionnaire reponses from 520 patients, approved by a doctor
# - Indicators collected are all of a physiological nature


#------------------------------------------------------------------------------
# PERSONAL MOTIVATION

# - Diabetes is a prevalent disease in my country and my extended family

# - Models would allow non-doctors (including other medical personnel as well as family and friends of concerned patient)
#   to look out for the most important physiological diabetes red flags 
#   to seek prompt medical evaluation rather than let early stage diabetes go 
#   undetected (which causes damage and becomes harder to control at later stages)

# - This is especially helpful in underdeveloped area where access to home devices
#   is not common or easy



###############################################################################
# LIBRARIES

library(dplyr)
library(DataExplorer)
library(ggplot2)
library(gridExtra)
library(matrixStats)
library(ggpubr)
library(caret)
library(pROC)
library(MASS)
library(rpart)
library(rpart.plot)
library(randomForest)
library(reshape2)
library(ggbreak) 



###############################################################################
# DATA SETUP

# Download dataset directly from online source
data_url= "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
data <- read.csv(data_url, header=TRUE, stringsAsFactors = TRUE)

# Option to read data from user selected local destination
#data <- read.csv(choose.files(), header=TRUE, stringsAsFactors = TRUE)

# view summary of data
summary(data)
str(data)
head(data)

colnames(data)
colnames(data)[17] <- "Diabetic"

# check for missing data
sum(is.na(data))

# quickly create report to explore data
# create_report(data)


#------------------------------------------------------------------------------
# DESCRIPTION
# - Polyuria    : excessive urination (frequency or volume)
# - Polydipsia  : excessive thirst
# - Polyphagia  : excessive eating
# - Paresis     : muscular weakness (partial)
# - Alopecia    : bodily hair loss



###############################################################################
# EXPLORATORY DATA ANALYSIS (EDA)


#------------------------------------------------------------------------------
# RESPONSE VARIABLE
p0 <- ggplot(data, aes(x = Diabetic)) +
  geom_bar(aes(fill = Diabetic)) +
  geom_text(aes(y = ..count.., 
                label = paste0(round(prop.table(..count..),3) * 100, '%')), 
            stat = 'count') +
  ggtitle("Distribution of Response Variable in Dataset") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")
p0


#------------------------------------------------------------------------------
# NUMERIC FEATURE
p1a <- ggplot(data = data, aes(x = Age)) +
  geom_histogram(binwidth = 5) +
  theme_minimal()

p1b <- ggplot(data = data, aes(x = Age, group = Diabetic, fill = Diabetic)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 5) +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p1c <- ggplot(data = data, aes(x = Diabetic, y = Age)) +
  geom_boxplot() +
  theme_minimal()

# Statistical summary of Age, grouped by Diabetes status
with(data, aggregate(Age, list(Diabetic = Diabetic), FUN = summary)) 

p1d <- ggline(data, x = "Diabetic", y = "Age", add = "mean_se") +
  theme_minimal()

# Anova
aov_age <- aov(Age ~ Diabetic, data = data)
summary(aov_age) # p-value < 0.001 indicating significant difference

# Display Plots
grid.arrange(p1a, p1b, p1c, p1d, ncol=2, top = "Age")

# Explore normality
qqnorm(data$Age, pch = 20, frame = FALSE)
qqline(data$Age, col="red", lwd = 2)


#------------------------------------------------------------------------------
# CATEGORICAL FEATURES

#..............................................................................
# Overall characteristics

p2 <- ggplot(data, aes(x = Gender, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p3 <- ggplot(data, aes(x = Obesity, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p4 <- ggplot(data, aes(x = sudden.weight.loss, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Sudden Weight Loss") +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

#..............................................................................
# Commonly named symptoms

p5 <- ggplot(data, aes(x = weakness, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Weakness") +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p6 <- ggplot(data, aes(x = muscle.stiffness, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Muscle Stiffness") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p7 <- ggplot(data, aes(x = visual.blurring, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Visual Blurring") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p8 <- ggplot(data, aes(x = Itching, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p9 <- ggplot(data, aes(x = Irritability, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p10 <- ggplot(data, aes(x = delayed.healing, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Delayed Healing") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

#..............................................................................
# Medically named symptoms

p11 <- ggplot(data, aes(x = Polyuria, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p12 <- ggplot(data, aes(x = Polydipsia, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p13 <- ggplot(data, aes(x = Polyphagia, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p14 <- ggplot(data, aes(x = Genital.thrush, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Genital Thrush") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p15 <- ggplot(data, aes(x = partial.paresis, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") + 
  xlab("Partial Paresis") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p16 <- ggplot(data, aes(x = Alopecia, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

# Display Plots

grid.arrange(p2, p3, p4, 
             ncol=3, top = "Overall Characteristic Traits")

grid.arrange(p5, p6, p7, p8, p9, p10, 
             ncol=3, top = "Commonly Named Symptoms")

grid.arrange(p11, p12, p13, p14, p15, p16, 
             ncol=3, top = "Medically Named Symptoms")



#------------------------------------------------------------------------------
# FEATURE ENGINEERING

# Explore relation of how many indicators exist with outcome
# create new column counting indicators for each row
CountIndicators <- 
  rowCounts(as.matrix(data), cols = colnames(data)[3:16], value = "Yes")

data_ci <- data.frame(CountIndicators, data["Diabetic"]) # separate df

# Statistical summary of Count of Indicators, grouped by Diabetes status
with(data_ci, aggregate(CountIndicators, 
                        list(Diabetic = Diabetic), FUN = summary)) 

# Plots
p17a <- ggplot(data = data_ci, 
               aes(x = CountIndicators, group = Diabetic, fill = Diabetic)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 1) +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p17b <- ggplot(data, aes(x = CountIndicators, fill = Diabetic)) + 
  geom_bar(position = "fill") + 
  ylab("Proportion") +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p17c <- ggplot(data = data_ci, aes(x = Diabetic, y = CountIndicators)) +
  geom_boxplot()  +
  theme_minimal()

p17d <- ggline(data_ci, 
               x = "Diabetic", y = "CountIndicators", add = "mean_se") +
  theme_minimal()

grid.arrange(p17a, p17b, p17c, p17d, ncol=2, top = "Quantity of Symptoms")

# Statistical values appears different for Diabetics vs Non-Diabetic
# Test if the mean count of indicators are statistically different

# Anova
aov_ci <- aov(CountIndicators ~ Diabetic, data = data)
summary(aov_ci) # p-value < 0.001 indicating significant difference

# This feature will not be provided to the models, so it was kept in its own df
# All models are able to account for it in some way, and it is of interest 
# to explore the explanatory power of each physiological feature,
# especially in importance plots in RF



###############################################################################
# MODELS

# Splitting Datasets
set.seed(1234)
split_train_test <- createDataPartition(data$Diabetic,p=0.7,list=FALSE)
dtrain<- data[split_train_test,]
dtest<- data[-split_train_test,]


#------------------------------------------------------------------------------
# LOGISTIC REGRESSION

lr_fit <- glm(Diabetic ~ ., data=dtrain, family=binomial(link='logit'))
summary(lr_fit)

# Confusion Matrices and Accuracies for LR
# Train set
lr_prob_dtrain <- predict(lr_fit, dtrain, type="response")
lr_pred_dtrain <- ifelse(lr_prob_dtrain > 0.5, "Positive", "Negative")
table(Predicted = lr_pred_dtrain, Actual = dtrain$Diabetic)
mean(lr_pred_dtrain == dtrain$Diabetic)

# confirm with built-in function
cm_lr_dtrain <- confusionMatrix(
  as.factor(lr_pred_dtrain),
  as.factor(dtrain$Diabetic),
  positive = "Positive" 
)
cm_lr_dtrain

# Test set
lr_prob_dtest <- predict(lr_fit, dtest, type="response")
lr_pred_dtest <- ifelse(lr_prob_dtest > 0.5, "Positive", "Negative")
table(Predicted = lr_pred_dtest, Actual = dtest$Diabetic)
mean(lr_pred_dtest == dtest$Diabetic)

# confirm with built-in function
cm_lr_dtest <- confusionMatrix(
  as.factor(lr_pred_dtest),
  as.factor(dtest$Diabetic),
  positive = "Positive" 
)
cm_lr_dtest

# ROC Curve
test_roc = roc(dtest$Diabetic ~ lr_prob_dtest, plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)

# Explore threshold
lr_thresholds <- c()
lr_sensitivities <- c()
lr_accuracies <- c()

for(t in 1:99){
  lr_pred_t <- ifelse(lr_prob_dtest > t/100.0, "Positive", "Negative")
  cm_t <- table(Predicted = lr_pred_t, Actual = dtest$Diabetic)
  lr_thresholds <- append(lr_thresholds, t/100.0)
  lr_sensitivities <- append(lr_sensitivities, 
                             sensitivity(cm_t, positive = "Positive"))
  lr_accuracies <- append(lr_accuracies, mean(lr_pred_t == dtest$Diabetic))
}

# Plot changes in Sensitivity (correct positives) and Accuracy
p18 <- ggplot(data=data.frame(lr_thresholds, lr_sensitivities),
              aes(x = lr_thresholds, y = lr_sensitivities)) +
  geom_line() + 
  labs(x = 'Thresholds',y='Sensitivity') +
  theme_minimal()

p19 <- ggplot(data=data.frame(lr_thresholds, lr_accuracies),
              aes(x = lr_thresholds, y = lr_accuracies)) +
  geom_line() + 
  labs(x = 'Thresholds',y='Accuracy') +
  theme_minimal()

grid.arrange(p18, p19, ncol=1) 

# Option 1: Optimization of Sensitivity
max(lr_sensitivities)
lr_thresholds[which(lr_sensitivities == max(lr_sensitivities))]
lr_accuracies[which(lr_sensitivities == max(lr_sensitivities))]

# the optimum sensitivity is achieved at t <= 0.09
# the best accuracy achievable in this range is 85.9% at t = 0.09

lr_sensitivities[23:36]
lr_accuracies[23:36]

# a good balance between both metrics could be 0.23<= t <=0.36
# it increases sensitivity (from 92.7% to 94.79%)
# without lowering accuracy at all (from default 87.8% at t = 0.5)
# best sensitivity is achieved at low thresholds but accuracy plunges alot

# Option 2: Optimization of Accuracy
max(lr_accuracies)
lr_thresholds[which(lr_accuracies == max(lr_accuracies))]
lr_sensitivities[which(lr_accuracies == max(lr_accuracies))]

# another good point is 0.38 <= t <= 0.4
# it increases test accuracy from default t = 0.5 (from 87.8% to 88.5%)
# while also increasing test sensitivity (from 92.7% to 93.75%) - lucky bonus

# Conclusion of LR Threshold
# The default t = 0.5 is good, but not optimum in either scenarios, hence:
# To optimize sensitivity (and luckily without loss of acc): t = 0.23-0.36
# To optimize accuracy (and luckily sensitivity in this case): t = 0.38-0.40

# Re-fit Logistic Regression 
# using option 2, since acc is the metric used to compare the different models

t = 0.4

lr_pred_dtrain_t <- ifelse(lr_prob_dtrain > t, "Positive", "Negative")
lr_pred_dtest_t <- ifelse(lr_prob_dtest > t, "Positive", "Negative")

cm_lr_dtrain_t <- confusionMatrix(
  as.factor(lr_pred_dtrain_t),
  as.factor(dtrain$Diabetic),
  positive = "Positive" 
)
cm_lr_dtrain_t

cm_lr_dtest_t <- confusionMatrix(
  as.factor(lr_pred_dtest_t),
  as.factor(dtest$Diabetic),
  positive = "Positive" 
)
cm_lr_dtest_t

#------------------------------------------------------------------------------
# LINEAR DISCRIMINANT ANALYSIS (LDA)

lda_fit = lda(Diabetic ~ ., data=dtrain)
lda_fit
plot(lda_fit)

# Confusion Matrices and Accuracies of LDA

# Training dataset
lda_pred_dtrain = predict(lda_fit, dtrain)$class
table(lda_pred_dtrain, dtrain$Diabetic)
mean(lda_pred_dtrain == dtrain$Diabetic)

# confirm with built-in function
cm_lda_dtrain <- confusionMatrix(
  as.factor(lda_pred_dtrain),
  as.factor(dtrain$Diabetic),
  positive = "Positive" 
)
cm_lda_dtrain

# Test dataset
lda_pred_dtest = predict(lda_fit, dtest)$class
table(lda_pred_dtest, dtest$Diabetic)
mean(lda_pred_dtest == dtest$Diabetic)

# confirm with built-in function
cm_lda_dtest <-confusionMatrix(
  as.factor(lda_pred_dtest),
  as.factor(dtest$Diabetic),
  positive = "Positive" 
)
cm_lda_dtest


#------------------------------------------------------------------------------
# DECISION TREE
tree <- rpart(formula = Diabetic ~ ., data=dtrain)
printcp(tree)
rpart.plot(tree, type=3, box.palette="YlGn")

tree_pred_dtrain = predict(tree, dtrain, type="class")
tree_pred_dtest = predict(tree, dtest, type="class")

cm_tree_dtrain <- confusionMatrix(
  as.factor(tree_pred_dtrain),
  as.factor(dtrain$Diabetic),
  positive = "Positive" 
)
cm_tree_dtrain

cm_tree_dtest <- confusionMatrix(
  as.factor(tree_pred_dtest),
  as.factor(dtest$Diabetic),
  positive = "Positive" 
)
cm_tree_dtest

plotcp(tree) # to choose cp corresponding to lowest X-val relative error


#------------------------------------------------------------------------------
# PRUNED DECISION TREE

ptree <- prune(tree, cp = 0.03)
printcp(ptree)
rpart.plot(ptree, type=3, box.palette="YlGn")

ptree_pred_dtrain = predict(ptree, dtrain, type="class")
ptree_pred_dtest = predict(ptree, dtest, type="class")

cm_ptree_dtrain <- confusionMatrix(
  as.factor(ptree_pred_dtrain),
  as.factor(dtrain$Diabetic),
  positive = "Positive" 
)
cm_ptree_dtrain

cm_ptree_dtest <- confusionMatrix(
  as.factor(ptree_pred_dtest),
  as.factor(dtest$Diabetic),
  positive = "Positive" 
)
cm_ptree_dtest



#------------------------------------------------------------------------------
# RANDOM FOREST

rf = randomForest(Diabetic ~ ., data = dtrain, 
                  ntree = 50, mtry = 3, importance = TRUE)

varImpPlot(rf, bg = "black", 
           main = "Variable Importance Plot (Random Forest)")

rf_pred_dtrain <- predict(rf, dtrain)
rf_pred_dtest <- predict(rf, dtest)

cm_rf_dtrain <- confusionMatrix(
  as.factor(rf_pred_dtrain),
  as.factor(dtrain$Diabetic),
  positive = "Positive" 
)
cm_rf_dtrain

cm_rf_dtest <- confusionMatrix(
  as.factor(rf_pred_dtest),
  as.factor(dtest$Diabetic),
  positive = "Positive" 
)
cm_rf_dtest

# RF models are not prone to overfitting
# So, the larger gap between train acc (100%) and test acc (94.23%)
# does not indicate a potential to improve test acc (like in other models)

# Also, RF achieves the highest train and test accuracies anyway

plot(dtrain$Diabetic, rf_pred_dtrain)
plot(dtest$Diabetic, rf_pred_dtest)



###############################################################################
# SUMMARY

abbr <- c("1a. LR", "1b. LR(t=0.4)", "2. LDA", "3a. DT", "3b. PDT", "4. RF")

fullname <- c("Logistic Regression",
              "Logistic Regression(thresh=0.4)",
              "Linear Discrimnant Analysis", 
              "Decision Tree", 
              "Pruned Decision Tree", 
              "Random Forest")

# Retrieve values from stored confusion matrices for accuracy and sensitivity
# for both the training and test datasets

acc_train <- c(cm_lr_dtrain$overall["Accuracy"],
               cm_lr_dtrain_t$overall["Accuracy"],
               cm_lda_dtrain$overall["Accuracy"],
               cm_tree_dtrain$overall["Accuracy"],
               cm_ptree_dtrain$overall["Accuracy"],
               cm_rf_dtrain$overall["Accuracy"])

acc_test <- c(cm_lr_dtest$overall["Accuracy"],
              cm_lr_dtest_t$overall["Accuracy"],
              cm_lda_dtest$overall["Accuracy"],
              cm_tree_dtest$overall["Accuracy"],
              cm_ptree_dtest$overall["Accuracy"],
              cm_rf_dtest$overall["Accuracy"])

snsv_train <- c(cm_lr_dtrain$byClass["Sensitivity"],
                cm_lr_dtrain_t$byClass["Sensitivity"],
                cm_lda_dtrain$byClass["Sensitivity"],
                cm_tree_dtrain$byClass["Sensitivity"],
                cm_ptree_dtrain$byClass["Sensitivity"],
                cm_rf_dtrain$byClass["Sensitivity"])

snsv_test <- c(cm_lr_dtest$byClass["Sensitivity"],
               cm_lr_dtest_t$byClass["Sensitivity"],
               cm_lda_dtest$byClass["Sensitivity"],
               cm_tree_dtest$byClass["Sensitivity"],
               cm_ptree_dtest$byClass["Sensitivity"],
               cm_rf_dtest$byClass["Sensitivity"])


# Manipulate dataframes for plotting purposes
acc_summary <- data.frame(abbr, fullname, acc_train, acc_test)
colnames(acc_summary)[3:4] <- c("Train", "Test")
acc_summary <- melt(acc_summary)

snsv_summary <- data.frame(abbr, fullname, snsv_train, snsv_test)
colnames(snsv_summary)[3:4] <- c("Train", "Test")
snsv_summary <- melt(snsv_summary)


#------------------------------------------------------------------------------
# MODELS COMPARISON

abbr_ticks <- c("LR", "LR(t=0.4)", "LDA", "DT", "PDT", "RF")

p20 <- ggplot(data = acc_summary, aes(x = abbr, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  coord_cartesian(ylim = c(.7, 1)) + 
  labs(title = "Comparison of Models - Accuracy", x = "Model", y = "Accuracy", 
       fill = "Data Split") + 
  scale_x_discrete(labels = abbr_ticks) +
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

p21 <- ggplot(data = snsv_summary, aes(x = abbr, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  coord_cartesian(ylim = c(.7, 1)) + 
  labs(title = "Comparison of Models - Sensitivity", x = "Model" , y = "Sensitivity", 
       fill = "Data Split") + 
  scale_x_discrete(labels = abbr_ticks) + 
  theme_minimal() + 
  scale_fill_brewer(palette = "YlGn")

grid.arrange(p20, p21, ncol=1) 



###############################################################################
# END