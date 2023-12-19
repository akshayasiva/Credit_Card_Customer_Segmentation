ccd.df <- read.csv("ccd.csv")
str(ccd.df)
summary(ccd.df)
cor(ccd.df)
ccd.df_no_missing <- na.omit(ccd.df)
sum(is.na(ccd.df))

##histogram of all the columns
hist(ccd.df$Avg_Credit_Limit, 
     col = "skyblue",           
     border = "black",          
     xlab = "Average Credit Limit",   
     ylab = "Frequency",        
     main = "Histogram of Average Credit Limit")

hist(ccd.df$Total_Credit_Cards,
     col = "skyblue",            
     border = "black",            
     xlab = "Total Credit Cards",   
     ylab = "Frequency",          
     main = "Histogram of Total Credit Cards")

hist(ccd.df$Total_visits_bank, 
     col = "skyblue",           
     border = "black",           
     xlab = "Total Visits Bank",   
     ylab = "Frequency",          
     main = "Histogram of Total Visits Bank")

hist(ccd.df$Total_visits_online, 
     col = "skyblue",           
     border = "black",            
     xlab = "Total Visits Online",   
     ylab = "Frequency",          
     main = "Histogram of Total Visits Online")

hist(ccd.df$Total_calls_made, 
     col = "skyblue",             
     border = "black",            
     xlab = "Total Calls Made",   
     ylab = "Frequency",          
     main = "Histogram of Total Calls Made")

## k means clustering
library(cluster)
library(ggplot2)
library(factoextra)
ccd_clustering.df <- read.csv("ccd.csv")
ccd_clustering.df <- ccd_clustering.df[, -c(1, 2)]

d <- dist(ccd_clustering.df, method = "euclidean")
ccd_clustering.df.norm <- sapply(ccd_clustering.df, scale)

row.names(ccd_clustering.df.norm) <- row.names(ccd_clustering.df) 
d.norm <- dist(ccd_clustering.df.norm,method = "euclidean")

set.seed(42)
km <- kmeans(ccd_clustering.df.norm, 5)
km$cluster
km$centers
km$withinss
km$size

fviz_cluster(km, ccd_clustering.df.norm, ellipse.type ="euclid",  ggtheme = theme_minimal())

# plot an empty scatter plot
plot(c(0), xaxt = 'n', ylab = "", type = "l", main = "Customer segmentation of Credit card data",
     ylim = c(min(km$centers), max(km$centers)), xlim = c(0, 5))

# label x-axes
axis(1, at = c(1:5), labels = names(ccd_clustering.df))

# plot centroids
for (i in c(1:5))
  lines(km$centers[i,], lty = i, lwd = 2, col = ifelse(i %in% c(1, 3, 5),
                                                       "black", "dark grey"))

# name clusters
text(x =0.5, y = km$centers[, 1], labels = paste("Cluster",c(1:5)))
----------------------------
##Decision Tree

library(rpart)
library(rpart.plot)
library(caret)

ccd.df <- read.csv("ccd.csv")
ccd.df <- ccd.df[ , -c(1, 2)]  # Drop Sl no and customer key columns.
threshold_Avg_Credit_Limit <- 85000
threshold_Total_Credit_Cards <- 11
threshold_Total_calls_made <- 1
defaulter_conditions <- (ccd.df$Avg_Credit_Limit >= threshold_Avg_Credit_Limit &
                           ccd.df$Total_Credit_Cards <= threshold_Total_Credit_Cards &
                           ccd.df$Total_calls_made  > threshold_Total_calls_made )
ccd.df$IsDefaulter <- ifelse(defaulter_conditions, 1, 0)
# partition
set.seed(1)  
train.index <- sample(c(1:dim(ccd.df)[1]), dim(ccd.df)[1]*0.52)  
train.df <- ccd.df[train.index, ]
valid.df <- ccd.df[-train.index, ]
default.ct <- rpart(IsDefaulter ~ ., data = train.df ,method = "class")
prp(default.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = 10)
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])

default.info.ct <- rpart(IsDefaulter ~ ., data = train.df, parms = list(split = 'information'), method = "class")
prp(default.info.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -10)
length(default.info.ct$frame$var[default.info.ct$frame$var == "<leaf>"])

deeper.ct <- rpart(IsDefaulter ~ ., data = train.df, method = "class", cp = 0.01, minsplit = 20)
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))  
default.ct.point.pred.train <- predict(default.ct,train.df,type = "class")
confusionMatrix(default.ct.point.pred.train, as.factor(train.df$IsDefaulter))

### repeat the code for the validation set, and the deeper tree

default.ct.point.pred.valid <- predict(default.ct,valid.df,type = "class")
confusionMatrix(default.ct.point.pred.valid, as.factor(valid.df$IsDefaulter))

##logistic regression
  
library(caTools)  # For sample.split function
library(caret)    # For confusionMatrix function

# Make sure 'IsDefaulter' is a factor
train.df$IsDefaulter <- as.factor(train.df$IsDefaulter)
valid.df$IsDefaulter <- as.factor(valid.df$IsDefaulter)

# Logistic Regression Model
logistic_model <- glm(IsDefaulter ~ Avg_Credit_Limit + Total_Credit_Cards + Total_calls_made+
                        Total_visits_bank+Total_visits_online, data = train.df, family = binomial)
summary(logistic_model)
# Predictions on the training set
train.predictions <- predict(logistic_model, type = "response", newdata = train.df)
train.predictions <- ifelse(train.predictions > 0.52, 1, 0)

# Confusion Matrix for Training set
confusionMatrix(factor(train.predictions), train.df$IsDefaulter)

# Predictions on the validation set
valid.predictions <- predict(logistic_model, type = "response", newdata = valid.df)
valid.predictions <- ifelse(valid.predictions > 0.52, 1, 0)

# Confusion Matrix for Validation set
confusionMatrix(factor(valid.predictions), valid.df$IsDefaulter)

# Calculate the proportion of defaulters
proportion_defaulters <- mean(ccd.df$IsDefaulter)
cat("Proportion of defaulters in the dataset:", proportion_defaulters)

# Calculate average credit limit for defaulters and non-defaulters
avg_credit_limit_defaulters <- mean(ccd.df$Avg_Credit_Limit[ccd.df$IsDefaulter == 1])
avg_credit_limit_non_defaulters <- mean(ccd.df$Avg_Credit_Limit[ccd.df$IsDefaulter == 0])

cat("Average credit limit for defaulters:", avg_credit_limit_defaulters, "\n")
cat("Average credit limit for non-defaulters:", avg_credit_limit_non_defaulters)

# Create a bar plot
# Calculate counts of defaulters and non-defaulters
counts <- table(ccd.df$IsDefaulter)

# Create a bar plot
barplot(counts, main = "Count of Defaulters vs. Non-Defaulters", 
        xlab = "IsDefaulter", ylab = "Count", col = c("blue", "red"), legend = names(counts))

# Load necessary libraries
library(pROC)

# Decision Tree ROC curve and AUC
default.ct.point.pred.valid <- predict(default.ct, valid.df, type = "prob")[, 2]  # Probability of being in class 1
roc_tree <- roc(response = valid.df$IsDefaulter, predictor = default.ct.point.pred.valid)
auc_tree <- auc(roc_tree)

# Logistic Regression ROC curve and AUC
valid.df$IsDefaulter <- as.numeric(as.character(valid.df$IsDefaulter))  # Ensure the response variable is numeric
logit_pred_valid <- predict(logistic_model, newdata = valid.df, type = "response")
roc_logit <- roc(response = valid.df$IsDefaulter, predictor = logit_pred_valid)
auc_logit <- auc(roc_logit)

plot(roc_tree, col = "blue", main = "ROC Curves", col.main = "black", lwd = 2)
lines(roc_logit, col = "red", lwd = 2)
legend("bottomright", legend = c("Decision Tree", "Logistic Regression"), col = c("blue", "red"), lwd = 2)
# Summarize ROC index (AUC)
auc_dt <- auc(roc_tree)
auc_logit <- auc(roc_logit)
cat("ROC Index (AUC) for Decision Tree:", auc_dt, "\n")
cat("ROC Index (AUC) for Logistic Regression:", auc_logit, "\n")